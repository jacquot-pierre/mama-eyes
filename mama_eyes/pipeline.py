"""Orchestration de la chaine, avec cache invalide par empreinte de parametres.

Le cache est indexe par l'empreinte des parametres qui l'ont produit : changer
un reglage change l'empreinte et regenere l'artefact, ce qui evite de comparer
un jour des images produites par deux versions differentes de la chaine.

Toutes les etapes sont reconstructibles : `data/derive/` peut etre supprime
entierement sans perte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .config import Emplacements, Parametres
from .domain import QualiteRecalage, Seance
from .pretraitement.champ import Champ, detecter_champ
from .pretraitement.debruitage import debruiter
from .pretraitement.fusion import fusionner_seance
from .pretraitement.illumination import (
    StatsLab,
    corriger_vignettage,
    harmoniser_lab,
    masque_hors_disque,
    planeite,
    stats_lab,
    stats_medianes,
)
from .pretraitement.recalage import (
    Transformation,
    appliquer,
    estimer_transformation,
    raffiner_ecc,
)
from .stockage.base import Base
from .stockage.bibliotheque import Bibliotheque

ETAPE_NORMALISE = "normalise"
ETAPE_RECALE = "recale"
ETAPE_GABARIT = "gabarit"

#: Nombre de seances echantillonnees pour construire le gabarit de groupe.
#: 20 images suffisent largement a une mediane stable, et evitent de charger
#: 105 images de 18 Mo en memoire.
TAILLE_ECHANTILLON_GABARIT = 20


@dataclass
class EtatEtape:
    """Compte rendu d'une etape, pour l'interface et les journaux."""

    nom: str
    traites: int = 0
    ignores: int = 0
    echecs: int = 0
    messages: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def __str__(self) -> str:
        s = f"{self.nom}: {self.traites} traite(s)"
        if self.ignores:
            s += f", {self.ignores} deja en cache"
        if self.echecs:
            s += f", {self.echecs} ECHEC(s)"
        return s


class Pipeline:
    """Chaine complete pour un oeil."""

    def __init__(
        self,
        biblio: Bibliotheque,
        base: Base,
        params: Parametres | None = None,
    ):
        self.biblio = biblio
        self.base = base
        self.params = params or Parametres()
        self.emp: Emplacements = biblio.emp

    # -- chemins de cache ----------------------------------------------------

    def _empreinte_normalisation(self) -> str:
        charge = json.dumps(
            {
                "champ": self.params.champ.en_dict(),
                "illumination": self.params.illumination.en_dict(),
                "debruitage": self.params.debruitage.en_dict(),
            },
            sort_keys=True,
        )
        import hashlib

        return hashlib.sha256(charge.encode()).hexdigest()[:12]

    def _empreinte_recalage(self) -> str:
        import hashlib

        charge = json.dumps(
            {
                "normalisation": self._empreinte_normalisation(),
                "recalage": self.params.recalage.en_dict(),
                "vaisseaux": self.params.vaisseaux.en_dict(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(charge.encode()).hexdigest()[:12]

    def chemin_normalise(self, oeil: str, jour: date) -> Path:
        nom = f"{jour:%Y_%m_%d}_{oeil}_{self._empreinte_normalisation()}.png"
        return self.biblio.chemin_derive(oeil, ETAPE_NORMALISE, nom)

    def chemin_recale(self, oeil: str, jour: date) -> Path:
        nom = f"{jour:%Y_%m_%d}_{oeil}_{self._empreinte_recalage()}.png"
        return self.biblio.chemin_derive(oeil, ETAPE_RECALE, nom)

    def chemin_gabarit(self, oeil: str) -> Path:
        nom = f"gabarit_{oeil}_{self._empreinte_recalage()}.png"
        return self.biblio.chemin_derive(oeil, ETAPE_GABARIT, nom)

    def chemin_masque_gabarit(self, oeil: str) -> Path:
        nom = f"masque_{oeil}_{self._empreinte_recalage()}.png"
        return self.biblio.chemin_derive(oeil, ETAPE_GABARIT, nom)

    # -- etape 1 : normalisation photometrique -------------------------------

    def _preparer_cliche(
        self, chemin: Path
    ) -> tuple[np.ndarray, np.ndarray, Champ, tuple[float, float]]:
        """Lit, masque, corrige le vignettage et debruite un cliche.

        L'image n'est lue qu'une fois : la mesure d'uniformite avant / apres
        est calculee ici, tant que l'original est encore en memoire.
        """
        image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"image illisible : {chemin}")
        champ = detecter_champ(image, self.params.champ)
        masque = champ.masque()
        sigma = champ.rayon * self.params.illumination.sigma_fond_fraction
        avant = planeite(image, masque, sigma)
        corrige = corriger_vignettage(
            image, masque, champ.rayon, self.params.illumination
        )
        apres = planeite(corrige, masque, sigma)
        corrige = debruiter(corrige, self.params.debruitage)
        return corrige, masque, champ, (avant, apres)

    def normaliser(
        self, oeil: str, forcer: bool = False, rappel=None
    ) -> EtatEtape:
        """Masque du champ, vignettage, debruitage, fusion intra-seance.

        Deux passes : la premiere collecte les statistiques LAB de la serie,
        la seconde harmonise chaque seance vers leur mediane.
        """
        etat = EtatEtape("normalisation")
        seances = [s for s in self.base.seances(oeil) if s.cliches_actifs]
        if not seances:
            etat.messages.append("aucune seance a traiter")
            return etat

        a_faire = [
            s
            for s in seances
            if forcer or not self.chemin_normalise(oeil, s.date).exists()
        ]
        etat.ignores = len(seances) - len(a_faire)
        if not a_faire:
            etat.messages.append("tout est deja en cache")
            return etat

        # --- passe 1 : correction, fusion, ecriture, collecte des statistiques ---
        # Les images sont ecrites au fur et a mesure plutot que conservees en
        # memoire : 105 seances a 18 Mo tiendraient 1,9 Go en RAM.
        stats: list[StatsLab] = []
        traitees: list[tuple[Seance, Path]] = []
        planeites: list[tuple[float, float]] = []

        for i, seance in enumerate(a_faire):
            if rappel:
                rappel("passe 1/2", i + 1, len(a_faire), str(seance.date))
            images, masques, nettetes = [], [], []
            for cliche in seance.cliches_actifs:
                try:
                    img, msk, _champ, mesure = self._preparer_cliche(
                        self.biblio.chemin_cliche(cliche)
                    )
                except OSError as exc:
                    etat.echecs += 1
                    etat.messages.append(f"{seance.date} : {exc}")
                    continue
                planeites.append(mesure)
                images.append(img)
                masques.append(msk)
                nettetes.append(cliche.nettete or 0.0)

            if not images:
                continue

            fusion, masque, journal = fusionner_seance(
                images, masques, nettetes, self.params.recalage
            )
            if len(images) > 1:
                etat.messages.append(f"{seance.date} : " + " | ".join(journal))

            stats.append(stats_lab(fusion, masque))
            cible_fichier = self.chemin_normalise(oeil, seance.date)
            cv2.imwrite(str(cible_fichier), fusion)
            traitees.append((seance, cible_fichier))
            del images, masques, fusion, masque

        if not stats:
            etat.messages.append("aucune statistique collectee")
            return etat

        cible = stats_medianes(stats)

        if planeites:
            avant = float(np.mean([a for a, _ in planeites]))
            apres = float(np.mean([b for _, b in planeites]))
            etat.messages.append(
                f"uniformite du champ : {avant:.2f} % -> {apres:.2f} % "
                f"(coefficient de variation du fond, plus bas = mieux)"
            )

        # --- passe 2 : harmonisation vers la mediane de la serie ---
        if not self.params.illumination.harmoniser_lab:
            etat.traites = len(traitees)
            return etat

        for i, (seance, chemin) in enumerate(traitees):
            if rappel:
                rappel("passe 2/2", i + 1, len(traitees), str(seance.date))
            image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
            if image is None:
                etat.echecs += 1
                continue
            masque = detecter_champ(image, self.params.champ).masque()
            harmonisee = harmoniser_lab(image, masque, cible, self.params.illumination)
            cv2.imwrite(str(chemin), harmonisee)
            etat.traites += 1

        return etat

    def lire_normalise(self, oeil: str, jour: date) -> np.ndarray | None:
        chemin = self.chemin_normalise(oeil, jour)
        if not chemin.exists():
            return None
        return cv2.imread(str(chemin), cv2.IMREAD_COLOR)

    # -- etape 2 : gabarit de groupe et recalage ------------------------------

    def construire_gabarit(
        self, oeil: str, forcer: bool = False, rappel=None
    ) -> np.ndarray | None:
        """Gabarit de reference par iteration de groupe.

        Aligner tout sur un cliche median, calculer la mediane des images
        alignees, ré-aligner sur cette mediane, iterer. Le gabarit converge
        vers une anatomie moyenne et n'est biaise par aucun cliche
        particulier - contrairement au premier cliche de la serie, qui apres
        7 ans de derive en est le pire representant.
        """
        chemin = self.chemin_gabarit(oeil)
        if chemin.exists() and not forcer:
            return cv2.imread(str(chemin), cv2.IMREAD_COLOR)

        seances = [
            s
            for s in self.base.seances(oeil)
            if self.chemin_normalise(oeil, s.date).exists()
        ]
        if not seances:
            return None

        # Echantillon regulierement reparti dans le temps.
        pas = max(1, len(seances) // TAILLE_ECHANTILLON_GABARIT)
        echantillon = seances[::pas][:TAILLE_ECHANTILLON_GABARIT]

        images = []
        for s in echantillon:
            img = self.lire_normalise(oeil, s.date)
            if img is not None:
                images.append(img)
        if not images:
            return None

        # Depart : le cliche median de l'echantillon.
        gabarit = images[len(images) // 2].copy()
        forme = gabarit.shape[:2]

        for iteration in range(max(1, self.params.recalage.iterations_gabarit)):
            m_gab = detecter_champ(gabarit, self.params.champ).masque()
            empiles = []
            for j, img in enumerate(images):
                if rappel:
                    rappel(
                        f"gabarit iter {iteration + 1}", j + 1, len(images), ""
                    )
                m_img = detecter_champ(img, self.params.champ).masque()
                tr = estimer_transformation(
                    img, gabarit, m_img, m_gab, self.params.recalage
                )
                if not tr.valide:
                    continue
                empiles.append(appliquer(img, tr.matrice, forme).astype(np.float32))
            if len(empiles) >= 3:
                gabarit = np.clip(
                    np.median(np.stack(empiles, axis=0), axis=0), 0, 255
                ).astype(np.uint8)

        cv2.imwrite(str(chemin), gabarit)
        masque = detecter_champ(gabarit, self.params.champ).masque()
        cv2.imwrite(str(self.chemin_masque_gabarit(oeil)), masque)
        return gabarit

    def lire_gabarit(self, oeil: str) -> tuple[np.ndarray, np.ndarray] | None:
        c_gab = self.chemin_gabarit(oeil)
        c_msk = self.chemin_masque_gabarit(oeil)
        if not c_gab.exists():
            return None
        gabarit = cv2.imread(str(c_gab), cv2.IMREAD_COLOR)
        if c_msk.exists():
            masque = cv2.imread(str(c_msk), cv2.IMREAD_GRAYSCALE)
        else:
            masque = detecter_champ(gabarit, self.params.champ).masque()
        return gabarit, masque

    def recaler(self, oeil: str, forcer: bool = False, rappel=None) -> EtatEtape:
        """Recale chaque seance sur le gabarit et journalise la qualite.

        Un echec de recalage n'est jamais silencieux : la seance est marquee
        invalide en base et signalee dans l'interface. Une seule image mal
        recalee suffit a produire un faux pic d'apparitions puis de
        disparitions, indiscernable d'un evenement reel.
        """
        etat = EtatEtape("recalage")
        gab = self.lire_gabarit(oeil)
        if gab is None:
            etat.messages.append("gabarit absent : lancer construire_gabarit d'abord")
            return etat
        gabarit, masque_gab = gab
        forme = gabarit.shape[:2]
        empreinte = self._empreinte_recalage()

        seances = [
            s
            for s in self.base.seances(oeil)
            if self.chemin_normalise(oeil, s.date).exists()
        ]

        for i, seance in enumerate(seances):
            cible = self.chemin_recale(oeil, seance.date)
            if cible.exists() and not forcer:
                etat.ignores += 1
                continue
            if rappel:
                rappel("recalage", i + 1, len(seances), str(seance.date))

            image = self.lire_normalise(oeil, seance.date)
            if image is None:
                etat.echecs += 1
                continue
            masque = detecter_champ(image, self.params.champ).masque()

            tr = estimer_transformation(
                image, gabarit, masque, masque_gab, self.params.recalage
            )

            if tr.valide and self.params.recalage.ecc_actif:
                matrice, affine = raffiner_ecc(
                    image,
                    gabarit,
                    masque,
                    masque_gab,
                    tr.matrice,
                    self.params.recalage,
                    self.params.vaisseaux,
                )
                if affine:
                    tr = Transformation(
                        matrice, tr.inliers, tr.residu_rms, True, "sift+ecc"
                    )

            # Une seule interpolation, a partir de la transformation composee.
            recalee = appliquer(image, tr.matrice, forme)
            cv2.imwrite(str(cible), recalee)

            for cliche in seance.cliches_actifs:
                self.base.enregistrer_recalage(
                    QualiteRecalage(
                        cliche_id=cliche.id,
                        matrice=tr.matrice.tolist(),
                        inliers=tr.inliers,
                        residu_rms=(
                            tr.residu_rms if np.isfinite(tr.residu_rms) else 9999.0
                        ),
                        valide=tr.valide,
                    ),
                    empreinte,
                )

            if tr.valide:
                etat.traites += 1
            else:
                etat.echecs += 1
                etat.messages.append(f"{seance.date} : {tr}")

        return etat

    def lire_recale(self, oeil: str, jour: date) -> np.ndarray | None:
        chemin = self.chemin_recale(oeil, jour)
        if not chemin.exists():
            return None
        return cv2.imread(str(chemin), cv2.IMREAD_COLOR)

    def seances_recalees(self, oeil: str, valides_seulement: bool = True) -> list[Seance]:
        """Seances disposant d'une image recalee, triees par date."""
        resultat = []
        invalides = {c for c, _ in self.base.recalages_invalides(oeil)}
        for s in self.base.seances(oeil):
            if not self.chemin_recale(oeil, s.date).exists():
                continue
            if valides_seulement and s.cliches_actifs:
                if all(c.id in invalides for c in s.cliches_actifs):
                    continue
            resultat.append(s)
        return resultat
