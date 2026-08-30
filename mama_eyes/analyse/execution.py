"""Orchestration des analyses : detection, suivi, statistiques.

Regroupe les etapes de la couche mesure en deux operations utilisables depuis
la CLI comme depuis l'interface :

- `executer_detection` : segmente les druses sur chaque seance recalee ;
- `executer_evolution` : apparie les detections dans le temps, applique la
  regle de confirmation et produit la serie d'evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Parametres
from ..domain import Reperes
from ..pretraitement.reperes import proposer_reperes
from ..pretraitement.vaisseaux import carte_vaisseaux, masque_vaisseaux
from .differentiel import comparer, objets_de_masque
from .druses import detecter_druses
from .metriques import PointEvolution, SerieEvolution, correler_nettete, surface_en_mm2
from .suivi import (
    compter_par_date,
    construire_pistes,
    detecter_fusions_scissions,
    evenements_confirmes,
)

ECHELLE_VAISSEAUX = 0.5


@dataclass
class ResultatDetection:
    oeil: str
    analyse_id: int | None = None
    seances: int = 0
    druses_totales: int = 0
    messages: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.seances:
            return "aucune seance traitee"
        moyenne = self.druses_totales / self.seances
        return (
            f"{self.seances} seance(s), {self.druses_totales} detection(s) "
            f"({moyenne:.0f} par cliche en moyenne)"
        )


def masque_vasculaire_gabarit(pipeline, oeil: str, params: Parametres) -> np.ndarray | None:
    """Carte vasculaire calculee une fois sur le gabarit.

    Les images etant recalees sur ce gabarit, le masque vaut pour toute la
    serie - ce qui evite 105 filtrages de Frangi.
    """
    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        return None
    gabarit, masque_champ = gab
    f = ECHELLE_VAISSEAUX
    petit = cv2.resize(gabarit, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    m_petit = cv2.resize(
        masque_champ, (petit.shape[1], petit.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    reponse = carte_vaisseaux(petit, m_petit, params.vaisseaux, echelle=f)
    binaire = masque_vaisseaux(reponse, m_petit, params.vaisseaux)
    return cv2.resize(
        binaire,
        (masque_champ.shape[1], masque_champ.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def obtenir_reperes(pipeline, base, oeil: str, params: Parametres) -> Reperes | None:
    """Reperes saisis manuellement, ou proposition automatique a defaut.

    Le clic manuel reste le mode principal (cf. `pretraitement.reperes`) :
    la proposition automatique n'est qu'un point de depart.
    """
    reperes = base.reperes(oeil)
    if reperes is not None:
        return reperes
    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        return None
    gabarit, masque = gab
    from ..pretraitement.champ import detecter_champ

    champ = detecter_champ(gabarit, params.champ)
    proposition = proposer_reperes(gabarit, masque, champ.rayon, oeil)
    base.enregistrer_reperes(proposition)
    return proposition


def executer_detection(pipeline, base, oeil: str, rappel=None) -> ResultatDetection:
    """Segmente les druses sur chaque seance recalee et enregistre en base."""
    params = pipeline.params
    resultat = ResultatDetection(oeil=oeil)

    seances = pipeline.seances_recalees(oeil)
    if not seances:
        resultat.messages.append("aucune seance recalee : lancer 'recaler' d'abord")
        return resultat

    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        resultat.messages.append("gabarit absent")
        return resultat
    _, masque_champ = gab

    vaisseaux = masque_vasculaire_gabarit(pipeline, oeil, params)
    reperes = obtenir_reperes(pipeline, base, oeil, params)

    analyse_id = base.creer_analyse(
        oeil, "absolue", params.druses.en_dict().__repr__(), params.empreinte()
    )
    resultat.analyse_id = analyse_id

    for i, seance in enumerate(seances):
        if rappel:
            rappel("detection", i + 1, len(seances), str(seance.date))
        image = pipeline.lire_recale(oeil, seance.date)
        if image is None:
            continue
        druses, _masque = detecter_druses(
            image, masque_champ, vaisseaux, reperes, params.druses
        )
        for d in druses:
            d.seance_id = seance.id
        base.enregistrer_druses(analyse_id, seance.id, druses)
        base.enregistrer_mesure(
            analyse_id,
            seance.id,
            nombre=len(druses),
            surface_px=sum(d.surface_px for d in druses),
            surface_mm2=surface_en_mm2(sum(d.surface_px for d in druses), reperes),
        )
        resultat.seances += 1
        resultat.druses_totales += len(druses)

    return resultat


def executer_evolution(
    pipeline, base, oeil: str, methode: str = "absolue", rappel=None
) -> SerieEvolution:
    """Apparie les detections dans le temps et produit la serie d'evolution."""
    params = pipeline.params
    serie = SerieEvolution(oeil=oeil, methode=methode)

    seances = pipeline.seances_recalees(oeil)
    if len(seances) < 3:
        serie.messages.append("au moins trois seances sont necessaires")
        return serie

    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        serie.messages.append("gabarit absent")
        return serie
    _, masque_champ = gab

    vaisseaux = masque_vasculaire_gabarit(pipeline, oeil, params)
    reperes = obtenir_reperes(pipeline, base, oeil, params)

    # --- detection date par date ---
    detections = []
    dates = []
    nettetes = []
    for i, seance in enumerate(seances):
        if rappel:
            rappel("detection", i + 1, len(seances), str(seance.date))
        image = pipeline.lire_recale(oeil, seance.date)
        if image is None:
            continue
        druses, _ = detecter_druses(
            image, masque_champ, vaisseaux, reperes, params.druses
        )
        for d in druses:
            d.seance_id = seance.id
        detections.append(druses)
        dates.append(seance.date)
        actifs = seance.cliches_actifs
        nettetes.append(
            float(np.mean([c.nettete or 0.0 for c in actifs])) if actifs else None
        )

    if len(detections) < 3:
        serie.messages.append("pas assez de detections exploitables")
        return serie

    # --- appariement temporel et regle de confirmation ---
    pistes = construire_pistes(detections, params.suivi)
    evenements = evenements_confirmes(pistes, dates, params.suivi)
    fs = detecter_fusions_scissions(detections, dates, params.suivi)
    comptes = compter_par_date(evenements + fs, len(dates))

    analyse_id = base.creer_analyse(
        oeil, methode, params.en_json(), params.empreinte()
    )

    invalides = {c for c, _ in base.recalages_invalides(oeil)}
    for i, (jour, druses) in enumerate(zip(dates, detections)):
        seance = seances[i]
        surface = sum(d.surface_px for d in druses)
        point = PointEvolution(
            date=jour,
            nombre=len(druses),
            surface_px=surface,
            surface_mm2=surface_en_mm2(surface, reperes),
            apparues=comptes["apparition"][i],
            disparues=comptes["disparition"][i],
            fusions=comptes["fusion"][i],
            scissions=comptes["scission"][i],
            nettete=nettetes[i],
            recalage_valide=not all(c.id in invalides for c in seance.cliches_actifs),
        )
        serie.points.append(point)
        base.enregistrer_druses(analyse_id, seance.id, druses)
        base.enregistrer_mesure(
            analyse_id,
            seance.id,
            nombre=point.nombre,
            surface_px=point.surface_px,
            surface_mm2=point.surface_mm2,
            apparues=point.apparues,
            disparues=point.disparues,
            fusions=point.fusions,
            scissions=point.scissions,
        )

    # --- garde-fou : les evenements sortent-ils du bruit de segmentation ? ---
    # Mesure sur les suites de dates rapprochees, ou aucune evolution reelle
    # n'est possible : tout evenement confirme y est un faux positif.
    try:
        from .controle import controle_sequence

        sequences = controle_sequence(pipeline, base, oeil)
        if sequences:
            faux = sum(s.evenements_confirmes for s in sequences)
            # Rapporte aux seules dates ou la regle pouvait se declencher :
            # les bords des suites en sont exclus.
            eligibles = sum(s.dates_eligibles for s in sequences)
            serie.plancher_evenements = faux / max(eligibles, 1)
            serie.messages.append(
                f"plancher de faux positifs : "
                f"{serie.plancher_evenements:.0f} evenement(s) par date "
                f"({faux} sur {eligibles} date(s) eligible(s) en periode "
                f"rapprochee)"
            )
            if serie.evenements_credibles is False:
                serie.messages.append(
                    "ATTENTION : les apparitions et disparitions mesurees "
                    "restent sous ce plancher. Elles refletent l'instabilite "
                    "de la segmentation, pas une evolution reelle. Suivez la "
                    "surface totale, plus stable."
                )
    except (ValueError, OSError) as exc:
        serie.messages.append(f"plancher de faux positifs non calcule : {exc}")

    # --- garde-fou : la mesure suit-elle la nettete plutot que la maladie ? ---
    correlation = correler_nettete(serie)
    if correlation is not None:
        serie.messages.append(f"correlation nombre <-> nettete : {correlation:+.2f}")
        if abs(correlation) > 0.5:
            serie.messages.append(
                "ATTENTION : correlation forte avec la nettete. La tendance "
                "observee peut refleter la qualite d'image plutot que "
                "l'evolution des druses."
            )

    return serie


def comparer_dates(
    pipeline, base, oeil: str, date_a, date_b
) -> tuple[np.ndarray, object] | None:
    """Comparaison ponctuelle entre deux dates, pour l'interface."""
    params = pipeline.params
    img_a = pipeline.lire_recale(oeil, date_a)
    img_b = pipeline.lire_recale(oeil, date_b)
    gab = pipeline.lire_gabarit(oeil)
    if img_a is None or img_b is None or gab is None:
        return None
    _, masque_champ = gab
    vaisseaux = masque_vasculaire_gabarit(pipeline, oeil, params)
    diff = comparer(img_a, img_b, masque_champ, vaisseaux, params.differentiel)
    return img_b, diff


def comparer_a_reference(pipeline, base, oeil: str, jour) -> object | None:
    """Compare une date a l'image « saine » fournie manuellement."""
    params = pipeline.params
    reference = pipeline.biblio.image_reference(oeil)
    if reference is None:
        return None
    image = pipeline.lire_recale(oeil, jour)
    gab = pipeline.lire_gabarit(oeil)
    if image is None or gab is None:
        return None
    gabarit, masque_champ = gab

    # La reference fournie n'est pas forcement recalee : on l'aligne d'abord.
    from ..pretraitement.champ import detecter_champ
    from ..pretraitement.recalage import appliquer, estimer_transformation

    if reference.shape[:2] != gabarit.shape[:2]:
        reference = cv2.resize(reference, (gabarit.shape[1], gabarit.shape[0]))
    m_ref = detecter_champ(reference, params.champ).masque()
    tr = estimer_transformation(reference, gabarit, m_ref, masque_champ, params.recalage)
    if tr.valide:
        reference = appliquer(reference, tr.matrice, gabarit.shape[:2])

    vaisseaux = masque_vasculaire_gabarit(pipeline, oeil, params)
    return comparer(reference, image, masque_champ, vaisseaux, params.differentiel)
