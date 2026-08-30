"""Controle negatif sur les paires de cliches rapprochees.

Principe : sur un intervalle de quelques jours, aucune evolution biologique
des druses n'est possible. **Tout ce que la chaine detecte sur ces paires est
donc du bruit**, et la distribution de ce bruit donne directement le plancher
de sensibilite de l'outil.

C'est le composant de validation le plus rentable du projet : il permet de
regler les seuils sur une base objective plutot qu'a l'oeil, et sert de test
de non-regression - toute degradation introduite par un changement de
parametre ou de code s'y voit immediatement.

La serie de reference comporte 16 intervalles de 10 jours ou moins, dont une
sequence hebdomadaire continue de six cliches (septembre-octobre 2019) et une
paire a trois jours d'ecart.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from ..config import Parametres


@dataclass
class PaireControle:
    """Une comparaison entre deux dates rapprochees."""

    date_a: date
    date_b: date
    ecart_jours: int
    surface_apparue: int
    surface_disparue: int
    objets_apparus: int
    objets_disparus: int
    gain: float
    ecart_type_diff: float
    p99_diff: float
    p999_diff: float

    @property
    def surface_totale(self) -> int:
        return self.surface_apparue + self.surface_disparue

    @property
    def objets_totaux(self) -> int:
        return self.objets_apparus + self.objets_disparus


@dataclass
class PaireDetection:
    """Stabilite de la detection absolue entre deux dates rapprochees.

    Mesure le *churn* de segmentation : combien d'objets apparaissent ou
    disparaissent alors que rien n'a change. C'est le plancher de bruit de la
    voie « detection absolue + suivi », distincte de la voie differentielle.
    """

    date_a: date
    date_b: date
    ecart_jours: int
    nombre_a: int
    nombre_b: int
    surface_a: int
    surface_b: int
    non_apparies_a: int
    non_apparies_b: int

    @property
    def churn(self) -> int:
        return self.non_apparies_a + self.non_apparies_b

    @property
    def ecart_relatif_nombre(self) -> float:
        base = max(self.nombre_a, 1)
        return abs(self.nombre_b - self.nombre_a) / base

    @property
    def ecart_relatif_surface(self) -> float:
        base = max(self.surface_a, 1)
        return abs(self.surface_b - self.surface_a) / base


@dataclass
class RapportControle:
    """Synthese du controle negatif pour un oeil."""

    oeil: str
    paires: list[PaireControle] = field(default_factory=list)
    detections: list[PaireDetection] = field(default_factory=list)
    ecart_max_jours: int = 10
    messages: list[str] = field(default_factory=list)

    # -- statistiques du plancher de bruit -----------------------------------

    def _valeurs(self, attribut: str) -> list[float]:
        return [float(getattr(p, attribut)) for p in self.paires]

    def plancher_objets(self, percentile: float = 95.0) -> float:
        """Nombre d'objets de changement au-dela duquel un signal est credible."""
        valeurs = self._valeurs("objets_totaux")
        if not valeurs:
            return 0.0
        return float(np.percentile(valeurs, percentile))

    def plancher_surface(self, percentile: float = 95.0) -> float:
        valeurs = self._valeurs("surface_totale")
        if not valeurs:
            return 0.0
        return float(np.percentile(valeurs, percentile))

    def seuil_diff_suggere(self) -> float:
        """Seuil au-dela duquel le bruit des paires rapprochees ne passe plus.

        On prend le **maximum** du 99,9e percentile de la difference absolue
        sur l'ensemble des paires : c'est le pire cas observe en l'absence de
        toute evolution reelle. Un 99e percentile median sous-estimerait
        largement le seuil utile, la queue de distribution etant precisement
        ce qui produit les faux objets.
        """
        valeurs = self._valeurs("p999_diff")
        if not valeurs:
            return 0.0
        return float(np.max(valeurs))

    # -- plancher de la voie « detection absolue + suivi » -------------------

    def plancher_churn(self, percentile: float = 95.0) -> float:
        """Evenements de suivi produits alors que rien n'a change."""
        if not self.detections:
            return 0.0
        return float(np.percentile([d.churn for d in self.detections], percentile))

    def instabilite_nombre(self) -> float:
        """Ecart relatif median du comptage entre deux dates rapprochees."""
        if not self.detections:
            return 0.0
        return float(np.median([d.ecart_relatif_nombre for d in self.detections]))

    def instabilite_surface(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.median([d.ecart_relatif_surface for d in self.detections]))

    def metrique_recommandee(self) -> str:
        """Designe la grandeur la plus stable entre comptage et surface."""
        if not self.detections:
            return "surface"
        return (
            "nombre"
            if self.instabilite_nombre() < self.instabilite_surface()
            else "surface"
        )

    def resume(self) -> dict:
        if not self.paires:
            return {}
        objets = self._valeurs("objets_totaux")
        surfaces = self._valeurs("surface_totale")
        return {
            "paires": len(self.paires),
            "ecart_median_jours": statistics.median(
                [p.ecart_jours for p in self.paires]
            ),
            "objets_median": statistics.median(objets),
            "objets_p95": self.plancher_objets(),
            "objets_max": max(objets),
            "surface_median_px": statistics.median(surfaces),
            "surface_p95_px": self.plancher_surface(),
            "seuil_diff_suggere": self.seuil_diff_suggere(),
        }

    # -- restitution ---------------------------------------------------------

    def rendu_texte(self) -> str:
        if not self.paires:
            return (
                "  Aucune paire rapprochee trouvee "
                f"(ecart <= {self.ecart_max_jours} jours)."
            )
        r = self.resume()
        lignes = [
            f"  {r['paires']} paire(s) a {self.ecart_max_jours} jours ou moins "
            f"(ecart median {r['ecart_median_jours']:.0f} j)",
            "",
            "  Aucune evolution biologique n'est possible sur ces intervalles :",
            "  tout ce qui suit est le plancher de bruit de la chaine.",
            "",
            f"    objets de changement   median {r['objets_median']:.0f}"
            f"   p95 {r['objets_p95']:.0f}   max {r['objets_max']:.0f}",
            f"    surface changee (px)   median {r['surface_median_px']:.0f}"
            f"   p95 {r['surface_p95_px']:.0f}",
            "",
            f"  => Un ecart n'est credible qu'au-dela de "
            f"{r['objets_p95']:.0f} objets de changement.",
            f"  => Seuil de difference minimal pour ne rien laisser passer : "
            f"{r['seuil_diff_suggere']:.1f}",
        ]
        if self.detections:
            churn = [d.churn for d in self.detections]
            lignes += [
                "",
                "  Voie « detection absolue + suivi » (distincte de la precedente) :",
                f"    evenements de suivi    median {statistics.median(churn):.0f}"
                f"   p95 {self.plancher_churn():.0f}   max {max(churn):.0f}",
                f"    instabilite du comptage  {self.instabilite_nombre() * 100:.1f} %",
                f"    instabilite de la surface {self.instabilite_surface() * 100:.1f} %",
                "",
                f"  => Les apparitions / disparitions issues du suivi ne sont "
                f"credibles qu'au-dela de {self.plancher_churn():.0f} par date.",
                f"  => Grandeur la plus stable a suivre : "
                f"{self.metrique_recommandee()}.",
            ]

        pires = sorted(self.paires, key=lambda p: -p.objets_totaux)[:3]
        if pires:
            lignes += ["", "  Paires les plus bruitees (a inspecter) :"]
            for p in pires:
                lignes.append(
                    f"    {p.date_a} -> {p.date_b} ({p.ecart_jours} j) : "
                    f"{p.objets_totaux} objets, {p.surface_totale} px"
                )
        return "\n".join(lignes)

    def exporter_csv(self, chemin: Path) -> None:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "date_a",
                    "date_b",
                    "ecart_jours",
                    "objets_apparus",
                    "objets_disparus",
                    "surface_apparue_px",
                    "surface_disparue_px",
                    "gain",
                    "ecart_type_diff",
                    "p99_diff",
                    "p999_diff",
                ]
            )
            for p in self.paires:
                w.writerow(
                    [
                        p.date_a,
                        p.date_b,
                        p.ecart_jours,
                        p.objets_apparus,
                        p.objets_disparus,
                        p.surface_apparue,
                        p.surface_disparue,
                        f"{p.gain:.4f}",
                        f"{p.ecart_type_diff:.3f}",
                        f"{p.p99_diff:.3f}",
                        f"{p.p999_diff:.3f}",
                    ]
                )


def paires_rapprochees(dates: list[date], ecart_max: int) -> list[tuple[int, int, int]]:
    """Couples consecutifs separes d'au plus `ecart_max` jours.

    Renvoie (index a, index b, ecart en jours).
    """
    resultat = []
    for i in range(len(dates) - 1):
        ecart = (dates[i + 1] - dates[i]).days
        if 0 < ecart <= ecart_max:
            resultat.append((i, i + 1, ecart))
    return resultat


def sequences_rapprochees(
    dates: list[date], ecart_max: int, longueur_min: int = 3
) -> list[list[int]]:
    """Suites de dates consecutives toutes espacees d'au plus `ecart_max`.

    Une telle suite permet d'appliquer la **regle de confirmation complete**
    la ou aucune evolution n'est possible : tout evenement confirme y est un
    faux positif de la methode entiere, et non du seul appariement par paires.
    """
    sequences: list[list[int]] = []
    courante = [0]
    for i in range(len(dates) - 1):
        if 0 < (dates[i + 1] - dates[i]).days <= ecart_max:
            courante.append(i + 1)
        else:
            if len(courante) >= longueur_min:
                sequences.append(courante)
            courante = [i + 1]
    if len(courante) >= longueur_min:
        sequences.append(courante)
    return sequences


@dataclass
class ResultatSequence:
    """Faux positifs de la methode complete sur une suite rapprochee."""

    dates: list[date]
    apparitions: int
    disparitions: int
    fusions: int
    scissions: int
    nombres: list[int]
    surfaces: list[int]

    #: Nombre de confirmations exigees par la regle, necessaire pour savoir
    #: sur combien de dates elle pouvait effectivement se declencher.
    confirmations: int = 2

    @property
    def evenements_confirmes(self) -> int:
        return self.apparitions + self.disparitions

    @property
    def dates_eligibles(self) -> int:
        """Dates ou la regle de confirmation peut reellement se declencher.

        Elle exige `k` dates avant et `k` apres : les bords d'une suite en
        sont exclus. Rapporter les faux positifs au nombre *total* de dates
        sous-estimerait fortement le plancher sur les suites courtes, ou les
        bords dominent.
        """
        return max(0, len(self.dates) - 2 * self.confirmations + 1)

    @property
    def faux_positifs_par_date(self) -> float:
        if self.dates_eligibles == 0:
            return 0.0
        return self.evenements_confirmes / self.dates_eligibles

    def __str__(self) -> str:
        empan = (self.dates[-1] - self.dates[0]).days
        return (
            f"{len(self.dates)} cliches sur {empan} j "
            f"({self.dates[0]} -> {self.dates[-1]}) : "
            f"{self.apparitions} apparition(s) et {self.disparitions} "
            f"disparition(s) confirmees sur {self.dates_eligibles} date(s) "
            f"eligible(s), soit {self.faux_positifs_par_date:.0f}/date - "
            f"toutes fausses par construction"
        )


def controle_sequence(
    pipeline, base, oeil: str, ecart_max_jours: int = 10, rappel=None
) -> list[ResultatSequence]:
    """Applique la chaine complete aux suites de dates rapprochees.

    C'est le controle negatif le plus severe : detection, appariement *et*
    regle de confirmation, sur une periode ou les druses ne peuvent pas avoir
    evolue. Tout evenement confirme est un faux positif de la methode.
    """
    import cv2

    from ..pretraitement.champ import detecter_champ
    from ..pretraitement.vaisseaux import carte_vaisseaux, masque_vaisseaux
    from .druses import detecter_druses
    from .suivi import (
        compter_par_date,
        construire_pistes,
        detecter_fusions_scissions,
        evenements_confirmes,
    )

    params = pipeline.params
    seances = pipeline.seances_recalees(oeil)
    dates = [s.date for s in seances]
    sequences = sequences_rapprochees(dates, ecart_max_jours, longueur_min=3)
    if not sequences:
        return []

    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        return []
    gabarit, masque_champ = gab

    f = 0.5
    petit = cv2.resize(gabarit, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    m_petit = cv2.resize(
        masque_champ, (petit.shape[1], petit.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    vaisseaux = cv2.resize(
        masque_vaisseaux(
            carte_vaisseaux(petit, m_petit, params.vaisseaux, echelle=f),
            m_petit,
            params.vaisseaux,
        ),
        (masque_champ.shape[1], masque_champ.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    reperes = base.reperes(oeil)

    resultats = []
    for numero, indices in enumerate(sequences):
        detections = []
        sous_dates = []
        for k, idx in enumerate(indices):
            if rappel:
                rappel(
                    f"sequence {numero + 1}", k + 1, len(indices), str(dates[idx])
                )
            image = pipeline.lire_recale(oeil, dates[idx])
            if image is None:
                continue
            druses, _ = detecter_druses(
                image, masque_champ, vaisseaux, reperes, params.druses
            )
            detections.append(druses)
            sous_dates.append(dates[idx])

        if len(detections) < 3:
            continue

        pistes = construire_pistes(detections, params.suivi)
        evenements = evenements_confirmes(pistes, sous_dates, params.suivi)
        evenements += detecter_fusions_scissions(detections, sous_dates, params.suivi)
        comptes = compter_par_date(evenements, len(sous_dates))

        resultats.append(
            ResultatSequence(
                dates=sous_dates,
                apparitions=sum(comptes["apparition"]),
                disparitions=sum(comptes["disparition"]),
                fusions=sum(comptes["fusion"]),
                scissions=sum(comptes["scission"]),
                nombres=[len(d) for d in detections],
                surfaces=[sum(x.surface_px for x in d) for d in detections],
                confirmations=params.suivi.confirmations_requises,
            )
        )
    return resultats


def controle_negatif(
    pipeline,
    base,
    oeil: str,
    ecart_max_jours: int = 10,
    params: Parametres | None = None,
    rappel=None,
) -> RapportControle:
    """Execute le controle negatif sur toutes les paires rapprochees."""
    from ..pretraitement.champ import detecter_champ
    from ..pretraitement.vaisseaux import carte_vaisseaux, masque_vaisseaux
    from .differentiel import comparer, objets_de_masque

    import cv2

    params = params or pipeline.params
    rapport = RapportControle(oeil=oeil, ecart_max_jours=ecart_max_jours)

    seances = pipeline.seances_recalees(oeil)
    if len(seances) < 2:
        rapport.messages.append("pas assez de seances recalees")
        return rapport

    dates = [s.date for s in seances]
    couples = paires_rapprochees(dates, ecart_max_jours)
    if not couples:
        return rapport

    gab = pipeline.lire_gabarit(oeil)
    masque_champ = (
        gab[1] if gab else detecter_champ(pipeline.lire_recale(oeil, dates[0])).masque()
    )

    # Le masque vasculaire est calcule une fois sur le gabarit : les images
    # etant recalees, il est valable pour toute la serie.
    facteur = 0.5
    petit = cv2.resize(gab[0], None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA)
    m_petit = cv2.resize(
        masque_champ, (petit.shape[1], petit.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    reponse = carte_vaisseaux(petit, m_petit, params.vaisseaux, echelle=facteur)
    v_petit = masque_vaisseaux(reponse, m_petit, params.vaisseaux)
    vaisseaux = cv2.resize(
        v_petit, (masque_champ.shape[1], masque_champ.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    from ..pretraitement.reperes import proposer_reperes
    from .druses import detecter_druses
    from .suivi import apparier

    reperes = base.reperes(oeil)

    for n, (i, j, ecart) in enumerate(couples):
        if rappel:
            rappel("controle", n + 1, len(couples), str(dates[j]))
        img_a = pipeline.lire_recale(oeil, dates[i])
        img_b = pipeline.lire_recale(oeil, dates[j])
        if img_a is None or img_b is None:
            continue

        # --- voie « detection absolue + suivi » ---
        druses_a, _ = detecter_druses(
            img_a, masque_champ, vaisseaux, reperes, params.druses
        )
        druses_b, _ = detecter_druses(
            img_b, masque_champ, vaisseaux, reperes, params.druses
        )
        couples_ab = apparier(druses_a, druses_b, params.suivi)
        apparies_a = {a for a, _ in couples_ab}
        apparies_b = {b for _, b in couples_ab}
        rapport.detections.append(
            PaireDetection(
                date_a=dates[i],
                date_b=dates[j],
                ecart_jours=ecart,
                nombre_a=len(druses_a),
                nombre_b=len(druses_b),
                surface_a=sum(d.surface_px for d in druses_a),
                surface_b=sum(d.surface_px for d in druses_b),
                non_apparies_a=len(druses_a) - len(apparies_a),
                non_apparies_b=len(druses_b) - len(apparies_b),
            )
        )

        # --- voie differentielle ---
        diff = comparer(
            img_a, img_b, masque_champ, vaisseaux, params.differentiel
        )
        valeurs = diff.difference[diff.zone_valide > 0]

        rapport.paires.append(
            PaireControle(
                date_a=dates[i],
                date_b=dates[j],
                ecart_jours=ecart,
                surface_apparue=diff.surface_apparue,
                surface_disparue=diff.surface_disparue,
                objets_apparus=len(
                    objets_de_masque(
                        diff.masque_apparition, params.differentiel.surface_min_px
                    )
                ),
                objets_disparus=len(
                    objets_de_masque(
                        diff.masque_disparition, params.differentiel.surface_min_px
                    )
                ),
                gain=diff.gain,
                ecart_type_diff=float(valeurs.std()) if valeurs.size else 0.0,
                p99_diff=(
                    float(np.percentile(np.abs(valeurs), 99)) if valeurs.size else 0.0
                ),
                p999_diff=(
                    float(np.percentile(np.abs(valeurs), 99.9)) if valeurs.size else 0.0
                ),
            )
        )

    return rapport
