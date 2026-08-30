"""Suivi temporel des druses par appariement d'objets.

La regle de confirmation - un changement n'est valide que s'il est observe sur
au moins deux comparaisons consecutives - gagne beaucoup a etre appliquee au
niveau des **objets** plutot que des pixels :

1. les druses sont detectees independamment a chaque date ;
2. elles sont appariees entre dates consecutives par affectation optimale
   (`scipy.optimize.linear_sum_assignment`), le cout combinant distance des
   centroides et ecart de surface. On obtient des *pistes* : chaque druse
   recoit une identite stable dans le temps ;
3. la regle s'applique aux pistes.

Cette formulation donne un decompte veritable, resiste a un cliche isole de
mauvaise qualite et autorise la retractation d'une detection erronee - ce
qu'un masque cumulatif, qui ne peut que croitre, ne permet pas.

Limite : l'appariement suppose des deplacements faibles entre dates. Sur les
grands intervalles et en cas de druses confluentes qui fusionnent ou se
scindent, la correspondance devient ambigue ; la fusion de deux druses en une
se lirait comme « une disparition ». Ces evenements sont donc journalises
separement des apparitions et disparitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import ParamsSuivi
from ..domain import Druse


@dataclass
class Piste:
    """Identite d'une druse suivie dans le temps."""

    id: int
    presences: dict[int, Druse] = field(default_factory=dict)  # index de date -> druse

    def indices(self) -> list[int]:
        return sorted(self.presences)

    def present(self, index: int) -> bool:
        return index in self.presences

    def duree(self) -> int:
        idx = self.indices()
        return (idx[-1] - idx[0] + 1) if idx else 0


@dataclass
class EvenementDruse:
    """Apparition ou disparition confirmee."""

    type: str  # 'apparition' | 'disparition' | 'fusion' | 'scission'
    index_date: int
    date: date
    piste_id: int
    cx: float
    cy: float
    surface_px: int


def _matrice_cout(
    a: list[Druse], b: list[Druse], params: ParamsSuivi
) -> np.ndarray:
    """Cout d'appariement : distance des centroides + ecart relatif de surface."""
    pos_a = np.array([[d.cx, d.cy] for d in a], dtype=np.float64)
    pos_b = np.array([[d.cx, d.cy] for d in b], dtype=np.float64)
    surf_a = np.array([d.surface_px for d in a], dtype=np.float64)
    surf_b = np.array([d.surface_px for d in b], dtype=np.float64)

    distance = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)

    denom = np.maximum(surf_a[:, None], surf_b[None, :])
    ecart_surface = np.abs(surf_a[:, None] - surf_b[None, :]) / np.maximum(denom, 1.0)

    cout = distance + params.poids_surface * params.distance_max_px * ecart_surface

    # Au-dela de la distance maximale, l'appariement est interdit.
    cout[distance > params.distance_max_px] = 1e9
    return cout


def apparier(
    a: list[Druse], b: list[Druse], params: ParamsSuivi | None = None
) -> list[tuple[int, int]]:
    """Affectation optimale entre deux ensembles de druses.

    Renvoie les couples (index dans a, index dans b).
    """
    params = params or ParamsSuivi()
    if not a or not b:
        return []
    cout = _matrice_cout(a, b, params)
    lignes, colonnes = linear_sum_assignment(cout)
    return [
        (int(i), int(j)) for i, j in zip(lignes, colonnes) if cout[i, j] < 1e9
    ]


def construire_pistes(
    detections: list[list[Druse]], params: ParamsSuivi | None = None
) -> list[Piste]:
    """Chaine les detections date par date en pistes, avec memoire des absences.

    Une piste n'est pas rompue par une absence isolee : elle reste candidate a
    l'appariement pendant `params.memoire_dates` dates. Sans cette memoire, une
    druse manquee sur un seul cliche - un flou, une variation d'exposition -
    produirait une nouvelle piste, donc *a la fois* une fausse disparition et
    une fausse apparition. C'est precisement le bruit que la regle de
    confirmation doit supprimer.
    """
    params = params or ParamsSuivi()
    memoire = max(1, params.memoire_dates)

    pistes: list[Piste] = []
    # Pistes encore candidates : (piste, derniere date vue, derniere druse vue).
    vivantes: list[tuple[Piste, int, Druse]] = []

    for index, druses in enumerate(detections):
        # On n'apparie qu'avec les pistes vues assez recemment.
        candidates = [
            (p, dernier, druse)
            for (p, dernier, druse) in vivantes
            if index - dernier <= memoire
        ]

        couples: list[tuple[int, int]] = []
        if candidates and druses:
            couples = apparier([c[2] for c in candidates], druses, params)

        apparies_b = {j for _, j in couples}
        prolongees: set[int] = set()

        for i, j in couples:
            piste = candidates[i][0]
            druses[j].piste_id = piste.id
            piste.presences[index] = druses[j]
            prolongees.add(piste.id)

        for j, druse in enumerate(druses):
            if j in apparies_b:
                continue
            piste = Piste(id=len(pistes))
            druse.piste_id = piste.id
            piste.presences[index] = druse
            pistes.append(piste)
            prolongees.add(piste.id)

        # Mise a jour de l'ensemble des pistes candidates.
        nouvelles_vivantes: list[tuple[Piste, int, Druse]] = []
        for piste in pistes:
            indices = piste.indices()
            if not indices:
                continue
            dernier = indices[-1]
            if index - dernier <= memoire:
                nouvelles_vivantes.append((piste, dernier, piste.presences[dernier]))
        vivantes = nouvelles_vivantes
        del prolongees

    return pistes


def evenements_confirmes(
    pistes: list[Piste],
    dates: list[date],
    params: ParamsSuivi | None = None,
) -> list[EvenementDruse]:
    """Applique la regle de confirmation aux pistes.

    Une druse est « disparue en t » si sa piste existe en t-1 et est absente
    en t **et** t+1 ; « apparue en t » si elle est absente en t-2 et t-1,
    presente en t **et** t+1.

    Le nombre de confirmations est reglable (`confirmations_requises`).
    """
    params = params or ParamsSuivi()
    k = max(1, params.confirmations_requises)
    n = len(dates)
    evenements: list[EvenementDruse] = []

    for piste in pistes:
        indices = set(piste.indices())
        if not indices:
            continue

        for t in range(n):
            # --- apparition ---
            avant_absent = all((t - d) not in indices for d in range(1, k + 1) if t - d >= 0)
            assez_avant = t >= k
            apres_present = all(
                (t + d) in indices for d in range(0, k) if t + d < n
            )
            assez_apres = t + k - 1 < n
            if assez_avant and assez_apres and avant_absent and apres_present:
                d = piste.presences[t]
                evenements.append(
                    EvenementDruse(
                        "apparition", t, dates[t], piste.id, d.cx, d.cy, d.surface_px
                    )
                )

            # --- disparition ---
            avant_present = (t - 1) in indices
            apres_absent = all(
                (t + d) not in indices for d in range(0, k) if t + d < n
            )
            if t >= 1 and t + k - 1 < n and avant_present and apres_absent:
                d = piste.presences[t - 1]
                evenements.append(
                    EvenementDruse(
                        "disparition", t, dates[t], piste.id, d.cx, d.cy, d.surface_px
                    )
                )

    return evenements


def detecter_fusions_scissions(
    detections: list[list[Druse]],
    dates: list[date],
    params: ParamsSuivi | None = None,
) -> list[EvenementDruse]:
    """Repere les fusions et scissions, comptees separement.

    Sans cela, la fusion de deux druses en une serait comptabilisee comme une
    disparition, et une scission comme une apparition.
    """
    params = params or ParamsSuivi()
    if not params.signaler_fusion_scission:
        return []

    evenements: list[EvenementDruse] = []
    for index in range(1, len(detections)):
        a, b = detections[index - 1], detections[index]
        if not a or not b:
            continue

        cout = _matrice_cout(a, b, params)
        proche = cout < 1e9

        # Fusion : plusieurs druses de `a` n'ont qu'une seule candidate dans `b`,
        # et c'est la meme.
        for j in range(len(b)):
            sources = np.nonzero(proche[:, j])[0]
            if len(sources) >= 2:
                surface_source = sum(a[i].surface_px for i in sources)
                if b[j].surface_px > 0.6 * surface_source:
                    evenements.append(
                        EvenementDruse(
                            "fusion",
                            index,
                            dates[index],
                            -1,
                            b[j].cx,
                            b[j].cy,
                            b[j].surface_px,
                        )
                    )

        # Scission : une druse de `a` correspond a plusieurs de `b`.
        for i in range(len(a)):
            cibles = np.nonzero(proche[i, :])[0]
            if len(cibles) >= 2:
                surface_cible = sum(b[j].surface_px for j in cibles)
                if a[i].surface_px > 0.6 * surface_cible:
                    evenements.append(
                        EvenementDruse(
                            "scission",
                            index,
                            dates[index],
                            -1,
                            a[i].cx,
                            a[i].cy,
                            a[i].surface_px,
                        )
                    )

    return evenements


def compter_par_date(
    evenements: list[EvenementDruse], nb_dates: int
) -> dict[str, list[int]]:
    """Agrege les evenements par type et par index de date."""
    resultat = {
        t: [0] * nb_dates for t in ("apparition", "disparition", "fusion", "scission")
    }
    for e in evenements:
        if 0 <= e.index_date < nb_dates and e.type in resultat:
            resultat[e.type][e.index_date] += 1
    return resultat
