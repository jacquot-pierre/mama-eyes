"""Correction d'illumination et harmonisation colorimetrique.

Deux problemes distincts, traites separement et dans cet ordre :

1. Le **vignettage intra-image** (centre clair, peripherie sombre, variable
   selon le cadrage et la dilatation pupillaire) est corrige en estimant le
   champ d'illumination lent par un flou gaussien de forte etendue, puis en
   divisant l'image par cette estimation. C'est l'etape la plus determinante
   de toute la chaine pour la validite des differences temporelles : sans
   elle, la derive de vignettage domine le signal et noie l'evolution reelle.

2. Les **ecarts globaux entre seances** (exposition, balance des blancs) sont
   corriges par transfert de Reinhard en espace LAB vers les statistiques
   *medianes* de la serie - la mediane, et non la moyenne, pour resister aux
   cliches aberrants.

Aucun CLAHE ici : le rehaussement adaptatif est reserve a l'affichage et a la
detection de points caracteristiques, jamais aux images qui alimentent une
mesure quantitative.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ParamsIllumination


@dataclass(frozen=True)
class StatsLab:
    """Moyenne et ecart-type des trois canaux LAB, sur le champ utile."""

    l_moy: float
    l_ect: float
    a_moy: float
    a_ect: float
    b_moy: float
    b_ect: float

    def en_tuple(self) -> tuple[float, ...]:
        return (self.l_moy, self.l_ect, self.a_moy, self.a_ect, self.b_moy, self.b_ect)


def stats_lab(image_bgr: np.ndarray, masque: np.ndarray) -> StatsLab:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    valeurs = []
    for i in range(3):
        moy, ect = cv2.meanStdDev(lab[:, :, i], mask=masque)
        valeurs.append(float(moy[0][0]))
        valeurs.append(max(float(ect[0][0]), 1e-6))
    return StatsLab(*valeurs)


def stats_medianes(liste: list[StatsLab]) -> StatsLab:
    """Cible de la serie. Mediane plutot que moyenne : robuste aux aberrants."""
    if not liste:
        raise ValueError("aucune statistique fournie")
    tableau = np.array([s.en_tuple() for s in liste], dtype=np.float64)
    return StatsLab(*np.median(tableau, axis=0).tolist())


def corriger_vignettage(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    rayon_champ: float,
    params: ParamsIllumination | None = None,
) -> np.ndarray:
    """Divise l'image par son champ d'illumination lent.

    Le sigma est proportionnel au rayon du champ, de sorte que la correction
    reste identique quel que soit le cadrage.
    """
    params = params or ParamsIllumination()
    sigma = max(4.0, rayon_champ * params.sigma_fond_fraction)

    travail = image_bgr.astype(np.float32)
    interieur = masque > 0
    if not interieur.any():
        return image_bgr.copy()

    sortie = np.empty_like(travail)
    for c in range(3):
        plan = travail[:, :, c]
        moyenne = float(plan[interieur].mean())

        # Le flou ne doit pas aspirer le noir de l'exterieur du champ : on
        # remplit d'abord l'exterieur avec la moyenne interieure.
        rempli = np.where(interieur, plan, moyenne).astype(np.float32)
        fond = cv2.GaussianBlur(rempli, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

        if params.mode == "soustraction":
            corrige = plan - fond + moyenne
        else:
            # Division par le fond, remise au niveau moyen d'origine. Aucun
            # etirement de contraste : un etirement global multiplierait la
            # non-uniformite residuelle au lieu de la reduire.
            fond = np.maximum(fond, max(1.0, moyenne * 0.05))
            corrige = plan / fond * moyenne

        sortie[:, :, c] = corrige

    sortie = np.clip(sortie, 0, 255).astype(np.uint8)
    sortie[~interieur] = 0
    return sortie


def planeite(image_bgr: np.ndarray, masque: np.ndarray, sigma: float) -> float:
    """Coefficient de variation du fond lisse, en %.

    Mesure de controle du vignettage : plus la valeur est basse, plus
    l'illumination est uniforme sur le champ. A comparer avant / apres
    `corriger_vignettage`, au meme sigma que la correction.
    """
    interieur = masque > 0
    if not interieur.any():
        return 0.0
    plan = image_bgr[:, :, 1].astype(np.float32)
    moyenne = float(plan[interieur].mean())
    rempli = np.where(interieur, plan, moyenne).astype(np.float32)
    fond = cv2.GaussianBlur(rempli, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    valeurs = fond[interieur]
    if valeurs.mean() < 1e-6:
        return 0.0
    return float(valeurs.std() / valeurs.mean() * 100.0)


def harmoniser_lab(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    cible: StatsLab,
    params: ParamsIllumination | None = None,
) -> np.ndarray:
    """Transfert de Reinhard vers les statistiques cibles de la serie."""
    params = params or ParamsIllumination()
    source = stats_lab(image_bgr, masque)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    couples = (
        (source.l_moy, source.l_ect, cible.l_moy, cible.l_ect),
        (source.a_moy, source.a_ect, cible.a_moy, cible.a_ect),
        (source.b_moy, source.b_ect, cible.b_moy, cible.b_ect),
    )
    for i, (s_moy, s_ect, c_moy, c_ect) in enumerate(couples):
        lab[:, :, i] = (lab[:, :, i] - s_moy) * (c_ect / s_ect) + c_moy

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    sortie = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    sortie[masque == 0] = 0
    return sortie


def masque_hors_disque(
    masque_champ: np.ndarray, disque_xy: tuple[float, float] | None, rayon: float
) -> np.ndarray:
    """Champ prive de la region du disque optique.

    Le disque est tres clair et sa taille apparente varie avec le cadrage :
    l'inclure dans le calcul des statistiques cibles biaise l'harmonisation.
    """
    if disque_xy is None:
        return masque_champ
    m = masque_champ.copy()
    cv2.circle(
        m,
        (int(round(disque_xy[0])), int(round(disque_xy[1]))),
        max(1, int(round(rayon))),
        0,
        -1,
    )
    return m


def clahe_affichage(image_bgr: np.ndarray, clip: float = 2.0, tuiles: int = 8) -> np.ndarray:
    """Rehaussement local, **pour l'affichage et la detection de points seulement**.

    Ne jamais appliquer en amont d'une mesure quantitative : CLAHE amplifie
    differemment chaque image selon son contenu, et l'apparition d'une lesion
    modifie le rehaussement local, donc la difference mesuree.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    outil = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tuiles, tuiles))
    lab[:, :, 0] = outil.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
