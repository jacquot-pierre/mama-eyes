"""Detection du reseau vasculaire par filtre de Frangi multi-echelle.

Double usage, et c'est ce qui rend l'etape rentable :

- les **bifurcations vasculaires** sont les reperes anatomiques les plus
  stables sur plusieurs annees, donc les meilleurs points d'ancrage pour le
  recalage ;
- le **masque vasculaire dilate** sert d'exclusion a la detection de druses,
  sans quoi les vaisseaux sont comptes comme lesions.

Limites : Frangi repond a toute structure tubulaire. Il s'allume sur le bord
du champ (d'ou l'erosion du masque *avant* filtrage), sur le pourtour du
disque optique et sur les contours de grosses druses confluentes. Sa reponse
est en outre intrinsequement faible aux croisements et bifurcations -
precisement les points que l'on veut utiliser : on les detecte donc sur le
squelette binarise, pas sur la reponse continue.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

from ..config import ParamsVaisseaux


def carte_vaisseaux(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    params: ParamsVaisseaux | None = None,
    echelle: float = 1.0,
) -> np.ndarray:
    """Reponse de Frangi normalisee en 0..1, sur le canal vert inverse.

    `echelle` permet de travailler sur une image sous-echantillonnee ; les
    sigmas sont mis a l'echelle en consequence.
    """
    params = params or ParamsVaisseaux()

    vert = image_bgr[:, :, 1].astype(np.float32) / 255.0
    # Les vaisseaux sont les structures tubulaires *sombres* du canal vert.
    inverse = 1.0 - vert
    inverse[masque == 0] = 0.0

    sigmas = np.linspace(
        params.sigma_min * echelle,
        params.sigma_max * echelle,
        params.nb_sigmas,
    )
    sigmas = np.maximum(sigmas, 0.6)

    reponse = frangi(
        inverse,
        sigmas=sigmas,
        black_ridges=False,
        alpha=0.5,
        beta=0.5,
    )
    reponse = np.nan_to_num(reponse, nan=0.0, posinf=0.0, neginf=0.0)
    reponse[masque == 0] = 0.0

    interieur = masque > 0
    if interieur.any():
        haut = float(np.percentile(reponse[interieur], 99.5))
        if haut > 1e-12:
            reponse = np.clip(reponse / haut, 0.0, 1.0)
    return reponse.astype(np.float32)


def masque_vaisseaux(
    reponse: np.ndarray,
    masque_champ: np.ndarray,
    params: ParamsVaisseaux | None = None,
) -> np.ndarray:
    """Binarise la reponse de Frangi et nettoie les petits objets."""
    params = params or ParamsVaisseaux()
    interieur = masque_champ > 0
    if not interieur.any():
        return np.zeros_like(masque_champ)

    seuil = float(np.percentile(reponse[interieur], params.percentile_seuil))
    binaire = (reponse > seuil) & interieur

    if params.taille_min_objet_px > 0:
        binaire = remove_small_objects(binaire, max_size=params.taille_min_objet_px)

    return (binaire.astype(np.uint8)) * 255


def masque_exclusion(
    binaire: np.ndarray, params: ParamsVaisseaux | None = None
) -> np.ndarray:
    """Masque vasculaire dilate, a soustraire de la detection de druses.

    Une erreur de recalage de 2 px sur un bord de vaisseau tres contraste
    produit une difference plus forte qu'une vraie druse : la dilatation
    absorbe ce residu.
    """
    params = params or ParamsVaisseaux()
    if params.dilatation_masque_px <= 1:
        return binaire
    k = params.dilatation_masque_px | 1
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binaire, noyau)


def bifurcations(binaire: np.ndarray, marge_bord: int = 30) -> np.ndarray:
    """Points de branchement du squelette vasculaire.

    Renvoie un tableau (n, 2) de coordonnees (x, y). Ces points sont les
    reperes anatomiques les plus stables dans le temps.
    """
    squelette = skeletonize(binaire > 0)
    sq = squelette.astype(np.uint8)

    # Nombre de voisins de chaque pixel du squelette.
    noyau = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    voisins = cv2.filter2D(sq, cv2.CV_8U, noyau)
    points = (sq > 0) & (voisins >= 3)

    ys, xs = np.nonzero(points)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    h, l = binaire.shape[:2]
    garde = (
        (xs >= marge_bord) & (xs < l - marge_bord)
        & (ys >= marge_bord) & (ys < h - marge_bord)
    )
    return np.column_stack([xs[garde], ys[garde]]).astype(np.float32)


def densite_vasculaire(binaire: np.ndarray, masque_champ: np.ndarray) -> float:
    """Fraction du champ occupee par des vaisseaux. Indicateur de qualite."""
    interieur = masque_champ > 0
    if not interieur.any():
        return 0.0
    return float((binaire[interieur] > 0).mean())
