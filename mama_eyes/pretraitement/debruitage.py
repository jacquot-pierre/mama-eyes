"""Debruitage spatial, image par image.

Filtre median : il supprime le bruit impulsionnel et les poussieres de
capteur en preservant les contours, contrairement a un flou gaussien.

Aucun moyennage temporel entre dates differentes - seule la fusion
intra-seance (meme jour) combine plusieurs cliches.

Un median 5x5 a 3008x2000 efface les structures de moins de ~5 px, ce qui
peut correspondre a des druses de 25-40 um : le noyau 3x3 est donc le defaut
pour l'image d'analyse, le 5x5 restant reserve a l'affichage.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ParamsDebruitage


def debruiter(
    image: np.ndarray,
    params: ParamsDebruitage | None = None,
    pour_affichage: bool = False,
) -> np.ndarray:
    params = params or ParamsDebruitage()
    taille = params.median_affichage_px if pour_affichage else params.median_analyse_px
    if taille < 3:
        return image
    return cv2.medianBlur(image, taille | 1)


def debruiter_non_local(image: np.ndarray, force: float = 3.0) -> np.ndarray:
    """Non-local means : preserve mieux les petites structures que le median.

    Beaucoup plus lent - reserve aux cas ou la detection des petites druses
    compte plus que le temps de calcul.
    """
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, force, 7, 21)
    return cv2.fastNlMeansDenoisingColored(image, None, force, force, 7, 21)
