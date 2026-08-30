"""Detection du champ retinien (FOV).

Seuillage d'Otsu sur le canal rouge - le plus lumineux, celui qui separe le
mieux la retine du pourtour noir -, puis plus grand contour et cercle
englobant minimal.

L'erosion finale n'est pas cosmetique : le degrade du bord fausse toute
statistique d'intensite et genere de fausses detections en peripherie.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ParamsChamp


@dataclass(frozen=True)
class Champ:
    """Disque retinien : centre, rayon, et masque binaire associe."""

    cx: float
    cy: float
    rayon: float
    forme: tuple[int, int]  # (hauteur, largeur)

    def masque(self, erosion: float = 1.0) -> np.ndarray:
        m = np.zeros(self.forme, dtype=np.uint8)
        cv2.circle(
            m,
            (int(round(self.cx)), int(round(self.cy))),
            max(1, int(round(self.rayon * erosion))),
            255,
            -1,
        )
        return m

    def rectangle(self, marge: int = 0) -> tuple[int, int, int, int]:
        """Rectangle englobant (x0, y0, x1, y1), borne a l'image."""
        h, l = self.forme
        x0 = max(0, int(self.cx - self.rayon) - marge)
        y0 = max(0, int(self.cy - self.rayon) - marge)
        x1 = min(l, int(self.cx + self.rayon) + marge + 1)
        y1 = min(h, int(self.cy + self.rayon) + marge + 1)
        return x0, y0, x1, y1


def masque_disque(forme: tuple[int, int], cx: float, cy: float, rayon: float) -> np.ndarray:
    m = np.zeros(forme, dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(rayon))), 255, -1)
    return m


def detecter_champ(image_bgr: np.ndarray, params: ParamsChamp | None = None) -> Champ:
    """Localise le disque retinien.

    Un arc lumineux de flash en bordure peut faire deborder Otsu au-dela du
    champ reel ; l'erosion de `params.facteur_erosion` absorbe ce cas.
    """
    params = params or ParamsChamp()
    hauteur, largeur = image_bgr.shape[:2]

    canal = {"rouge": 2, "vert": 1, "bleu": 0}.get(params.canal, 2)
    plan = image_bgr[:, :, canal]

    if params.flou_px > 1:
        k = params.flou_px | 1
        plan = cv2.GaussianBlur(plan, (k, k), 0)

    _, binaire = cv2.threshold(plan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fermeture puis remplissage des trous : le disque optique et les reflets
    # tres clairs ne doivent pas creuser le masque.
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, noyau)

    contours, _ = cv2.findContours(binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Repli : tout le cadre.
        return Champ(largeur / 2, hauteur / 2, min(hauteur, largeur) / 2, (hauteur, largeur))

    plus_grand = max(contours, key=cv2.contourArea)
    (cx, cy), rayon = cv2.minEnclosingCircle(plus_grand)
    rayon *= params.facteur_erosion

    return Champ(float(cx), float(cy), float(rayon), (hauteur, largeur))


def appliquer_masque(image: np.ndarray, masque: np.ndarray) -> np.ndarray:
    """Met a zero hors du champ, en preservant le nombre de canaux."""
    if image.ndim == 2:
        return cv2.bitwise_and(image, image, mask=masque)
    return cv2.bitwise_and(image, image, mask=masque)


def recadrer(image: np.ndarray, champ: Champ, marge: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = champ.rectangle(marge)
    return image[y0:y1, x0:x1]
