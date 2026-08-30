"""Outils de controle visuel.

La planche-contact et le comparateur clignotant sont les seuls moyens fiables
de juger a l'oeil une normalisation et un recalage : une image mal masquee ou
mal alignee y saute aux yeux instantanement.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import cv2
import numpy as np


def planche_contact(
    images: list[tuple[str, np.ndarray]],
    largeur_vignette: int = 260,
    colonnes: int | None = None,
    marge: int = 6,
) -> np.ndarray:
    """Mosaique legendee de toute une serie."""
    if not images:
        raise ValueError("aucune image")

    if colonnes is None:
        colonnes = max(1, int(math.ceil(math.sqrt(len(images) * 1.4))))
    lignes = int(math.ceil(len(images) / colonnes))

    hauteur_vignette = None
    vignettes = []
    for libelle, img in images:
        facteur = largeur_vignette / img.shape[1]
        petit = cv2.resize(
            img, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA
        )
        if hauteur_vignette is None:
            hauteur_vignette = petit.shape[0]
        if petit.shape[0] != hauteur_vignette:
            petit = cv2.resize(petit, (largeur_vignette, hauteur_vignette))
        vignettes.append((libelle, petit))

    bandeau = 22
    cell_h = hauteur_vignette + bandeau
    planche = np.full(
        (
            lignes * (cell_h + marge) + marge,
            colonnes * (largeur_vignette + marge) + marge,
            3,
        ),
        28,
        dtype=np.uint8,
    )

    for i, (libelle, vignette) in enumerate(vignettes):
        r, c = divmod(i, colonnes)
        y = marge + r * (cell_h + marge)
        x = marge + c * (largeur_vignette + marge)
        planche[y : y + hauteur_vignette, x : x + largeur_vignette] = vignette
        cv2.putText(
            planche,
            libelle,
            (x + 3, y + hauteur_vignette + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return planche


def damier(
    a: np.ndarray, b: np.ndarray, cases: int = 8, epaisseur_trait: int = 1
) -> np.ndarray:
    """Superposition en damier : verifie la continuite des vaisseaux.

    Un vaisseau qui se brise a la frontiere d'une case signale un defaut
    d'alignement.
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    hauteur, largeur = a.shape[:2]
    pas_y, pas_x = hauteur // cases, largeur // cases
    sortie = a.copy()
    for r in range(cases):
        for c in range(cases):
            if (r + c) % 2 == 0:
                continue
            y0, y1 = r * pas_y, min((r + 1) * pas_y, hauteur)
            x0, x1 = c * pas_x, min((c + 1) * pas_x, largeur)
            sortie[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    if epaisseur_trait > 0:
        for r in range(1, cases):
            cv2.line(sortie, (0, r * pas_y), (largeur, r * pas_y), (0, 200, 255), epaisseur_trait)
        for c in range(1, cases):
            cv2.line(sortie, (c * pas_x, 0), (c * pas_x, hauteur), (0, 200, 255), epaisseur_trait)
    return sortie


def superposition_fausses_couleurs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a` en vert, `b` en magenta : le gris signale un alignement parfait."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sortie = np.zeros((*ga.shape, 3), dtype=np.float32)
    sortie[:, :, 1] = ga  # vert
    sortie[:, :, 0] = gb  # bleu
    sortie[:, :, 2] = gb  # rouge -> magenta
    return np.clip(sortie, 0, 255).astype(np.uint8)


def erreur_alignement(
    a: np.ndarray, b: np.ndarray, masque: np.ndarray | None = None
) -> float:
    """Erreur quadratique moyenne d'intensite, sur la zone commune.

    Indicateur brut mais utile pour classer des recalages entre eux.
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if masque is None:
        masque = ((ga > 0) & (gb > 0)).astype(np.uint8) * 255
    zone = masque > 0
    if not zone.any():
        return float("inf")
    return float(np.sqrt(((ga[zone] - gb[zone]) ** 2).mean()))


def legender(image: np.ndarray, texte: str, position: tuple[int, int] = (20, 40)) -> np.ndarray:
    sortie = image.copy()
    cv2.putText(
        sortie, texte, position, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(
        sortie, texte, position, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA
    )
    return sortie
