"""Incrustations : date, frise chronologique, reperes d'evenements.

La frise reprend le rendu des scripts d'origine, en y ajoutant les reperes
therapeutiques enregistres en base.
"""

from __future__ import annotations

from datetime import date

import cv2
import numpy as np

from ..domain import Evenement

BLANC = (255, 255, 255)
GRIS = (90, 90, 90)
ACCENT = (60, 180, 255)  # BGR : ambre


def incruster_date(
    trame: np.ndarray,
    jour: date,
    position: tuple[int, int] | None = None,
    echelle: float = 2.0,
) -> np.ndarray:
    hauteur, largeur = trame.shape[:2]
    if position is None:
        position = (50, hauteur - 150)
    texte = jour.strftime("%Y-%m-%d")
    cv2.putText(
        trame, texte, position, cv2.FONT_HERSHEY_SIMPLEX, echelle, (0, 0, 0), 7, cv2.LINE_AA
    )
    cv2.putText(
        trame, texte, position, cv2.FONT_HERSHEY_SIMPLEX, echelle, BLANC, 3, cv2.LINE_AA
    )
    return trame


def incruster_frise(
    trame: np.ndarray,
    jour: date,
    debut: date,
    fin: date,
    evenements: list[Evenement] | None = None,
) -> np.ndarray:
    """Frise chronologique avec graduations annuelles et reperes d'evenements."""
    hauteur, largeur = trame.shape[:2]
    y = hauteur - 80
    x0, x1 = 50, largeur - 50
    empan = max((fin - debut).days, 1)

    def position(d: date) -> int:
        return x0 + int((d - debut).days / empan * (x1 - x0))

    cv2.rectangle(trame, (x0, y - 10), (x1, y + 10), GRIS, -1)
    cv2.rectangle(trame, (x0, y - 10), (position(jour), y + 10), BLANC, -1)

    for annee in range(debut.year, fin.year + 1):
        borne = date(annee, 1, 1)
        if not (debut <= borne <= fin):
            continue
        x = position(borne)
        cv2.line(trame, (x, y - 20), (x, y + 20), (200, 200, 200), 2)
        cv2.putText(
            trame,
            str(annee),
            (x - 38, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (215, 215, 215),
            2,
            cv2.LINE_AA,
        )

    for e in evenements or []:
        if not (debut <= e.date <= fin):
            continue
        x = position(e.date)
        cv2.line(trame, (x, y - 34), (x, y + 34), ACCENT, 3)
        cv2.putText(
            trame,
            e.libelle[:26],
            (x + 8, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            trame,
            e.libelle[:26],
            (x + 8, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            ACCENT,
            2,
            cv2.LINE_AA,
        )
    return trame


def incruster_ecart(trame: np.ndarray, jours: int | None) -> np.ndarray:
    """Affiche l'ecart depuis le cliche precedent.

    Les intervalles allant de 3 a 130 jours, une cadence constante laisserait
    croire a un rythme regulier : l'ecart doit rester lisible.
    """
    if jours is None:
        return trame
    hauteur = trame.shape[0]
    texte = f"+{jours} j"
    cv2.putText(
        trame, texte, (50, hauteur - 210), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(
        trame, texte, (50, hauteur - 210), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (190, 190, 190), 2, cv2.LINE_AA
    )
    return trame


def incruster_avertissement(trame: np.ndarray, texte: str) -> np.ndarray:
    """Bandeau d'alerte, par exemple pour un recalage invalide."""
    largeur = trame.shape[1]
    cv2.rectangle(trame, (0, 0), (largeur, 54), (0, 0, 140), -1)
    cv2.putText(
        trame, texte, (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLANC, 2, cv2.LINE_AA
    )
    return trame
