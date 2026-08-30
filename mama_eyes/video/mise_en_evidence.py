"""Video mettant en evidence les changements confirmes.

Superposition des zones de disparition et d'apparition, en contours ou en
carte de chaleur avec transparence, **avec apparition progressive sur
plusieurs trames** pour que l'oeil ait le temps de les accrocher.

Contrairement au masque cumulatif de la version initiale - qui ne pouvait que
croitre et rendait le decompte ininterpretable -, l'incrustation ici reflete
l'etat courant du suivi : une zone peut cesser d'etre signalee si la
detection est retractee.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from ..config import ParamsVideo
from .incrustation import incruster_date, incruster_frise
from .timelapse import EcrivainVideo

VERT = (120, 255, 120)
ROUGE = (110, 110, 255)


def superposer_masque(
    image: np.ndarray,
    masque: np.ndarray,
    couleur: tuple[int, int, int],
    alpha: float = 0.45,
    contour: bool = True,
) -> np.ndarray:
    """Teinte les zones du masque et en trace le contour."""
    if masque is None or not masque.any():
        return image
    sortie = image.copy()
    voile = np.zeros_like(image)
    voile[masque > 0] = couleur
    sortie = cv2.addWeighted(sortie, 1.0, voile, alpha, 0)
    if contour:
        contours, _ = cv2.findContours(
            masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(sortie, contours, -1, couleur, 2)
    return sortie


def legende(trame: np.ndarray, sous_titre: str = "") -> np.ndarray:
    """Legende des couleurs, en haut a droite."""
    largeur = trame.shape[1]
    x = largeur - 330
    for i, (couleur, texte) in enumerate(
        ((VERT, "druse disparue"), (ROUGE, "druse apparue"))
    ):
        y = 40 + i * 42
        cv2.rectangle(trame, (x, y - 18), (x + 26, y + 4), couleur, -1)
        cv2.putText(
            trame, texte, (x + 38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA
        )
        cv2.putText(
            trame, texte, (x + 38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, couleur, 2, cv2.LINE_AA
        )
    if sous_titre:
        cv2.putText(
            trame, sous_titre, (x, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            trame, sous_titre, (x, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (200, 200, 200), 1, cv2.LINE_AA
        )
    return trame


def generer_mise_en_evidence(
    pipeline,
    base,
    oeil: str,
    cible: Path,
    params: ParamsVideo | None = None,
    largeur_max: int = 1600,
    trames_fondu: int = 3,
    reference: str = "glissante",
    recul_jours: int = 365,
    rappel=None,
) -> Path | None:
    """Video des changements par rapport a une reference.

    `reference` vaut :

    - ``"glissante"`` (defaut) : chaque date est comparee au cliche le plus
      proche d'il y a `recul_jours`. Entre deux dates consecutives, le
      changement reel est sous le plancher de bruit une fois les seuils
      calibres : la video serait vide. Un recul d'environ un an fait
      apparaitre l'evolution effective.
    - ``"precedente"`` : comparaison a la date precedente.
    - ``"initiale"`` : comparaison au premier cliche de la serie.
    """
    from ..analyse.differentiel import comparer
    from ..analyse.execution import masque_vasculaire_gabarit

    params = params or pipeline.params.video
    seances = pipeline.seances_recalees(oeil)
    if len(seances) < 3:
        return None

    gab = pipeline.lire_gabarit(oeil)
    if gab is None:
        return None
    _, masque_champ = gab
    vaisseaux = masque_vasculaire_gabarit(pipeline, oeil, pipeline.params)

    dates = [s.date for s in seances]
    evenements = base.evenements()

    premiere = pipeline.lire_recale(oeil, dates[0])
    if premiere is None:
        return None
    facteur = min(1.0, largeur_max / premiere.shape[1])
    largeur = int(premiere.shape[1] * facteur) // 2 * 2
    hauteur = int(premiere.shape[0] * facteur) // 2 * 2

    def index_reference(i: int) -> int | None:
        """Index du cliche servant de base de comparaison pour la date `i`."""
        if reference == "initiale":
            return 0 if i > 0 else None
        if reference == "precedente":
            return i - 1 if i > 0 else None
        # Glissante : le cliche le plus proche d'il y a `recul_jours`.
        vise = dates[i] - timedelta(days=recul_jours)
        candidats = [k for k in range(i) if dates[k] <= vise]
        if not candidats:
            return None
        return max(candidats, key=lambda k: dates[k])

    with EcrivainVideo(cible, params.fps, (largeur, hauteur), params.qualite) as sortie:
        for i in range(1, len(seances)):
            if rappel:
                rappel("evolution", i, len(seances) - 1, str(dates[i]))
            courante = pipeline.lire_recale(oeil, dates[i])
            if courante is None:
                continue

            j = index_reference(i)
            if j is None:
                continue
            base_image = pipeline.lire_recale(oeil, dates[j])
            if base_image is None:
                continue

            diff = comparer(
                base_image,
                courante,
                masque_champ,
                vaisseaux,
                pipeline.params.differentiel,
            )
            ecart_reference = (dates[i] - dates[j]).days

            base_trame = cv2.resize(courante, (largeur, hauteur), interpolation=cv2.INTER_AREA)
            m_disp = cv2.resize(
                diff.masque_disparition, (largeur, hauteur), interpolation=cv2.INTER_NEAREST
            )
            m_app = cv2.resize(
                diff.masque_apparition, (largeur, hauteur), interpolation=cv2.INTER_NEAREST
            )

            # Fondu progressif : l'oeil a besoin de quelques trames pour
            # accrocher une zone qui s'allume.
            for k in range(trames_fondu):
                alpha = 0.45 * (k + 1) / trames_fondu
                trame = superposer_masque(base_trame, m_disp, VERT, alpha, contour=False)
                trame = superposer_masque(trame, m_app, ROUGE, alpha, contour=False)
                incruster_date(trame, dates[i])
                incruster_frise(trame, dates[i], dates[0], dates[-1], evenements)
                legende(trame, f"par rapport a -{ecart_reference} j")
                sortie.ecrire(trame)

            trame = superposer_masque(base_trame, m_disp, VERT, 0.45)
            trame = superposer_masque(trame, m_app, ROUGE, 0.45)
            incruster_date(trame, dates[i])
            incruster_frise(trame, dates[i], dates[0], dates[-1], evenements)
            legende(trame, f"par rapport a -{ecart_reference} j")
            for _ in range(max(1, params.trames_par_date)):
                sortie.ecrire(trame)


    return cible
