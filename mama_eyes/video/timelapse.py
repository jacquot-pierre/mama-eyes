"""Timelapse a partir des seances recalees.

Deux precautions que la version initiale n'avait pas :

- **lissage temporel de l'exposition** : meme apres harmonisation, un residu
  de scintillement subsiste d'une image a l'autre ; une moyenne glissante du
  facteur de gain le supprime sans toucher au contenu ;
- **duree proportionnelle a l'ecart reel** : les intervalles allant de 3 a
  130 jours, une cadence constante laisse croire a un rythme regulier et
  exagere visuellement les periodes denses.

L'encodage passe par ffmpeg (H.264) plutot que par le codec `mp4v`, qui
produit du MPEG-4 partie 2 mal lu par les navigateurs et les telephones.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np

from ..config import ParamsVideo
from .incrustation import (
    incruster_avertissement,
    incruster_date,
    incruster_ecart,
    incruster_frise,
)


class EcrivainVideo:
    """Encapsule imageio-ffmpeg, avec repli sur OpenCV si absent."""

    def __init__(self, chemin: Path, fps: int, taille: tuple[int, int], qualite: int = 6):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.taille = taille
        self._imageio = None
        self._cv = None
        try:
            import imageio.v2 as imageio

            self._imageio = imageio.get_writer(
                str(self.chemin),
                fps=fps,
                codec="libx264",
                quality=qualite,
                macro_block_size=8,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        except Exception:
            self._cv = cv2.VideoWriter(
                str(self.chemin),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                taille,
            )

    def ecrire(self, trame_bgr: np.ndarray) -> None:
        if trame_bgr.shape[1] != self.taille[0] or trame_bgr.shape[0] != self.taille[1]:
            trame_bgr = cv2.resize(trame_bgr, self.taille)
        if self._imageio is not None:
            self._imageio.append_data(cv2.cvtColor(trame_bgr, cv2.COLOR_BGR2RGB))
        else:
            self._cv.write(trame_bgr)

    def fermer(self) -> None:
        if self._imageio is not None:
            self._imageio.close()
        elif self._cv is not None:
            self._cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fermer()


def _gains_lisses(
    images_moyennes: list[float], fenetre: int
) -> list[float]:
    """Facteurs de correction d'exposition, lisses dans le temps."""
    if fenetre < 2 or len(images_moyennes) < fenetre:
        cible = float(np.median(images_moyennes))
        return [cible / max(m, 1e-6) for m in images_moyennes]

    valeurs = np.array(images_moyennes, dtype=np.float64)
    noyau = np.ones(fenetre) / fenetre
    rembourre = np.pad(valeurs, fenetre // 2, mode="edge")
    lisse = np.convolve(rembourre, noyau, mode="valid")[: len(valeurs)]
    return [float(l / max(v, 1e-6)) for v, l in zip(valeurs, lisse)]


def _nb_trames(ecart_jours: int | None, params: ParamsVideo, ecart_median: float) -> int:
    if not params.duree_proportionnelle or ecart_jours is None:
        return params.trames_par_date
    proportion = ecart_jours / max(ecart_median, 1.0)
    return int(
        np.clip(
            round(params.trames_par_date * proportion),
            params.trames_min,
            params.trames_max,
        )
    )


def generer_timelapse(
    pipeline,
    base,
    oeil: str,
    cible: Path,
    params: ParamsVideo | None = None,
    largeur_max: int = 1600,
    rappel=None,
) -> Path | None:
    """Produit le timelapse d'un oeil a partir des seances recalees."""
    params = params or pipeline.params.video
    seances = pipeline.seances_recalees(oeil, valides_seulement=False)
    if len(seances) < 2:
        return None

    dates = [s.date for s in seances]
    ecarts = [None] + [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    ecart_median = float(np.median([e for e in ecarts if e])) if len(dates) > 1 else 1.0
    evenements = base.evenements()
    invalides = {c for c, _ in base.recalages_invalides(oeil)}

    # Passe 1 : niveau moyen de chaque trame, pour le lissage d'exposition.
    moyennes = []
    for s in seances:
        img = pipeline.lire_recale(oeil, s.date)
        if img is None:
            moyennes.append(1.0)
            continue
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        utile = gris[gris > 0]
        moyennes.append(float(utile.mean()) if utile.size else 1.0)
    gains = _gains_lisses(moyennes, params.lissage_exposition)

    # Dimensions de sortie (paires, exigees par H.264).
    premiere = pipeline.lire_recale(oeil, dates[0])
    if premiere is None:
        return None
    facteur = min(1.0, largeur_max / premiere.shape[1])
    largeur = int(premiere.shape[1] * facteur) // 2 * 2
    hauteur = int(premiere.shape[0] * facteur) // 2 * 2

    with EcrivainVideo(cible, params.fps, (largeur, hauteur), params.qualite) as sortie:
        for i, seance in enumerate(seances):
            if rappel:
                rappel("timelapse", i + 1, len(seances), str(seance.date))
            image = pipeline.lire_recale(oeil, seance.date)
            if image is None:
                continue

            image = np.clip(image.astype(np.float32) * gains[i], 0, 255).astype(np.uint8)
            trame = cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_AREA)

            incruster_date(trame, seance.date)
            incruster_ecart(trame, ecarts[i])
            incruster_frise(trame, seance.date, dates[0], dates[-1], evenements)
            if seance.cliches_actifs and all(
                c.id in invalides for c in seance.cliches_actifs
            ):
                incruster_avertissement(trame, "recalage a revoir")

            for _ in range(_nb_trames(ecarts[i], params, ecart_median)):
                sortie.ecrire(trame)

    return cible
