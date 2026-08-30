"""Fusion des cliches d'une meme seance.

Mediane pixel a pixel apres recalage mutuel : elle elimine les reflets de
flash et les poussieres qui n'apparaissent que sur une prise, ce qu'une
moyenne etalerait sur toutes.

A n=2 la mediane degenere en moyenne et perd sa robustesse - elle moyenne
l'artefact au lieu de le rejeter. On conserve alors le cliche le plus net.
La fusion ne devient reellement interessante qu'a partir de 3 prises.
"""

from __future__ import annotations

import numpy as np

from ..config import ParamsRecalage
from .recalage import appliquer, estimer_transformation


def fusionner_seance(
    images: list[np.ndarray],
    masques: list[np.ndarray],
    nettetes: list[float] | None = None,
    params: ParamsRecalage | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine les cliches d'une seance en une image de reference.

    Renvoie (image fusionnee, masque commun, journal).
    """
    journal: list[str] = []
    if not images:
        raise ValueError("aucune image a fusionner")

    if len(images) == 1:
        return images[0].copy(), masques[0].copy(), ["1 cliche : aucune fusion"]

    if len(images) == 2:
        # cf. docstring : a n=2 on prefere le plus net a une moyenne.
        if nettetes:
            meilleur = int(np.argmax(nettetes))
        else:
            meilleur = 0
        journal.append(
            f"2 cliches : mediane degeneree en moyenne, "
            f"conservation du plus net (index {meilleur})"
        )
        return images[meilleur].copy(), masques[meilleur].copy(), journal

    # n >= 3 : recalage mutuel sur le plus net, puis mediane.
    reference = int(np.argmax(nettetes)) if nettetes else 0
    journal.append(f"{len(images)} cliches, reference interne = index {reference}")

    ref_img, ref_msk = images[reference], masques[reference]
    forme = ref_img.shape[:2]

    empiles = [ref_img.astype(np.float32)]
    masque_commun = ref_msk.copy()

    for i, (img, msk) in enumerate(zip(images, masques)):
        if i == reference:
            continue
        tr = estimer_transformation(img, ref_img, msk, ref_msk, params)
        if not tr.valide:
            journal.append(f"  cliche {i} ecarte : {tr}")
            continue
        aligne = appliquer(img, tr.matrice, forme)
        msk_aligne = appliquer(msk, tr.matrice, forme)
        empiles.append(aligne.astype(np.float32))
        masque_commun = np.minimum(masque_commun, msk_aligne)
        journal.append(f"  cliche {i} aligne : {tr}")

    if len(empiles) == 1:
        journal.append("aucun cliche secondaire aligne : reference conservee telle quelle")
        return ref_img.copy(), ref_msk.copy(), journal

    if len(empiles) == 2:
        journal.append("un seul cliche aligne : conservation de la reference")
        return ref_img.copy(), masque_commun, journal

    fusion = np.median(np.stack(empiles, axis=0), axis=0)
    fusion = np.clip(fusion, 0, 255).astype(np.uint8)
    fusion[masque_commun == 0] = 0
    journal.append(f"mediane sur {len(empiles)} cliches alignes")
    return fusion, masque_commun, journal
