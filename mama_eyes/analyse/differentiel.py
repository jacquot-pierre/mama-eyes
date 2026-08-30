"""Comparaison entre deux images recalees.

Une derniere normalisation s'impose *juste avant* la soustraction : on ajuste
un modele ``I_b ~ a * I_a + b`` par regression robuste sur les seuls pixels de
fond (hors vaisseaux, hors lesions). Cela elimine le gain et l'offset
residuels qui, sinon, dominent la difference.

La difference est **signee** et en virgule flottante : positif = apparition
ou eclaircissement, negatif = disparition ou attenuation. Une soustraction
entiere saturerait les negatifs a zero et perdrait silencieusement une des
deux directions du changement.

Limite : la difference amplifie l'erreur de recalage. Un decalage de 2 px sur
un bord de vaisseau tres contraste produit un signal plus fort qu'une vraie
druse - d'ou l'exclusion du masque vasculaire dilate et le leger flou
prealable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from ..config import ParamsDifferentiel
from .druses import canal_druses


@dataclass
class DifferenceSignee:
    """Resultat d'une comparaison entre deux dates."""

    difference: np.ndarray  # float32, signee
    masque_apparition: np.ndarray  # uint8
    masque_disparition: np.ndarray  # uint8
    gain: float
    offset: float
    zone_valide: np.ndarray

    @property
    def surface_apparue(self) -> int:
        return int((self.masque_apparition > 0).sum())

    @property
    def surface_disparue(self) -> int:
        return int((self.masque_disparition > 0).sum())

    def __str__(self) -> str:
        return (
            f"gain={self.gain:.3f} offset={self.offset:+.1f} | "
            f"apparu={self.surface_apparue} px, disparu={self.surface_disparue} px"
        )


def ajuster_gain(
    canal_a: np.ndarray,
    canal_b: np.ndarray,
    zone: np.ndarray,
    iterations: int = 3,
) -> tuple[float, float]:
    """Regression robuste ``b ~ gain * a + offset`` sur les pixels de fond.

    Les iterations ecartent progressivement les points aberrants : ce sont
    precisement les lesions, qu'il ne faut surtout pas laisser influencer la
    normalisation censee les reveler.
    """
    valides = zone > 0
    if valides.sum() < 100:
        return 1.0, 0.0

    a = canal_a[valides].astype(np.float64)
    b = canal_b[valides].astype(np.float64)

    # Sous-echantillonnage : 200 000 points suffisent et bornent le cout.
    if a.size > 200_000:
        pas = a.size // 200_000
        a, b = a[::pas], b[::pas]

    garde = np.ones(a.shape, dtype=bool)
    gain, offset = 1.0, 0.0
    for _ in range(iterations):
        if garde.sum() < 50:
            break
        va, vb = a[garde], b[garde]
        variance = float(va.var())
        # Regression mal conditionnee : sans dynamique dans `a`, la pente
        # cov/var est instable. On se rabat sur une correction d'offset seul,
        # qui reste juste, plutot que de propager un gain aberrant.
        if variance < 1.0:
            return 1.0, float(vb.mean() - va.mean())
        gain = float(((va - va.mean()) * (vb - vb.mean())).mean() / variance)
        offset = float(vb.mean() - gain * va.mean())
        residus = np.abs(b - (gain * a + offset))
        seuil = np.percentile(residus, 90)
        garde = residus <= max(seuil, 1e-6)

    if not np.isfinite(gain) or not (0.2 < gain < 5.0):
        # Gain hors de tout domaine plausible : correction d'offset seul.
        return 1.0, float(b.mean() - a.mean())
    return gain, offset


def comparer(
    image_a: np.ndarray,
    image_b: np.ndarray,
    masque_champ: np.ndarray,
    masque_vaisseaux: np.ndarray | None = None,
    params: ParamsDifferentiel | None = None,
) -> DifferenceSignee:
    """Compare `a` (reference ou date anterieure) a `b` (date posterieure)."""
    params = params or ParamsDifferentiel()

    # Echelle fixe des deux cotes : toute normalisation dependant du contenu
    # fausserait la difference (cf. docstring de `canal_druses`).
    canal_a = canal_druses(
        image_a, masque_champ, normaliser_par_image=False
    ).astype(np.float32)
    canal_b = canal_druses(
        image_b, masque_champ, normaliser_par_image=False
    ).astype(np.float32)

    if params.flou_pre_diff_px > 0:
        canal_a = cv2.GaussianBlur(canal_a, (0, 0), params.flou_pre_diff_px)
        canal_b = cv2.GaussianBlur(canal_b, (0, 0), params.flou_pre_diff_px)

    # Zone de comparaison : champ commun, erode, prive des vaisseaux dilates.
    zone = masque_champ.copy()
    zone[(image_a.sum(axis=2) == 0) | (image_b.sum(axis=2) == 0)] = 0
    zone = cv2.erode(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    if masque_vaisseaux is not None and params.dilater_exclusion_vaisseaux_px > 1:
        k = params.dilater_exclusion_vaisseaux_px | 1
        dilate = cv2.dilate(
            masque_vaisseaux, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        zone[dilate > 0] = 0

    gain, offset = (1.0, 0.0)
    if params.normaliser_gain:
        gain, offset = ajuster_gain(canal_a, canal_b, zone)

    difference = canal_b - (gain * canal_a + offset)
    difference[zone == 0] = 0.0

    seuil = params.seuil_diff
    apparition = (difference > seuil) & (zone > 0)
    disparition = (difference < -seuil) & (zone > 0)

    if params.surface_min_px > 1:
        apparition = remove_small_objects(apparition, max_size=params.surface_min_px)
        disparition = remove_small_objects(disparition, max_size=params.surface_min_px)

    return DifferenceSignee(
        difference=difference.astype(np.float32),
        masque_apparition=(apparition.astype(np.uint8) * 255),
        masque_disparition=(disparition.astype(np.uint8) * 255),
        gain=gain,
        offset=offset,
        zone_valide=zone,
    )


def objets_de_masque(masque: np.ndarray, surface_min: int = 12) -> list[dict]:
    """Composantes connexes d'un masque de changement."""
    etiquettes = label(masque > 0)
    objets = []
    for region in regionprops(etiquettes):
        if region.area < surface_min:
            continue
        cy, cx = region.centroid
        objets.append(
            {"cx": float(cx), "cy": float(cy), "surface_px": int(region.area)}
        )
    return objets


def carte_de_chaleur(
    difference: np.ndarray, amplitude: float = 40.0
) -> np.ndarray:
    """Rend une difference signee en fausses couleurs.

    Bleu = disparition, rouge = apparition, neutre = pas de changement.
    """
    normalise = np.clip(difference / amplitude, -1.0, 1.0)
    hauteur, largeur = normalise.shape
    sortie = np.zeros((hauteur, largeur, 3), dtype=np.float32)
    sortie[:, :, 2] = np.clip(normalise, 0, 1) * 255  # rouge : apparition
    sortie[:, :, 0] = np.clip(-normalise, 0, 1) * 255  # bleu : disparition
    return sortie.astype(np.uint8)
