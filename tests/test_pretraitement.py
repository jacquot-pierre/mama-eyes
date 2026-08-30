"""Prétraitement : champ, vignettage, differentiel.

Les tests portent sur des proprietes verifiables plutot que sur des valeurs
figees : le masque doit couvrir le disque, la correction doit *reduire* la
non-uniformite, la difference doit rester nulle entre deux images identiques.
"""

import cv2
import numpy as np
import pytest

from mama_eyes.analyse.differentiel import ajuster_gain, comparer
from mama_eyes.config import ParamsDifferentiel
from mama_eyes.pretraitement.champ import detecter_champ
from mama_eyes.pretraitement.illumination import (
    corriger_vignettage,
    harmoniser_lab,
    planeite,
    stats_lab,
    stats_medianes,
)


def _image_test(taille=600, rayon_relatif=0.45, vignettage=0.0):
    """Disque retinien uniforme, avec vignettage radial optionnel."""
    image = np.zeros((taille, taille, 3), dtype=np.uint8)
    centre = (taille // 2, taille // 2)
    rayon = int(taille * rayon_relatif)
    cv2.circle(image, centre, rayon, (45, 95, 170), -1)

    rng = np.random.default_rng(3)
    for _ in range(40):
        x = int(rng.uniform(centre[0] - rayon * 0.6, centre[0] + rayon * 0.6))
        y = int(rng.uniform(centre[1] - rayon * 0.6, centre[1] + rayon * 0.6))
        cv2.circle(image, (x, y), 7, (120, 195, 230), -1)

    if vignettage > 0:
        yy, xx = np.mgrid[0:taille, 0:taille]
        distance = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / rayon
        attenuation = np.clip(1.0 - vignettage * distance**2, 0.15, 1.0)
        image = np.clip(image * attenuation[:, :, None], 0, 255).astype(np.uint8)
    return image, centre, rayon


# --- champ retinien ----------------------------------------------------------


def test_champ_localise_le_disque():
    image, centre, rayon = _image_test()
    champ = detecter_champ(image)
    assert abs(champ.cx - centre[0]) < 12
    assert abs(champ.cy - centre[1]) < 12
    # L'erosion de 5 % est volontaire : le bord fausse les statistiques.
    assert 0.85 * rayon < champ.rayon <= rayon


def test_masque_couvre_le_disque_sans_deborder():
    image, _, _ = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()
    interieur = masque > 0
    exterieur = ~interieur
    # A l'interieur, l'image est coloree ; a l'exterieur, elle est noire.
    assert image[interieur].mean() > 40
    assert image[exterieur].mean() < 25


def test_rectangle_englobant_dans_les_bornes():
    image, _, _ = _image_test()
    champ = detecter_champ(image)
    x0, y0, x1, y1 = champ.rectangle(marge=8)
    assert 0 <= x0 < x1 <= image.shape[1]
    assert 0 <= y0 < y1 <= image.shape[0]


# --- vignettage --------------------------------------------------------------


@pytest.mark.parametrize("force", [0.3, 0.5, 0.7])
def test_correction_reduit_la_non_uniformite(force):
    image, _, rayon = _image_test(vignettage=force)
    champ = detecter_champ(image)
    masque = champ.masque()
    sigma = champ.rayon * 0.125

    avant = planeite(image, masque, sigma)
    corrigee = corriger_vignettage(image, masque, champ.rayon)
    apres = planeite(corrigee, masque, sigma)

    assert apres < avant, f"vignettage non corrige : {avant:.2f} -> {apres:.2f}"
    assert apres < avant * 0.6


def test_correction_ne_degrade_pas_une_image_deja_uniforme():
    image, _, _ = _image_test(vignettage=0.0)
    champ = detecter_champ(image)
    masque = champ.masque()
    sigma = champ.rayon * 0.125
    avant = planeite(image, masque, sigma)
    apres = planeite(corriger_vignettage(image, masque, champ.rayon), masque, sigma)
    assert apres <= avant + 0.6


def test_exterieur_du_champ_reste_noir():
    image, _, _ = _image_test(vignettage=0.4)
    champ = detecter_champ(image)
    masque = champ.masque()
    corrigee = corriger_vignettage(image, masque, champ.rayon)
    assert corrigee[masque == 0].max() == 0


# --- harmonisation colorimetrique -------------------------------------------

def test_harmonisation_rapproche_des_statistiques_cibles():
    image_a, _, _ = _image_test()
    image_b = np.clip(image_a.astype(np.float32) * 0.7 + 18, 0, 255).astype(np.uint8)

    champ = detecter_champ(image_a)
    masque = champ.masque()
    image_b[masque == 0] = 0

    cible = stats_lab(image_a, masque)
    avant = stats_lab(image_b, masque)
    apres = stats_lab(harmoniser_lab(image_b, masque, cible), masque)

    assert abs(apres.l_moy - cible.l_moy) < abs(avant.l_moy - cible.l_moy)
    assert abs(apres.l_moy - cible.l_moy) < 3.0


def test_statistiques_medianes_resistent_a_un_aberrant():
    image, _, _ = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()

    normale = stats_lab(image, masque)
    aberrante = stats_lab(
        np.clip(image.astype(np.float32) * 0.2, 0, 255).astype(np.uint8), masque
    )
    mediane = stats_medianes([normale, normale, normale, aberrante])
    assert abs(mediane.l_moy - normale.l_moy) < 2.0


# --- differentiel ------------------------------------------------------------


def test_difference_nulle_entre_images_identiques():
    """Le controle negatif le plus elementaire."""
    image, _, _ = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()
    diff = comparer(image, image, masque)
    assert diff.surface_apparue == 0
    assert diff.surface_disparue == 0
    assert abs(diff.gain - 1.0) < 0.02


def test_gain_global_est_neutralise():
    """Un changement d'exposition ne doit pas produire de fausse detection."""
    image, _, _ = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()
    plus_clair = np.clip(image.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    plus_clair[masque == 0] = 0

    diff = comparer(image, plus_clair, masque)
    assert diff.surface_apparue < 400
    assert diff.surface_disparue < 400


def test_lesion_ajoutee_est_detectee_comme_apparition():
    image, centre, rayon = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()

    avec_lesion = image.copy()
    cv2.circle(avec_lesion, (centre[0] + 40, centre[1] - 30), 14, (170, 235, 250), -1)

    diff = comparer(image, avec_lesion, masque)
    assert diff.surface_apparue > 100
    assert diff.surface_apparue > diff.surface_disparue


def test_difference_est_signee_dans_les_deux_sens():
    """Une soustraction entiere saturerait les negatifs et perdrait un sens."""
    image, centre, _ = _image_test()
    champ = detecter_champ(image)
    masque = champ.masque()

    avec_lesion = image.copy()
    cv2.circle(avec_lesion, (centre[0] + 40, centre[1] - 30), 14, (170, 235, 250), -1)

    apparition = comparer(image, avec_lesion, masque)
    disparition = comparer(avec_lesion, image, masque)

    assert apparition.surface_apparue > 100
    assert disparition.surface_disparue > 100
    assert disparition.difference.min() < -ParamsDifferentiel().seuil_diff


def test_ajustement_de_gain_retrouve_les_coefficients():
    rng = np.random.default_rng(11)
    a = rng.uniform(20, 200, (400, 400)).astype(np.float32)
    zone = np.full(a.shape, 255, dtype=np.uint8)
    b = 1.3 * a + 12.0
    gain, offset = ajuster_gain(a, b, zone)
    assert abs(gain - 1.3) < 0.05
    assert abs(offset - 12.0) < 4.0
