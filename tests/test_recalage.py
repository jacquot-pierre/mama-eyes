"""Recalage : verifie qu'une transformation connue est retrouvee au pixel pres.

Ce test isole l'etage de recalage du reste de la chaine : on injecte une
rotation et une translation connues dans une image synthetique riche en
structures tubulaires, puis on verifie que l'estimation les retrouve.
"""

import cv2
import numpy as np
import pytest

from mama_eyes.pretraitement.recalage import (
    appliquer,
    composer,
    estimer_transformation,
    inverser,
)


def _fond_synthetique(taille: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Faux fond d'oeil : disque clair, arbre vasculaire, taches claires."""
    rng = np.random.default_rng(1234)
    image = np.zeros((taille, taille, 3), dtype=np.uint8)
    centre = (taille // 2, taille // 2)
    rayon = int(taille * 0.45)

    cv2.circle(image, centre, rayon, (40, 90, 165), -1)

    # Arbre vasculaire : branches rectilignes puis ramifications.
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        x0 = int(centre[0] + 0.10 * rayon * np.cos(angle))
        y0 = int(centre[1] + 0.10 * rayon * np.sin(angle))
        x1 = int(centre[0] + 0.88 * rayon * np.cos(angle))
        y1 = int(centre[1] + 0.88 * rayon * np.sin(angle))
        cv2.line(image, (x0, y0), (x1, y1), (18, 40, 95), 7)
        milieu = ((x0 + x1) // 2, (y0 + y1) // 2)
        for delta in (-0.45, 0.45):
            x2 = int(milieu[0] + 0.35 * rayon * np.cos(angle + delta))
            y2 = int(milieu[1] + 0.35 * rayon * np.sin(angle + delta))
            cv2.line(image, milieu, (x2, y2), (18, 40, 95), 4)

    # Disque optique et lesions claires : donnent des points caracteristiques.
    cv2.circle(image, (centre[0] + rayon // 2, centre[1]), rayon // 8, (150, 205, 240), -1)
    for _ in range(60):
        x = int(rng.uniform(centre[0] - rayon * 0.7, centre[0] + rayon * 0.7))
        y = int(rng.uniform(centre[1] - rayon * 0.7, centre[1] + rayon * 0.7))
        cv2.circle(image, (x, y), int(rng.uniform(4, 11)), (110, 190, 225), -1)

    masque = np.zeros((taille, taille), dtype=np.uint8)
    cv2.circle(masque, centre, int(rayon * 0.95), 255, -1)
    return image, masque


def _matrice(rotation_deg: float, tx: float, ty: float, centre) -> np.ndarray:
    m = cv2.getRotationMatrix2D(centre, rotation_deg, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


@pytest.mark.parametrize(
    "rotation,tx,ty",
    [(0.0, 18.0, -12.0), (3.0, 0.0, 0.0), (-2.5, 25.0, 14.0), (5.0, -30.0, 20.0)],
)
def test_transformation_connue_retrouvee(rotation, tx, ty):
    image, masque = _fond_synthetique()
    centre = (image.shape[1] / 2, image.shape[0] / 2)
    verite = _matrice(rotation, tx, ty, centre)

    deformee = appliquer(image, verite, image.shape[:2])
    masque_deforme = appliquer(masque, verite, masque.shape[:2])

    # On cherche la transformation qui ramene `deformee` sur `image` :
    # elle doit etre l'inverse de celle appliquee.
    estimee = estimer_transformation(deformee, image, masque_deforme, masque)

    assert estimee.valide, f"recalage invalide : {estimee}"

    attendue = inverser(verite)
    ecart = composer(estimee.matrice, verite)  # doit approcher l'identite
    identite = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert np.allclose(ecart[:, :2], identite[:, :2], atol=0.02), (
        f"partie lineaire incorrecte :\n{ecart}"
    )
    assert np.abs(ecart[:, 2]).max() < 3.0, f"translation residuelle : {ecart[:, 2]}"
    assert estimee.residu_rms < 3.0
    del attendue


def test_identite_sur_image_identique():
    image, masque = _fond_synthetique()
    estimee = estimer_transformation(image, image, masque, masque)
    assert estimee.valide
    assert np.abs(estimee.matrice[:, 2]).max() < 1.0
    assert abs(estimee.echelle() - 1.0) < 0.01
    assert abs(estimee.rotation_deg()) < 0.5


def test_similarite_n_a_que_quatre_degres_de_liberte():
    """Le modele par defaut ne doit pas pouvoir introduire de cisaillement.

    Une homographie, elle, disposerait d'assez de liberte pour absorber une
    partie du changement pathologique - c'est precisement ce qu'on evite.
    """
    image, masque = _fond_synthetique()
    centre = (image.shape[1] / 2, image.shape[0] / 2)
    verite = _matrice(4.0, 10.0, -8.0, centre)
    deformee = appliquer(image, verite, image.shape[:2])
    masque_deforme = appliquer(masque, verite, masque.shape[:2])

    estimee = estimer_transformation(deformee, image, masque_deforme, masque)
    m = estimee.matrice
    # Une similarite verifie m[0,0] == m[1,1] et m[0,1] == -m[1,0].
    assert abs(m[0, 0] - m[1, 1]) < 1e-6
    assert abs(m[0, 1] + m[1, 0]) < 1e-6


def test_echec_signale_sur_images_sans_rapport():
    """Un echec doit etre signale, jamais silencieux."""
    image, masque = _fond_synthetique()
    bruit = np.random.default_rng(7).integers(0, 255, image.shape, dtype=np.uint8)
    estimee = estimer_transformation(bruit, image, masque, masque)
    assert not estimee.valide
