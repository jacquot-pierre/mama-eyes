"""Segmentation absolue des druses.

Les druses sont des depots jaunatres, plus clairs que le fond, de forme
approximativement ronde. Le canal de travail combine le canal vert corrige et
le canal b* de LAB (axe bleu <-> jaune) - ce dernier etant pleinement
exploitable ici, les sources etant encodees en 4:4:4 sans sous-echantillonnage
de chrominance.

Le seuillage est **local** : un seuil global echouerait, la luminosite de fond
variant du centre a la peripherie.

Limites : la photographie du fond d'oeil est une modalite mediocre pour
*quantifier* les druses (la reference est l'OCT). Les petites druses dures
sont proches du plancher de bruit, les pseudodruses reticulees sont
pratiquement invisibles. Le seuil depend de la mise au point, de la
transparence des milieux et du flash : le nombre detecte derive avec la
qualite d'image autant qu'avec la maladie. C'est un indicateur de tendance sur
une serie homogene, pas une mesure.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from ..config import ParamsDruses
from ..domain import Druse, Reperes


def canal_druses(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    params: ParamsDruses | None = None,
    normaliser_par_image: bool = True,
) -> np.ndarray:
    """Canal de travail : melange du vert corrige et de la jaunite (b*).

    Renvoie un uint8 ou les druses sont claires.

    `normaliser_par_image` etire le contraste sur les percentiles de l'image.
    C'est utile pour une **detection absolue**, ou chaque image est traitee
    isolement, mais il faut imperativement le desactiver pour une
    **comparaison entre deux dates** : un etirement dependant du contenu est
    un rehaussement adaptatif, et l'apparition d'une lesion deplacerait les
    percentiles, donc l'echelle - exactement le defaut qu'on reproche a un
    CLAHE applique avant la mesure. L'harmonisation inter-images doit rester
    une operation unique et explicite (cf. `differentiel.ajuster_gain`).
    """
    params = params or ParamsDruses()

    vert = image_bgr[:, :, 1].astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    jaune = lab[:, :, 2].astype(np.float32)  # b* : eleve = jaune

    if normaliser_par_image:

        def mettre_a_echelle(x):
            interieur = masque > 0
            if not interieur.any():
                return np.zeros_like(x)
            lo, hi = np.percentile(x[interieur], [1, 99])
            if hi - lo < 1e-6:
                return np.zeros_like(x)
            return np.clip((x - lo) / (hi - lo), 0, 1) * 255.0

    else:

        def mettre_a_echelle(x):
            # Echelle fixe : la transformation ne depend pas du contenu.
            return x

    total = max(params.poids_vert + params.poids_jaune, 1e-6)
    melange = (
        params.poids_vert * mettre_a_echelle(vert)
        + params.poids_jaune * mettre_a_echelle(jaune)
    ) / total

    sortie = np.clip(melange, 0, 255).astype(np.uint8)
    sortie[masque == 0] = 0
    return sortie


def _seuillage_local(canal: np.ndarray, bloc: int, offset: float) -> np.ndarray:
    """Seuil adaptatif : le fond local est estime par moyenne glissante.

    `cv2.adaptiveThreshold` est bien plus rapide que `threshold_local` de
    scikit-image a cette resolution, pour un resultat equivalent.
    """
    bloc = max(3, int(bloc) | 1)
    return cv2.adaptiveThreshold(
        canal,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        bloc,
        offset,
    )


def detecter_druses(
    image_bgr: np.ndarray,
    masque_champ: np.ndarray,
    masque_vaisseaux: np.ndarray | None = None,
    reperes: Reperes | None = None,
    params: ParamsDruses | None = None,
) -> tuple[list[Druse], np.ndarray]:
    """Segmente les druses. Renvoie (liste d'objets, masque binaire).

    Le `seance_id` des objets renvoyes vaut 0 : il est renseigne par
    l'appelant.
    """
    params = params or ParamsDruses()
    canal = canal_druses(image_bgr, masque_champ, params)

    binaire = _seuillage_local(canal, params.bloc_seuil_px, params.offset_seuil)
    binaire[masque_champ == 0] = 0

    # Le bord du champ produit une transition franche que le seuillage local
    # interprete comme une lesion : on l'erode franchement.
    bord = cv2.erode(
        masque_champ,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    binaire[bord == 0] = 0

    if params.exclure_vaisseaux and masque_vaisseaux is not None:
        binaire[masque_vaisseaux > 0] = 0

    if params.exclure_disque and reperes is not None:
        cv2.circle(
            binaire,
            (int(round(reperes.disque_x)), int(round(reperes.disque_y))),
            max(1, int(round(reperes.disque_rayon_px * params.rayon_exclusion_disque))),
            0,
            -1,
        )

    # Ouverture : supprime les grains isoles sans eroder les vraies lesions.
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_OPEN, noyau)

    booleen = binaire > 0
    if params.surface_min_px > 1:
        booleen = remove_small_objects(booleen, max_size=params.surface_min_px)

    etiquettes = label(booleen)
    canal_f = canal.astype(np.float32)

    druses: list[Druse] = []
    retenu = np.zeros_like(binaire)
    for region in regionprops(etiquettes, intensity_image=canal_f):
        if region.area < params.surface_min_px or region.area > params.surface_max_px:
            continue
        if region.eccentricity > params.excentricite_max:
            continue
        if region.solidity < params.solidite_min:
            continue
        cy, cx = region.centroid
        druses.append(
            Druse(
                seance_id=0,
                piste_id=None,
                cx=float(cx),
                cy=float(cy),
                surface_px=int(region.area),
                excentricite=float(region.eccentricity),
                solidite=float(region.solidity),
                intensite_moy=float(region.intensity_mean),
            )
        )
        retenu[etiquettes == region.label] = 255

    return druses, retenu


def dessiner_druses(
    image_bgr: np.ndarray,
    masque_druses: np.ndarray,
    couleur: tuple[int, int, int] = (0, 220, 255),
    epaisseur: int = 2,
) -> np.ndarray:
    """Superpose les contours des druses detectees."""
    sortie = image_bgr.copy()
    contours, _ = cv2.findContours(
        masque_druses, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(sortie, contours, -1, couleur, epaisseur)
    return sortie


def statistiques_grille(
    druses: list[Druse],
    masques_grille: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Repartit les druses dans les anneaux de la grille maculaire.

    Renvoie une liste de (nombre, surface totale) par anneau.
    """
    resultat = []
    for masque in masques_grille:
        nombre = 0
        surface = 0
        for d in druses:
            y, x = int(round(d.cy)), int(round(d.cx))
            if 0 <= y < masque.shape[0] and 0 <= x < masque.shape[1]:
                if masque[y, x] > 0:
                    nombre += 1
                    surface += d.surface_px
        resultat.append((nombre, surface))
    return resultat
