"""Recalage a deux etages.

Etage 1 - points caracteristiques : SIFT sur le canal vert rehausse par
CLAHE, **restreint au masque du champ**. Sans ce masque, le detecteur trouve
des points sur la frontiere du disque noir, qui depend du cadrage et non de
l'anatomie. Appariement par ratio de Lowe puis RANSAC estimant une
**similarite a 4 DDL** (rotation, translation, echelle uniforme).

Jamais d'homographie : ses 8 DDL laissent assez de liberte pour « expliquer »
une partie du changement pathologique par une deformation geometrique, et
donc pour l'effacer du signal que l'on cherche a mesurer.

Etage 2 - raffinement par intensite (ECC), applique aux **cartes vasculaires**
plutot qu'aux images brutes : les vaisseaux sont stables dans le temps alors
que l'apparence retinienne change avec la pathologie.

Le gabarit de reference est construit par iteration de groupe, et non pris au
premier cliche de la serie - qui, apres 7 ans de derive, en est le pire
representant.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ParamsRecalage, ParamsVaisseaux
from .illumination import clahe_affichage
from .vaisseaux import carte_vaisseaux

IDENTITE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass
class Transformation:
    """Matrice affine 2x3 accompagnee de son diagnostic de qualite."""

    matrice: np.ndarray
    inliers: int
    residu_rms: float
    valide: bool
    methode: str = "sift"

    @property
    def est_identite(self) -> bool:
        return bool(np.allclose(self.matrice, IDENTITE, atol=1e-9))

    def echelle(self) -> float:
        a, b = self.matrice[0, 0], self.matrice[0, 1]
        return float(np.hypot(a, b))

    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrice[1, 0], self.matrice[0, 0])))

    def translation(self) -> tuple[float, float]:
        return float(self.matrice[0, 2]), float(self.matrice[1, 2])

    def __str__(self) -> str:
        etat = "ok" if self.valide else "ECHEC"
        dx, dy = self.translation()
        return (
            f"[{etat}] {self.inliers} inliers, rms={self.residu_rms:.2f} px, "
            f"rot={self.rotation_deg():+.2f} deg, ech={self.echelle():.4f}, "
            f"t=({dx:+.1f},{dy:+.1f})"
        )


def _preparer_pour_features(image_bgr: np.ndarray) -> np.ndarray:
    """Canal vert rehausse : c'est la que les vaisseaux contrastent le mieux."""
    rehausse = clahe_affichage(image_bgr)
    return rehausse[:, :, 1]


def _mettre_a_echelle(matrice: np.ndarray, facteur: float) -> np.ndarray:
    """Convertit une transformation estimee a l'echelle `facteur` vers 1:1."""
    m = matrice.copy().astype(np.float64)
    m[0, 2] /= facteur
    m[1, 2] /= facteur
    return m


def estimer_transformation(
    source_bgr: np.ndarray,
    cible_bgr: np.ndarray,
    masque_source: np.ndarray,
    masque_cible: np.ndarray,
    params: ParamsRecalage | None = None,
) -> Transformation:
    """Estime la transformation qui amene `source` sur `cible`."""
    params = params or ParamsRecalage()
    f = params.echelle_travail

    def reduire(img, msk):
        if f >= 0.999:
            return img, msk
        petit = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        pmsk = cv2.resize(msk, (petit.shape[1], petit.shape[0]), interpolation=cv2.INTER_NEAREST)
        return petit, pmsk

    src_img, src_msk = reduire(source_bgr, masque_source)
    dst_img, dst_msk = reduire(cible_bgr, masque_cible)

    src_gris = _preparer_pour_features(src_img)
    dst_gris = _preparer_pour_features(dst_img)

    if params.detecteur.upper() == "ORB":
        detecteur = cv2.ORB_create(nfeatures=params.nb_features)
        norme = cv2.NORM_HAMMING
    else:
        detecteur = cv2.SIFT_create(nfeatures=params.nb_features)
        norme = cv2.NORM_L2

    # Le masque restreint la detection a l'anatomie (cf. docstring du module).
    kp1, des1 = detecteur.detectAndCompute(src_gris, src_msk)
    kp2, des2 = detecteur.detectAndCompute(dst_gris, dst_msk)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return Transformation(IDENTITE.copy(), 0, float("inf"), False, "aucun_point")

    apparieur = cv2.BFMatcher(norme, crossCheck=False)
    couples = apparieur.knnMatch(des1, des2, k=2)
    bons = [
        m
        for paire in couples
        if len(paire) == 2
        for m, n in [paire]
        if m.distance < params.ratio_lowe * n.distance
    ]

    if len(bons) < params.inliers_min:
        return Transformation(
            IDENTITE.copy(), len(bons), float("inf"), False, "appariements_insuffisants"
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)

    if params.modele == "affine":
        matrice, masque_in = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=params.seuil_ransac_px * f,
            maxIters=5000,
            confidence=0.995,
        )
    else:
        matrice, masque_in = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=params.seuil_ransac_px * f,
            maxIters=5000,
            confidence=0.995,
        )

    if matrice is None:
        return Transformation(IDENTITE.copy(), 0, float("inf"), False, "ransac_echec")

    masque_in = masque_in.ravel().astype(bool)
    nb_inliers = int(masque_in.sum())

    # Residu sur les seuls inliers, ramene a l'echelle 1:1.
    src_in = src_pts[masque_in].reshape(-1, 2)
    dst_in = dst_pts[masque_in].reshape(-1, 2)
    if nb_inliers >= 3:
        projete = (matrice[:, :2] @ src_in.T).T + matrice[:, 2]
        residu = float(np.sqrt(((projete - dst_in) ** 2).sum(axis=1).mean()) / f)
    else:
        residu = float("inf")

    matrice = _mettre_a_echelle(matrice, f)
    valide = nb_inliers >= params.inliers_min and residu <= params.residu_max_px
    return Transformation(matrice, nb_inliers, residu, valide, "sift")


def raffiner_ecc(
    source_bgr: np.ndarray,
    cible_bgr: np.ndarray,
    masque_source: np.ndarray,
    masque_cible: np.ndarray,
    initiale: np.ndarray,
    params: ParamsRecalage | None = None,
    params_vaisseaux: ParamsVaisseaux | None = None,
) -> tuple[np.ndarray, bool]:
    """Affine la transformation par ECC sur les cartes vasculaires.

    Recaler sur les vaisseaux plutot que sur l'intensite brute evite que la
    lesion elle-meme n'influence l'alignement.
    """
    params = params or ParamsRecalage()
    if not params.ecc_actif:
        return initiale, False

    f = min(params.echelle_travail, 0.5)
    src = cv2.resize(source_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    dst = cv2.resize(cible_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    m_src = cv2.resize(masque_source, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_dst = cv2.resize(masque_cible, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_NEAREST)

    v_src = carte_vaisseaux(src, m_src, params_vaisseaux, echelle=f)
    v_dst = carte_vaisseaux(dst, m_dst, params_vaisseaux, echelle=f)

    depart = initiale.copy().astype(np.float32)
    depart[0, 2] *= f
    depart[1, 2] *= f

    criteres = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        params.ecc_iterations,
        params.ecc_epsilon,
    )
    try:
        _, matrice = cv2.findTransformECC(
            v_dst,
            v_src,
            depart,
            cv2.MOTION_AFFINE if params.modele == "affine" else cv2.MOTION_EUCLIDEAN,
            criteres,
            m_dst,
            5,
        )
    except cv2.error:
        return initiale, False

    matrice = matrice.astype(np.float64)
    matrice[0, 2] /= f
    matrice[1, 2] /= f

    # Garde-fou : un ECC qui derape doit etre rejete, pas propage.
    ecart = np.abs(matrice - initiale)
    if ecart[:, 2].max() > 80 or ecart[:, :2].max() > 0.08:
        return initiale, False
    return matrice, True


def appliquer(
    image: np.ndarray,
    matrice: np.ndarray,
    forme: tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Applique une transformation affine. **Une seule interpolation**.

    Composez toujours les transformations avant d'appeler cette fonction :
    chaque warp successif ajoute du flou qui degrade la detection des petites
    druses.
    """
    hauteur, largeur = forme
    return cv2.warpAffine(
        image,
        matrice.astype(np.float32),
        (largeur, hauteur),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def composer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose deux transformations affines 2x3 : resultat = a apres b."""
    A = np.vstack([a, [0, 0, 1]])
    B = np.vstack([b, [0, 0, 1]])
    return (A @ B)[:2, :]


def inverser(matrice: np.ndarray) -> np.ndarray:
    M = np.vstack([matrice, [0, 0, 1]])
    return np.linalg.inv(M)[:2, :]
