"""Localisation du disque optique et de la macula.

Les detecteurs automatiques echouent precisement dans le cas qui nous occupe :
un amas de druses confluentes peut etre aussi clair que le disque optique et
le supplanter comme maximum d'intensite ; et le repere de la fovea sombre
disparait en cas de DMLA - atrophie et remaniements rendent la macula
*claire*, la pathologie detruisant le repere cense la localiser.

Pour une application mono-patient, le **clic manuel** est donc le mode
principal : une fois par oeil sur un cliche net, propage ensuite a toute la
serie par les transformations de recalage. Les fonctions automatiques
ci-dessous servent a proposer une position de depart, pas a decider seules.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..domain import Reperes

#: Distance fovea <-> centre du disque, en diametres papillaires.
DISTANCE_FOVEA_DP = 2.5
#: Decalage vertical de la fovea (vers le bas), en diametres papillaires.
DECALAGE_FOVEA_DP = 0.35


def detecter_disque(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    rayon_champ: float,
) -> tuple[float, float, float]:
    """Propose une position et un rayon pour le disque optique.

    Combine forte intensite et forte variance locale : les vaisseaux
    convergent vers le disque, ce qui le distingue d'un simple reflet.

    Renvoie (cx, cy, rayon). A confirmer par l'utilisateur.
    """
    # Le disque fait environ 1/12 du diametre du champ.
    rayon_estime = max(12.0, rayon_champ / 6.0)
    k = int(rayon_estime) | 1

    rouge = image_bgr[:, :, 2].astype(np.float32)
    rouge[masque == 0] = 0

    # Composante 1 : intensite moyenne sur un disque de la bonne taille.
    intensite = cv2.blur(rouge, (k, k))

    # Composante 2 : variance locale (convergence vasculaire).
    vert = image_bgr[:, :, 1].astype(np.float32)
    moyenne = cv2.blur(vert, (k, k))
    variance = cv2.blur(vert * vert, (k, k)) - moyenne * moyenne
    variance = np.maximum(variance, 0)

    def normaliser(x):
        interieur = masque > 0
        if not interieur.any():
            return x
        lo, hi = np.percentile(x[interieur], [1, 99])
        if hi - lo < 1e-6:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0, 1)

    score = 0.65 * normaliser(intensite) + 0.35 * normaliser(variance)

    # On ecarte le bord du champ, ou l'artefact de flash est frequent.
    interieur = cv2.erode(
        masque, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 | 1, k * 2 | 1))
    )
    score[interieur == 0] = 0

    _, _, _, maxloc = cv2.minMaxLoc(score)
    return float(maxloc[0]), float(maxloc[1]), float(rayon_estime)


def predire_macula(
    oeil: str,
    disque_x: float,
    disque_y: float,
    rayon_disque: float,
) -> tuple[float, float]:
    """Position attendue de la fovea a partir du disque optique.

    La fovea se situe a environ 2,5 diametres papillaires du centre du disque,
    du cote **temporal**, legerement en dessous.

    Le cote temporal depend de l'oeil : a gauche de l'image pour l'oeil droit,
    a droite pour l'oeil gauche. C'est exactement la raison pour laquelle
    l'etiquette D/G du nom de fichier doit etre fiable.
    """
    diametre = 2.0 * rayon_disque
    sens = -1.0 if oeil == "D" else 1.0
    return (
        disque_x + sens * DISTANCE_FOVEA_DP * diametre,
        disque_y + DECALAGE_FOVEA_DP * diametre,
    )


def affiner_macula(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    prediction: tuple[float, float],
    rayon_recherche: float,
) -> tuple[float, float]:
    """Cherche la zone la plus sombre autour de la position predite.

    En cas de DMLA avancee ce raffinement peut degrader la prediction
    geometrique : il n'est applique que si un minimum franc existe.
    """
    hauteur, largeur = image_bgr.shape[:2]
    px, py = prediction
    r = int(rayon_recherche)
    x0, x1 = max(0, int(px) - r), min(largeur, int(px) + r)
    y0, y1 = max(0, int(py) - r), min(hauteur, int(py) + r)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return prediction

    fenetre = image_bgr[y0:y1, x0:x1, 1].astype(np.float32)
    m_fen = masque[y0:y1, x0:x1]
    lisse = cv2.GaussianBlur(fenetre, (0, 0), max(3.0, rayon_recherche / 8))
    lisse[m_fen == 0] = 255

    minval, maxval, minloc, _ = cv2.minMaxLoc(lisse)
    # Un minimum trop peu marque n'est pas un repere fiable.
    if maxval - minval < 8:
        return prediction
    return float(x0 + minloc[0]), float(y0 + minloc[1])


def proposer_reperes(
    image_bgr: np.ndarray,
    masque: np.ndarray,
    rayon_champ: float,
    oeil: str,
) -> Reperes:
    """Proposition automatique complete, a valider par l'utilisateur."""
    dx, dy, rayon = detecter_disque(image_bgr, masque, rayon_champ)
    mx, my = predire_macula(oeil, dx, dy, rayon)
    mx, my = affiner_macula(image_bgr, masque, (mx, my), rayon * 1.2)
    return Reperes(
        oeil=oeil,
        disque_x=dx,
        disque_y=dy,
        disque_rayon_px=rayon,
        macula_x=mx,
        macula_y=my,
        manuel=False,
    )


def masque_grille_maculaire(
    forme: tuple[int, int],
    reperes: Reperes,
    rayons_mm: tuple[float, ...],
) -> list[np.ndarray]:
    """Masques concentriques centres sur la macula, facon grille ETDRS.

    Rapporter les mesures a cette grille les rend comparables dans le temps
    meme si le cadrage bouge, puisqu'elle suit l'anatomie et non l'image.
    """
    masques = []
    precedent = None
    for rayon_mm in rayons_mm:
        rayon_px = rayon_mm * reperes.px_par_mm
        m = np.zeros(forme, dtype=np.uint8)
        cv2.circle(
            m,
            (int(round(reperes.macula_x)), int(round(reperes.macula_y))),
            max(1, int(round(rayon_px))),
            255,
            -1,
        )
        anneau = m if precedent is None else cv2.subtract(m, precedent)
        masques.append(anneau)
        precedent = m
    return masques


def masque_exclusion_disque(
    forme: tuple[int, int], reperes: Reperes, facteur: float = 1.4
) -> np.ndarray:
    """Masque a 255 partout sauf sur le disque optique (et sa marge)."""
    m = np.full(forme, 255, dtype=np.uint8)
    cv2.circle(
        m,
        (int(round(reperes.disque_x)), int(round(reperes.disque_y))),
        max(1, int(round(reperes.disque_rayon_px * facteur))),
        0,
        -1,
    )
    return m


def transformer_reperes(reperes: Reperes, matrice: np.ndarray) -> Reperes:
    """Propage les reperes a travers une transformation affine."""
    points = np.array(
        [[reperes.disque_x, reperes.disque_y], [reperes.macula_x, reperes.macula_y]],
        dtype=np.float64,
    )
    projete = (matrice[:, :2] @ points.T).T + matrice[:, 2]
    echelle = float(np.hypot(matrice[0, 0], matrice[0, 1]))
    return Reperes(
        oeil=reperes.oeil,
        disque_x=float(projete[0, 0]),
        disque_y=float(projete[0, 1]),
        disque_rayon_px=reperes.disque_rayon_px * echelle,
        macula_x=float(projete[1, 0]),
        macula_y=float(projete[1, 1]),
        manuel=reperes.manuel,
    )
