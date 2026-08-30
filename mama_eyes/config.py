"""Parametres de la chaine de traitement et emplacements sur disque.

Chaque groupe de parametres est une dataclass figee : son empreinte
(`empreinte()`) sert de cle de cache pour les artefacts derives. Changer un
reglage change l'empreinte, ce qui invalide automatiquement le cache et evite
de comparer un jour des images produites par deux versions de la chaine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# --- Oeil -------------------------------------------------------------------

OEILS = ("D", "G")

LIBELLE_OEIL = {"D": "Oeil droit", "G": "Oeil gauche"}


def valider_oeil(oeil: str) -> str:
    if oeil not in OEILS:
        raise ValueError(f"oeil doit valoir 'D' ou 'G', recu {oeil!r}")
    return oeil


# --- Emplacements -----------------------------------------------------------


@dataclass(frozen=True)
class Emplacements:
    """Arborescence de la bibliotheque.

    `brut` est immuable apres import ; tout le reste est un cache
    reconstructible et peut etre supprime sans perte.
    """

    racine: Path

    @property
    def base(self) -> Path:
        return self.racine / "mama_eyes.db"

    @property
    def brut(self) -> Path:
        return self.racine / "brut"

    @property
    def derive(self) -> Path:
        return self.racine / "derive"

    @property
    def reference(self) -> Path:
        return self.racine / "reference"

    @property
    def exports(self) -> Path:
        return self.racine / "exports"

    def brut_oeil(self, oeil: str) -> Path:
        return self.brut / valider_oeil(oeil)

    def derive_oeil(self, oeil: str, etape: str) -> Path:
        return self.derive / valider_oeil(oeil) / etape

    def reference_oeil(self, oeil: str) -> Path:
        return self.reference / valider_oeil(oeil)

    def creer(self) -> None:
        for oeil in OEILS:
            self.brut_oeil(oeil).mkdir(parents=True, exist_ok=True)
            self.reference_oeil(oeil).mkdir(parents=True, exist_ok=True)
        self.exports.mkdir(parents=True, exist_ok=True)
        self.derive.mkdir(parents=True, exist_ok=True)


def emplacements_par_defaut(racine: Path | str | None = None) -> Emplacements:
    if racine is None:
        racine = Path.cwd() / "data"
    return Emplacements(Path(racine).expanduser().resolve())


# --- Parametres de traitement ----------------------------------------------


class _Empreintable:
    """Fournit une empreinte stable et lisible d'un jeu de parametres."""

    def en_dict(self) -> dict:
        return asdict(self)  # type: ignore[arg-type]

    def empreinte(self) -> str:
        charge = json.dumps(self.en_dict(), sort_keys=True, default=str)
        return hashlib.sha256(charge.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ParamsChamp(_Empreintable):
    """Detection du champ retinien (FOV)."""

    canal: str = "rouge"  # canal le plus lumineux : separe le mieux du pourtour noir
    flou_px: int = 5
    facteur_erosion: float = 0.95  # cf. §5.1 : le degrade de bord fausse les stats
    marge_recadrage_px: int = 8


@dataclass(frozen=True)
class ParamsIllumination(_Empreintable):
    """Correction du vignettage puis harmonisation colorimetrique.

    L'etape la plus determinante de la chaine pour la validite des
    differences temporelles (cf. §5.2 du document d'architecture).
    """

    # Vignettage : le sigma est exprime en fraction du rayon du champ.
    sigma_fond_fraction: float = 0.125
    mode: str = "division"  # 'division' | 'soustraction'
    # Harmonisation LAB (Reinhard robuste) vers les statistiques de la serie.
    harmoniser_lab: bool = True
    exclure_disque: bool = True
    percentile_ecretage: float = 0.5


@dataclass(frozen=True)
class ParamsDebruitage(_Empreintable):
    median_analyse_px: int = 3  # 3x3 : preserve les petites druses
    median_affichage_px: int = 5


@dataclass(frozen=True)
class ParamsVaisseaux(_Empreintable):
    """Filtre de Frangi multi-echelle sur le canal vert inverse."""

    sigma_min: float = 2.0
    sigma_max: float = 12.0
    nb_sigmas: int = 6
    percentile_seuil: float = 88.0
    taille_min_objet_px: int = 250
    dilatation_masque_px: int = 9  # masque d'exclusion pour la detection de druses


@dataclass(frozen=True)
class ParamsRecalage(_Empreintable):
    """Recalage a deux etages : points caracteristiques puis intensite."""

    detecteur: str = "SIFT"
    nb_features: int = 6000
    ratio_lowe: float = 0.75
    seuil_ransac_px: float = 3.0
    modele: str = "similarite"  # 'similarite' (4 DDL) | 'affine' (6 DDL)
    inliers_min: int = 25
    residu_max_px: float = 6.0
    # Etage 2 : raffinement ECC sur les cartes vasculaires.
    ecc_actif: bool = True
    ecc_iterations: int = 60
    ecc_epsilon: float = 1e-5
    # Gabarit de groupe (cf. §5.6) : evite le biais d'une image de reference.
    iterations_gabarit: int = 2
    echelle_travail: float = 0.5  # sous-echantillonnage pour l'appariement


@dataclass(frozen=True)
class ParamsDruses(_Empreintable):
    """Segmentation absolue des druses."""

    poids_vert: float = 0.5
    poids_jaune: float = 0.5  # canal b* de LAB, exploitable grace au 4:4:4
    bloc_seuil_px: int = 151
    offset_seuil: float = -6.0
    surface_min_px: int = 12
    surface_max_px: int = 8000
    excentricite_max: float = 0.92
    solidite_min: float = 0.55
    exclure_vaisseaux: bool = True
    exclure_disque: bool = True
    rayon_exclusion_disque: float = 1.4  # en diametres papillaires


@dataclass(frozen=True)
class ParamsDifferentiel(_Empreintable):
    """Comparaison entre deux dates, ou contre une reference saine."""

    normaliser_gain: bool = True
    flou_pre_diff_px: float = 1.5  # amortit les erreurs de recalage sub-pixel
    # Seuil et surface minimale calibres sur le controle negatif (commande
    # `controle`) : sur les paires de 3 a 10 jours, ou aucune evolution
    # biologique n'est possible, ce couple ramene le bruit a ~41 px tandis que
    # les comparaisons a 3-7 ans conservent ~1180 px, soit une separation de
    # pres de 30x. A 36 le signal reel disparait a son tour.
    # A recalibrer avec `controle` apres tout changement de la chaine amont.
    seuil_diff: float = 26.0
    surface_min_px: int = 60
    dilater_exclusion_vaisseaux_px: int = 11


@dataclass(frozen=True)
class ParamsSuivi(_Empreintable):
    """Appariement temporel des druses et regle de confirmation."""

    distance_max_px: float = 28.0
    poids_surface: float = 0.35
    confirmations_requises: int = 2  # cf. cahier des charges
    # Une piste survit a une absence isolee : sans cette memoire, une druse
    # manquee sur un seul cliche produirait a la fois une fausse disparition
    # et une fausse apparition.
    memoire_dates: int = 2
    signaler_fusion_scission: bool = True


@dataclass(frozen=True)
class ParamsVideo(_Empreintable):
    fps: int = 6
    trames_par_date: int = 4
    duree_proportionnelle: bool = True  # les ecarts vont de 3 a 130 jours
    trames_min: int = 2
    trames_max: int = 20
    lissage_exposition: int = 3
    qualite: int = 6


@dataclass(frozen=True)
class Parametres(_Empreintable):
    """Jeu complet, serialisable en preset."""

    champ: ParamsChamp = field(default_factory=ParamsChamp)
    illumination: ParamsIllumination = field(default_factory=ParamsIllumination)
    debruitage: ParamsDebruitage = field(default_factory=ParamsDebruitage)
    vaisseaux: ParamsVaisseaux = field(default_factory=ParamsVaisseaux)
    recalage: ParamsRecalage = field(default_factory=ParamsRecalage)
    druses: ParamsDruses = field(default_factory=ParamsDruses)
    differentiel: ParamsDifferentiel = field(default_factory=ParamsDifferentiel)
    suivi: ParamsSuivi = field(default_factory=ParamsSuivi)
    video: ParamsVideo = field(default_factory=ParamsVideo)

    # -- serialisation ------------------------------------------------------

    def en_json(self) -> str:
        return json.dumps(self.en_dict(), sort_keys=True, indent=2)

    @classmethod
    def depuis_dict(cls, donnees: dict) -> "Parametres":
        classes = {
            "champ": ParamsChamp,
            "illumination": ParamsIllumination,
            "debruitage": ParamsDebruitage,
            "vaisseaux": ParamsVaisseaux,
            "recalage": ParamsRecalage,
            "druses": ParamsDruses,
            "differentiel": ParamsDifferentiel,
            "suivi": ParamsSuivi,
            "video": ParamsVideo,
        }
        kwargs = {}
        for nom, klass in classes.items():
            if nom in donnees and donnees[nom] is not None:
                kwargs[nom] = klass(**donnees[nom])
        return cls(**kwargs)

    @classmethod
    def depuis_json(cls, texte: str) -> "Parametres":
        return cls.depuis_dict(json.loads(texte))

    def avec(self, **groupes) -> "Parametres":
        """Copie avec un ou plusieurs groupes remplaces."""
        return replace(self, **groupes)


# --- Calibration ------------------------------------------------------------

#: Diametre moyen du disque optique chez l'adulte. Sert d'etalon de longueur.
#: Varie de +/-15 % selon les individus : la calibration est juste *en interne*
#: a une serie, les millimetres affiches restent indicatifs (cf. §5.10).
DIAMETRE_PAPILLAIRE_MM = 1.85

#: Seuils de taille des druses, en micrometres.
DRUSE_PETITE_UM = 63.0
DRUSE_INTERMEDIAIRE_UM = 125.0

#: Rayons de la grille maculaire concentrique, en millimetres (facon ETDRS).
GRILLE_MACULAIRE_MM = (0.5, 1.5, 3.0)
