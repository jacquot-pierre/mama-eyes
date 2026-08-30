"""Objets metier, independants du stockage et de l'interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class Cliche:
    """Un fichier image unique, rattache a une seance."""

    id: int | None
    seance_id: int | None
    fichier: str  # relatif a data/brut/
    sha256: str
    largeur: int
    hauteur: int
    nettete: float | None = None
    exclu: bool = False
    importe_le: str | None = None

    def chemin(self, racine_brut: Path) -> Path:
        return racine_brut / self.fichier


@dataclass
class Seance:
    """Une date de prise de vue pour un oeil ; porte 1..n cliches."""

    id: int | None
    oeil: str
    date: date
    note: str | None = None
    cliches: list[Cliche] = field(default_factory=list)

    @property
    def cliches_actifs(self) -> list[Cliche]:
        return [c for c in self.cliches if not c.exclu]

    def __str__(self) -> str:
        return f"{self.oeil} {self.date.isoformat()} ({len(self.cliches_actifs)} cliche(s))"


@dataclass
class Evenement:
    """Repere therapeutique porte sur les courbes d'evolution."""

    id: int | None
    date: date
    libelle: str
    note: str | None = None


@dataclass
class QualiteRecalage:
    """Diagnostic du recalage d'un cliche vers le gabarit."""

    cliche_id: int
    matrice: list[list[float]]  # 2x3
    inliers: int
    residu_rms: float
    valide: bool

    def __str__(self) -> str:
        etat = "ok" if self.valide else "ECHEC"
        return f"[{etat}] {self.inliers} inliers, rms={self.residu_rms:.2f} px"


@dataclass
class Reperes:
    """Disque optique et macula, en coordonnees du gabarit.

    Saisis une seule fois par oeil (cf. §5.5 : le clic manuel est le mode
    principal, les detecteurs automatiques echouent precisement en cas de
    DMLA), puis propages a toute la serie par les transformations.
    """

    oeil: str
    disque_x: float
    disque_y: float
    disque_rayon_px: float
    macula_x: float
    macula_y: float
    manuel: bool = True

    @property
    def px_par_mm(self) -> float:
        """Echelle deduite du diametre papillaire (cf. config.DIAMETRE_PAPILLAIRE_MM)."""
        from .config import DIAMETRE_PAPILLAIRE_MM

        return (2.0 * self.disque_rayon_px) / DIAMETRE_PAPILLAIRE_MM

    def um_en_px(self, micrometres: float) -> float:
        return micrometres * 1e-3 * self.px_par_mm

    def px_en_um(self, pixels: float) -> float:
        return pixels / self.px_par_mm * 1e3

    def surface_px_en_mm2(self, surface_px: float) -> float:
        return surface_px / (self.px_par_mm**2)


@dataclass
class Druse:
    """Une lesion detectee sur une seance donnee."""

    seance_id: int
    piste_id: int | None
    cx: float
    cy: float
    surface_px: int
    excentricite: float
    solidite: float
    intensite_moy: float

    def diametre_um(self, reperes: Reperes | None) -> float | None:
        if reperes is None:
            return None
        import math

        diam_px = 2.0 * math.sqrt(self.surface_px / math.pi)
        return reperes.px_en_um(diam_px)


@dataclass
class MesureSeance:
    """Agregat par seance, pour une execution d'analyse."""

    seance_id: int
    date: date
    nombre: int
    surface_px: int
    surface_mm2: float | None
    apparues: int = 0
    disparues: int = 0
    fusions: int = 0
    scissions: int = 0
