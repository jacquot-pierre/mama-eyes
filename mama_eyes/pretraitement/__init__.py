"""Couche image : masque du champ, photometrie, vaisseaux, recalage, fusion."""

from .champ import Champ, detecter_champ, masque_disque
from .debruitage import debruiter
from .fusion import fusionner_seance
from .illumination import (
    StatsLab,
    corriger_vignettage,
    harmoniser_lab,
    stats_lab,
    stats_medianes,
)

__all__ = [
    "Champ",
    "detecter_champ",
    "masque_disque",
    "debruiter",
    "fusionner_seance",
    "StatsLab",
    "corriger_vignettage",
    "harmoniser_lab",
    "stats_lab",
    "stats_medianes",
]
