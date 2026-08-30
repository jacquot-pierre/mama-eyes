"""Agregation des mesures et series temporelles.

Les surfaces sont converties en millimetres carres via l'echelle deduite du
diametre papillaire. Cette calibration est juste *en interne* a une serie -
le diametre du disque variant de +/-15 % d'un individu a l'autre, les
millimetres affiches restent indicatifs. Ce n'est pas genant : c'est
l'evolution relative qui porte l'information.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from ..domain import Reperes


@dataclass
class PointEvolution:
    """Un point de la courbe d'evolution."""

    date: date
    nombre: int
    surface_px: int
    surface_mm2: float | None
    apparues: int = 0
    disparues: int = 0
    fusions: int = 0
    scissions: int = 0
    nettete: float | None = None
    recalage_valide: bool = True


@dataclass
class SerieEvolution:
    """Serie temporelle complete pour un oeil."""

    oeil: str
    methode: str
    points: list[PointEvolution] = field(default_factory=list)
    plancher_bruit: float | None = None
    #: Faux positifs par date mesures sur les suites de dates rapprochees
    #: (cf. `analyse.controle.controle_sequence`). Les comptages d'evenements
    #: qui restent en dessous ne sont pas interpretables.
    plancher_evenements: float | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def evenements_credibles(self) -> bool | None:
        """Les evenements confirmes depassent-ils le plancher de faux positifs ?"""
        if self.plancher_evenements is None or not self.points:
            return None
        mesure = statistics.median(
            [p.apparues + p.disparues for p in self.points]
        )
        return mesure > self.plancher_evenements

    def dates(self) -> list[date]:
        return [p.date for p in self.points]

    def valeurs(self, champ: str) -> list:
        return [getattr(p, champ) for p in self.points]

    def tendance(self, champ: str = "nombre") -> tuple[float, float] | None:
        """Regression lineaire simple : (pente par an, ordonnee)."""
        if len(self.points) < 3:
            return None
        jours = np.array(
            [(p.date - self.points[0].date).days for p in self.points], dtype=float
        )
        valeurs = np.array(self.valeurs(champ), dtype=float)
        if np.allclose(jours, jours[0]):
            return None
        pente, ordonnee = np.polyfit(jours, valeurs, 1)
        return float(pente * 365.25), float(ordonnee)

    def lisser(self, champ: str = "nombre", fenetre: int = 5) -> list[float]:
        """Moyenne glissante centree, pour degager la tendance du bruit."""
        valeurs = np.array(self.valeurs(champ), dtype=float)
        if len(valeurs) < fenetre or fenetre < 2:
            return valeurs.tolist()
        noyau = np.ones(fenetre) / fenetre
        rembourre = np.pad(valeurs, fenetre // 2, mode="edge")
        return np.convolve(rembourre, noyau, mode="valid")[: len(valeurs)].tolist()

    def __str__(self) -> str:
        if not self.points:
            return f"{self.oeil} : aucune mesure"
        texte = (
            f"{len(self.points)} date(s), "
            f"{self.points[0].date} -> {self.points[-1].date}"
        )
        # La surface est la grandeur la plus stable quand les druses sont
        # confluentes : c'est elle qu'on met en avant.
        surface = self.tendance("surface_px")
        if surface and self.points[0].surface_px:
            moyenne = statistics.mean(self.valeurs("surface_px")) or 1
            texte += f", surface {surface[0] / moyenne * 100:+.1f} %/an"

        credibles = self.evenements_credibles
        evenements = (
            f"{sum(p.apparues for p in self.points)} apparition(s) / "
            f"{sum(p.disparues for p in self.points)} disparition(s)"
        )
        if credibles is False:
            texte += (
                f" | evenements NON INTERPRETABLES ({evenements}, "
                f"sous le plancher de {self.plancher_evenements:.0f} faux "
                f"positifs par date)"
            )
        else:
            texte += f" | {evenements} confirmee(s)"
        return texte

    def exporter_csv(self, chemin: Path) -> None:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "date",
                    "nombre",
                    "surface_px",
                    "surface_mm2",
                    "apparues",
                    "disparues",
                    "fusions",
                    "scissions",
                    "nettete",
                    "recalage_valide",
                ]
            )
            for p in self.points:
                w.writerow(
                    [
                        p.date,
                        p.nombre,
                        p.surface_px,
                        f"{p.surface_mm2:.4f}" if p.surface_mm2 is not None else "",
                        p.apparues,
                        p.disparues,
                        p.fusions,
                        p.scissions,
                        f"{p.nettete:.1f}" if p.nettete is not None else "",
                        int(p.recalage_valide),
                    ]
                )


def surface_en_mm2(surface_px: int, reperes: Reperes | None) -> float | None:
    if reperes is None:
        return None
    return reperes.surface_px_en_mm2(surface_px)


def correler_nettete(serie: SerieEvolution) -> float | None:
    """Correlation entre nombre de druses et nettete de l'image.

    Une correlation forte est un signal d'alarme : elle indique que la mesure
    suit la qualite d'image plutot que la maladie. A verifier avant toute
    interpretation d'une tendance.
    """
    couples = [
        (p.nombre, p.nettete)
        for p in serie.points
        if p.nettete is not None and p.nettete > 0
    ]
    if len(couples) < 5:
        return None
    a = np.array([c[0] for c in couples], dtype=float)
    b = np.array([c[1] for c in couples], dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])
