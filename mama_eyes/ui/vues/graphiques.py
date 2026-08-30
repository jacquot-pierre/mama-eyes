"""Courbes d'evolution, avec reperes d'evenements et plancher de bruit.

Le plancher de bruit issu du controle negatif est trace en bande grisee : une
variation qui reste dans cette bande n'est pas distinguable du bruit de la
chaine, et ne doit pas etre interpretee.
"""

from __future__ import annotations

from datetime import date

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

COULEUR_PRINCIPALE = "#b26a12"
COULEUR_LISSE = "#2c6a60"
COULEUR_EVENEMENT = "#9e3a1e"


class AxeDates(pg.AxisItem):
    """Affiche les abscisses en dates lisibles plutot qu'en secondes."""

    def tickStrings(self, valeurs, echelle, espacement):
        sortie = []
        for v in valeurs:
            try:
                sortie.append(date.fromordinal(int(v)).strftime("%Y-%m"))
            except (ValueError, OverflowError):
                sortie.append("")
        return sortie


class VueGraphiques(QWidget):
    """Courbes d'evolution d'un oeil."""

    CHAMPS = {
        "Nombre de druses": "nombre",
        "Surface totale (px)": "surface_px",
        "Apparitions confirmees": "apparues",
        "Disparitions confirmees": "disparues",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        self.selecteur = QComboBox()
        self.selecteur.addItems(list(self.CHAMPS))
        self.selecteur.currentTextChanged.connect(self._redessiner)

        self.etiquette = QLabel("Aucune analyse chargee.")
        self.etiquette.setWordWrap(True)

        barre = QHBoxLayout()
        barre.addWidget(QLabel("Grandeur :"))
        barre.addWidget(self.selecteur)
        barre.addStretch(1)

        self.graphe = pg.PlotWidget(axisItems={"bottom": AxeDates(orientation="bottom")})
        self.graphe.setBackground(None)
        self.graphe.showGrid(x=True, y=True, alpha=0.22)
        self.graphe.addLegend(offset=(-10, 10))

        disposition = QVBoxLayout(self)
        disposition.addLayout(barre)
        disposition.addWidget(self.graphe, 1)
        disposition.addWidget(self.etiquette)

        self.serie = None
        self.evenements: list = []
        self.plancher: float | None = None

    def charger(self, serie, evenements=None, plancher: float | None = None) -> None:
        self.serie = serie
        self.evenements = evenements or []
        self.plancher = plancher
        self._redessiner()

    def _redessiner(self, *_) -> None:
        self.graphe.clear()
        if self.serie is None or not self.serie.points:
            self.etiquette.setText("Aucune analyse chargee.")
            return

        champ = self.CHAMPS[self.selecteur.currentText()]
        x = [p.date.toordinal() for p in self.serie.points]
        y = [getattr(p, champ) for p in self.serie.points]

        # Bande de bruit : en deca, rien n'est interpretable.
        if self.plancher and champ in ("apparues", "disparues"):
            bande = pg.LinearRegionItem(
                values=(0, self.plancher),
                orientation="horizontal",
                brush=pg.mkBrush(150, 150, 150, 46),
                movable=False,
            )
            bande.setZValue(-10)
            self.graphe.addItem(bande)

        self.graphe.plot(
            x,
            y,
            pen=pg.mkPen(COULEUR_PRINCIPALE, width=1.6),
            symbol="o",
            symbolSize=5,
            symbolBrush=COULEUR_PRINCIPALE,
            name="mesure",
        )

        if len(y) >= 5:
            lisse = self.serie.lisser(champ, fenetre=5)
            self.graphe.plot(
                x,
                lisse,
                pen=pg.mkPen(COULEUR_LISSE, width=2.4),
                name="tendance (moy. glissante)",
            )

        for e in self.evenements:
            ligne = pg.InfiniteLine(
                pos=e.date.toordinal(),
                angle=90,
                pen=pg.mkPen(COULEUR_EVENEMENT, width=2, style=Qt.DashLine),
                label=e.libelle,
                labelOpts={"position": 0.92, "color": COULEUR_EVENEMENT},
            )
            self.graphe.addItem(ligne)

        self.graphe.setLabel("left", self.selecteur.currentText())
        self._mettre_a_jour_etiquette(champ)

    def _mettre_a_jour_etiquette(self, champ: str) -> None:
        parties = [str(self.serie)]
        tendance = self.serie.tendance(champ)
        if tendance:
            parties.append(f"tendance : {tendance[0]:+.1f} / an")
        if self.plancher:
            parties.append(
                f"plancher de bruit (controle negatif) : {self.plancher:.0f} objets"
            )
        parties.extend(self.serie.messages)
        self.etiquette.setText("   ·   ".join(parties))
