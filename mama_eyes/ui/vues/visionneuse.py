"""Visionneuse : zoom, panoramique, superpositions et comparateur clignotant.

Le comparateur clignotant (bascule rapide entre deux dates recalees) est le
seul moyen fiable de juger un recalage a l'oeil : des vaisseaux qui bougent
sautent immediatement aux yeux.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


def bgr_en_pixmap(image: np.ndarray) -> QPixmap:
    """Convertit un tableau BGR OpenCV en QPixmap."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    hauteur, largeur, _ = rgb.shape
    qimage = QImage(rgb.data, largeur, hauteur, 3 * largeur, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class Visionneuse(QGraphicsView):
    """Vue image avec zoom molette et panoramique au glisser."""

    clic_image = Signal(float, float)  # coordonnees dans l'image

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.item = QGraphicsPixmapItem()
        self.scene_.addItem(self.item)

        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.black)

        self._ajuste = False
        self._mode_clic = False

        # Comparateur clignotant
        self._images_bascule: list[np.ndarray] = []
        self._index_bascule = 0
        self._minuteur = QTimer(self)
        self._minuteur.timeout.connect(self._basculer)

    # -- affichage -----------------------------------------------------------

    def afficher(self, image: np.ndarray | None, ajuster: bool = False) -> None:
        self.arreter_bascule()
        if image is None:
            self.item.setPixmap(QPixmap())
            return
        self.item.setPixmap(bgr_en_pixmap(image))
        self.setSceneRect(self.item.boundingRect())
        if ajuster or not self._ajuste:
            self.ajuster()

    def ajuster(self) -> None:
        if self.item.pixmap().isNull():
            return
        self.fitInView(self.item, Qt.KeepAspectRatio)
        self._ajuste = True

    def zoom_reel(self) -> None:
        self.resetTransform()

    # -- comparateur clignotant ---------------------------------------------

    def demarrer_bascule(self, images: list[np.ndarray], periode_ms: int = 550) -> None:
        """Alterne entre plusieurs images au meme cadrage."""
        images = [i for i in images if i is not None]
        if len(images) < 2:
            return
        self._images_bascule = images
        self._index_bascule = 0
        self.item.setPixmap(bgr_en_pixmap(images[0]))
        self.setSceneRect(self.item.boundingRect())
        self._minuteur.start(periode_ms)

    def arreter_bascule(self) -> None:
        self._minuteur.stop()
        self._images_bascule = []

    @property
    def bascule_active(self) -> bool:
        return self._minuteur.isActive()

    def _basculer(self) -> None:
        if not self._images_bascule:
            self._minuteur.stop()
            return
        self._index_bascule = (self._index_bascule + 1) % len(self._images_bascule)
        self.item.setPixmap(bgr_en_pixmap(self._images_bascule[self._index_bascule]))

    # -- saisie de reperes ---------------------------------------------------

    def activer_clic(self, actif: bool) -> None:
        self._mode_clic = actif
        self.setDragMode(
            QGraphicsView.NoDrag if actif else QGraphicsView.ScrollHandDrag
        )
        self.setCursor(Qt.CrossCursor if actif else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self._mode_clic and event.button() == Qt.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            self.clic_image.emit(point.x(), point.y())
            return
        super().mousePressEvent(event)

    # -- interactions --------------------------------------------------------

    def wheelEvent(self, event):
        facteur = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(facteur, facteur)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.ajuster()
        elif event.key() == Qt.Key_1:
            self.zoom_reel()
        else:
            super().keyPressEvent(event)
