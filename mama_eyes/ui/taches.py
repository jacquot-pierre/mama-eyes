"""Execution des traitements longs hors du thread d'interface.

Un recalage de 105 seances prend plusieurs minutes : le lancer dans le thread
Qt figerait la fenetre. Chaque traitement passe donc par un `QRunnable` qui
communique par signaux.
"""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class SignauxTache(QObject):
    progression = Signal(str, int, int, str)  # etiquette, i, n, detail
    termine = Signal(object)
    echec = Signal(str)
    message = Signal(str)


class Tache(QRunnable):
    """Enveloppe une fonction bloquante et l'execute dans le pool."""

    def __init__(self, fonction: Callable, *args, **kwargs):
        super().__init__()
        self.fonction = fonction
        self.args = args
        self.kwargs = kwargs
        self.signaux = SignauxTache()
        self._annule = False

    def annuler(self) -> None:
        self._annule = True

    @property
    def annule(self) -> bool:
        return self._annule

    def _rappel(self, etiquette: str, i: int, n: int, detail: str = "") -> None:
        if self._annule:
            raise InterruptionTraitement()
        self.signaux.progression.emit(etiquette, i, n, detail)

    @Slot()
    def run(self) -> None:
        try:
            resultat = self.fonction(*self.args, rappel=self._rappel, **self.kwargs)
        except InterruptionTraitement:
            self.signaux.message.emit("Traitement interrompu.")
            self.signaux.termine.emit(None)
        except Exception:
            self.signaux.echec.emit(traceback.format_exc(limit=6))
        else:
            self.signaux.termine.emit(resultat)


class InterruptionTraitement(Exception):
    """Levee pour interrompre proprement un traitement en cours."""


class Ordonnanceur:
    """Petit gestionnaire de file, pour n'avoir qu'un traitement a la fois."""

    def __init__(self):
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(1)
        self.courante: Tache | None = None

    def lancer(
        self,
        fonction: Callable,
        *args,
        sur_progression=None,
        sur_fin=None,
        sur_echec=None,
        sur_message=None,
        **kwargs,
    ) -> Tache:
        tache = Tache(fonction, *args, **kwargs)
        if sur_progression:
            tache.signaux.progression.connect(sur_progression)
        if sur_fin:
            tache.signaux.termine.connect(sur_fin)
        if sur_echec:
            tache.signaux.echec.connect(sur_echec)
        if sur_message:
            tache.signaux.message.connect(sur_message)
        tache.signaux.termine.connect(lambda _: setattr(self, "courante", None))
        self.courante = tache
        self.pool.start(tache)
        return tache

    @property
    def occupe(self) -> bool:
        return self.courante is not None

    def annuler(self) -> None:
        if self.courante:
            self.courante.annuler()
