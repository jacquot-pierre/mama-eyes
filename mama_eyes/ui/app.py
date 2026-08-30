"""Fenetre principale.

Organisation en quatre onglets qui suivent l'ordre reel du travail :
bibliotheque -> traitements -> analyse -> videos. La visionneuse est partagee,
avec son comparateur clignotant pour juger les recalages.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import LIBELLE_OEIL, OEILS, Parametres, emplacements_par_defaut
from ..domain import Reperes
from ..pipeline import Pipeline
from ..stockage.base import Base
from ..stockage.bibliotheque import Bibliotheque
from .taches import Ordonnanceur
from .vues.graphiques import VueGraphiques
from .vues.visionneuse import Visionneuse

AVERTISSEMENT = (
    "Outil de suivi personnel, sans finalite diagnostique. Les mesures sont "
    "des indicateurs de tendance sur une serie homogene."
)


class DialogueImport(QDialog):
    """Saisie de la date et de l'oeil pour un nouveau cliche.

    Les fichiers ne portant pas d'EXIF, la date ne peut pas etre devinee :
    elle est pre-remplie depuis le nom quand c'est possible, sinon demandee.
    """

    def __init__(self, fichiers: list[Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajouter des cliches")
        self.fichiers = fichiers

        from ..stockage.nommage import analyser_nom

        analyse = analyser_nom(fichiers[0].name) if fichiers else None

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(analyse.date if analyse else date.today())

        self.oeil = QComboBox()
        for o in OEILS:
            self.oeil.addItem(LIBELLE_OEIL[o], o)
        if analyse:
            self.oeil.setCurrentIndex(OEILS.index(analyse.oeil))

        self.meme_seance = QCheckBox(
            "Tous ces cliches appartiennent a la meme seance"
        )
        self.meme_seance.setChecked(len(fichiers) > 1)

        formulaire = QFormLayout()
        formulaire.addRow("Date de prise de vue", self.date_edit)
        formulaire.addRow("Oeil", self.oeil)
        if len(fichiers) > 1:
            formulaire.addRow(self.meme_seance)
        formulaire.addRow(QLabel(f"{len(fichiers)} fichier(s) selectionne(s)"))

        boutons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        boutons.accepted.connect(self.accept)
        boutons.rejected.connect(self.reject)

        disposition = QVBoxLayout(self)
        disposition.addLayout(formulaire)
        disposition.addWidget(boutons)

    def valeurs(self) -> tuple[date, str, bool]:
        return (
            self.date_edit.date().toPython(),
            self.oeil.currentData(),
            self.meme_seance.isChecked(),
        )


class FenetrePrincipale(QMainWindow):
    def __init__(self, racine_data: str | None = None):
        super().__init__()
        self.setWindowTitle("Mama Eyes - suivi du fond d'oeil")
        self.resize(1420, 900)

        emp = emplacements_par_defaut(racine_data)
        self.base = Base(emp.base)
        self.biblio = Bibliotheque(emp, self.base)
        self.pipeline = Pipeline(self.biblio, self.base, Parametres())
        self.ordonnanceur = Ordonnanceur()

        self.oeil_courant = "D"
        self.serie_courante = None
        self.plancher_bruit: float | None = None
        self._reperes_en_saisie: str | None = None
        self._reperes_temp: dict = {}

        self._construire()
        self._rafraichir_bibliotheque()

    # -- construction --------------------------------------------------------

    def _construire(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Barre superieure : selection de l'oeil.
        self.selecteur_oeil = QComboBox()
        for o in OEILS:
            self.selecteur_oeil.addItem(LIBELLE_OEIL[o], o)
        self.selecteur_oeil.currentIndexChanged.connect(self._changer_oeil)

        haut = QHBoxLayout()
        haut.addWidget(QLabel("Oeil :"))
        haut.addWidget(self.selecteur_oeil)
        haut.addSpacing(20)
        self.resume_label = QLabel()
        haut.addWidget(self.resume_label)
        haut.addStretch(1)

        self.onglets = QTabWidget()
        self.onglets.addTab(self._onglet_bibliotheque(), "Bibliotheque")
        self.onglets.addTab(self._onglet_traitements(), "Traitements")
        self.onglets.addTab(self._onglet_analyse(), "Analyse")
        self.onglets.addTab(self._onglet_videos(), "Videos")

        self.progression = QProgressBar()
        self.progression.setVisible(False)
        self.etat = QLabel(AVERTISSEMENT)
        self.etat.setStyleSheet("color: #7a6e66;")

        bas = QHBoxLayout()
        bas.addWidget(self.etat, 1)
        bas.addWidget(self.progression)

        disposition = QVBoxLayout(central)
        disposition.addLayout(haut)
        disposition.addWidget(self.onglets, 1)
        disposition.addLayout(bas)

    def _onglet_bibliotheque(self) -> QWidget:
        widget = QWidget()

        self.liste = QListWidget()
        self.liste.currentItemChanged.connect(self._selection_changee)

        bouton_ajouter = QPushButton("Ajouter des cliches...")
        bouton_ajouter.clicked.connect(self._ajouter_cliches)
        bouton_dossier = QPushButton("Importer un dossier...")
        bouton_dossier.clicked.connect(self._importer_dossier)
        self.bouton_exclure = QPushButton("Exclure / reintegrer")
        self.bouton_exclure.clicked.connect(self._basculer_exclusion)

        boutons = QVBoxLayout()
        boutons.addWidget(bouton_ajouter)
        boutons.addWidget(bouton_dossier)
        boutons.addWidget(self.bouton_exclure)
        boutons.addStretch(1)

        gauche = QVBoxLayout()
        gauche.addWidget(QLabel("Seances"))
        gauche.addWidget(self.liste, 1)
        gauche.addLayout(boutons)
        cadre_gauche = QWidget()
        cadre_gauche.setLayout(gauche)
        cadre_gauche.setMaximumWidth(300)

        self.vue_biblio = Visionneuse()
        self.info_cliche = QLabel()
        self.info_cliche.setWordWrap(True)

        droite = QVBoxLayout()
        droite.addWidget(self.vue_biblio, 1)
        droite.addWidget(self.info_cliche)
        cadre_droite = QWidget()
        cadre_droite.setLayout(droite)

        separateur = QSplitter()
        separateur.addWidget(cadre_gauche)
        separateur.addWidget(cadre_droite)
        separateur.setStretchFactor(1, 1)

        disposition = QHBoxLayout(widget)
        disposition.addWidget(separateur)
        return widget

    def _onglet_traitements(self) -> QWidget:
        widget = QWidget()

        groupe_chaine = QGroupBox("Chaine de traitement")
        self.bouton_normaliser = QPushButton("1. Normaliser (photometrie, fusion)")
        self.bouton_normaliser.clicked.connect(lambda: self._lancer("normaliser"))
        self.bouton_gabarit = QPushButton("2. Construire le gabarit de groupe")
        self.bouton_gabarit.clicked.connect(lambda: self._lancer("gabarit"))
        self.bouton_recaler = QPushButton("3. Recaler sur le gabarit")
        self.bouton_recaler.clicked.connect(lambda: self._lancer("recaler"))
        self.bouton_annuler = QPushButton("Interrompre")
        self.bouton_annuler.clicked.connect(self.ordonnanceur.annuler)
        self.bouton_annuler.setEnabled(False)

        vc = QVBoxLayout(groupe_chaine)
        for b in (
            self.bouton_normaliser,
            self.bouton_gabarit,
            self.bouton_recaler,
            self.bouton_annuler,
        ):
            vc.addWidget(b)

        groupe_controle = QGroupBox("Controle du recalage")
        self.bouton_clignotant = QPushButton("Comparateur clignotant (2 dates)")
        self.bouton_clignotant.clicked.connect(self._comparateur_clignotant)
        self.bouton_damier = QPushButton("Damier de superposition")
        self.bouton_damier.clicked.connect(self._damier)
        self.liste_echecs = QListWidget()
        vk = QVBoxLayout(groupe_controle)
        vk.addWidget(self.bouton_clignotant)
        vk.addWidget(self.bouton_damier)
        vk.addWidget(QLabel("Recalages a revoir :"))
        vk.addWidget(self.liste_echecs)

        groupe_reperes = QGroupBox("Reperes anatomiques")
        info = QLabel(
            "Le clic manuel est le mode principal : les detecteurs automatiques "
            "echouent en cas de DMLA (les druses confluentes rivalisent de "
            "clarte avec le disque, l'atrophie efface la fovea sombre). "
            "Deux clics par oeil suffisent pour toute la serie."
        )
        info.setWordWrap(True)
        self.bouton_disque = QPushButton("Pointer le disque optique")
        self.bouton_disque.clicked.connect(lambda: self._saisir_repere("disque"))
        self.bouton_macula = QPushButton("Pointer la macula")
        self.bouton_macula.clicked.connect(lambda: self._saisir_repere("macula"))
        self.label_reperes = QLabel()
        vr = QVBoxLayout(groupe_reperes)
        vr.addWidget(info)
        vr.addWidget(self.bouton_disque)
        vr.addWidget(self.bouton_macula)
        vr.addWidget(self.label_reperes)

        gauche = QVBoxLayout()
        gauche.addWidget(groupe_chaine)
        gauche.addWidget(groupe_controle)
        gauche.addWidget(groupe_reperes)
        gauche.addStretch(1)
        cadre_gauche = QWidget()
        cadre_gauche.setLayout(gauche)
        cadre_gauche.setMaximumWidth(390)

        self.vue_traitement = Visionneuse()
        self.vue_traitement.clic_image.connect(self._clic_repere)
        self.journal = QPlainTextEdit()
        self.journal.setReadOnly(True)
        self.journal.setMaximumHeight(170)

        droite = QVBoxLayout()
        droite.addWidget(self.vue_traitement, 1)
        droite.addWidget(self.journal)
        cadre_droite = QWidget()
        cadre_droite.setLayout(droite)

        separateur = QSplitter()
        separateur.addWidget(cadre_gauche)
        separateur.addWidget(cadre_droite)
        separateur.setStretchFactor(1, 1)

        disposition = QHBoxLayout(widget)
        disposition.addWidget(separateur)
        return widget

    def _onglet_analyse(self) -> QWidget:
        widget = QWidget()

        groupe = QGroupBox("Analyses")
        self.bouton_controle = QPushButton("Controle negatif (paires rapprochees)")
        self.bouton_controle.setToolTip(
            "Compare les cliches separes de moins de dix jours. Aucune evolution "
            "biologique n'y est possible : tout ce qui est detecte est du bruit, "
            "ce qui calibre les seuils."
        )
        self.bouton_controle.clicked.connect(lambda: self._lancer("controle"))
        self.bouton_evolution = QPushButton("Detection et suivi temporel")
        self.bouton_evolution.clicked.connect(lambda: self._lancer("evolution"))
        self.bouton_reference = QPushButton("Choisir l'image de reference saine...")
        self.bouton_reference.clicked.connect(self._choisir_reference)
        self.bouton_csv = QPushButton("Exporter en CSV")
        self.bouton_csv.clicked.connect(self._exporter_csv)

        vg = QVBoxLayout(groupe)
        for b in (
            self.bouton_controle,
            self.bouton_evolution,
            self.bouton_reference,
            self.bouton_csv,
        ):
            vg.addWidget(b)

        groupe_evt = QGroupBox("Reperes therapeutiques")
        self.champ_evt_date = QDateEdit()
        self.champ_evt_date.setCalendarPopup(True)
        self.champ_evt_date.setDisplayFormat("yyyy-MM-dd")
        self.champ_evt_date.setDate(date.today())
        self.champ_evt_libelle = QLineEdit()
        self.champ_evt_libelle.setPlaceholderText("ex. debut resveratrol")
        bouton_evt = QPushButton("Ajouter le repere")
        bouton_evt.clicked.connect(self._ajouter_evenement)
        self.liste_evt = QListWidget()
        fe = QFormLayout()
        fe.addRow("Date", self.champ_evt_date)
        fe.addRow("Libelle", self.champ_evt_libelle)
        ve = QVBoxLayout(groupe_evt)
        ve.addLayout(fe)
        ve.addWidget(bouton_evt)
        ve.addWidget(self.liste_evt)

        self.rapport_controle = QPlainTextEdit()
        self.rapport_controle.setReadOnly(True)

        gauche = QVBoxLayout()
        gauche.addWidget(groupe)
        gauche.addWidget(groupe_evt)
        gauche.addWidget(QLabel("Controle negatif"))
        gauche.addWidget(self.rapport_controle, 1)
        cadre_gauche = QWidget()
        cadre_gauche.setLayout(gauche)
        cadre_gauche.setMaximumWidth(430)

        self.vue_graphiques = VueGraphiques()

        separateur = QSplitter()
        separateur.addWidget(cadre_gauche)
        separateur.addWidget(self.vue_graphiques)
        separateur.setStretchFactor(1, 1)

        disposition = QHBoxLayout(widget)
        disposition.addWidget(separateur)
        return widget

    def _onglet_videos(self) -> QWidget:
        widget = QWidget()
        self.bouton_timelapse = QPushButton("Generer le timelapse")
        self.bouton_timelapse.clicked.connect(lambda: self._lancer("timelapse"))
        self.bouton_evidence = QPushButton("Generer la video de mise en evidence")
        self.bouton_evidence.clicked.connect(lambda: self._lancer("evidence"))
        self.bouton_ouvrir_exports = QPushButton("Ouvrir le dossier des exports")
        self.bouton_ouvrir_exports.clicked.connect(self._ouvrir_exports)
        self.liste_videos = QListWidget()

        info = QLabel(
            "La cadence est proportionnelle a l'ecart reel entre cliches : les "
            "intervalles vont de 3 a 130 jours, une cadence constante donnerait "
            "une fausse impression de regularite."
        )
        info.setWordWrap(True)

        disposition = QVBoxLayout(widget)
        disposition.addWidget(info)
        disposition.addWidget(self.bouton_timelapse)
        disposition.addWidget(self.bouton_evidence)
        disposition.addWidget(self.bouton_ouvrir_exports)
        disposition.addWidget(QLabel("Fichiers produits :"))
        disposition.addWidget(self.liste_videos, 1)
        return widget

    # -- etat ----------------------------------------------------------------

    def _changer_oeil(self) -> None:
        self.oeil_courant = self.selecteur_oeil.currentData()
        self._rafraichir_bibliotheque()

    def _rafraichir_bibliotheque(self) -> None:
        self.liste.clear()
        seances = self.base.seances(self.oeil_courant)
        invalides = {c for c, _ in self.base.recalages_invalides(self.oeil_courant)}
        for s in seances:
            marques = []
            if self.pipeline.chemin_recale(self.oeil_courant, s.date).exists():
                marques.append("R")
            elif self.pipeline.chemin_normalise(self.oeil_courant, s.date).exists():
                marques.append("N")
            if s.cliches_actifs and all(c.id in invalides for c in s.cliches_actifs):
                marques.append("!")
            suffixe = f"  [{''.join(marques)}]" if marques else ""
            item = QListWidgetItem(f"{s.date}  ({len(s.cliches_actifs)}){suffixe}")
            item.setData(Qt.UserRole, s.id)
            if "!" in marques:
                item.setForeground(Qt.red)
            self.liste.addItem(item)

        resume = self.base.resume()[self.oeil_courant]
        self.resume_label.setText(
            f"{resume['seances']} seance(s), {resume['cliches']} cliche(s)   "
            f"{resume['debut'] or '-'} -> {resume['fin'] or '-'}"
        )

        self.liste_echecs.clear()
        for _, fichier in self.base.recalages_invalides(self.oeil_courant):
            self.liste_echecs.addItem(fichier)

        self.liste_evt.clear()
        for e in self.base.evenements():
            self.liste_evt.addItem(f"{e.date}  {e.libelle}")

        self._rafraichir_reperes()
        self._rafraichir_videos()

    def _rafraichir_reperes(self) -> None:
        r = self.base.reperes(self.oeil_courant)
        if r is None:
            self.label_reperes.setText("Aucun repere enregistre.")
            return
        origine = "manuels" if r.manuel else "proposition automatique"
        self.label_reperes.setText(
            f"Disque ({r.disque_x:.0f}, {r.disque_y:.0f}) r={r.disque_rayon_px:.0f} px\n"
            f"Macula ({r.macula_x:.0f}, {r.macula_y:.0f})\n"
            f"Echelle : {r.px_par_mm:.0f} px/mm  ({origine})"
        )

    def _rafraichir_videos(self) -> None:
        self.liste_videos.clear()
        dossier = self.pipeline.emp.exports
        if dossier.is_dir():
            for f in sorted(dossier.glob("*.mp4")):
                taille = f.stat().st_size / 1e6
                self.liste_videos.addItem(f"{f.name}   ({taille:.1f} Mo)")

    def _seance_selectionnee(self):
        item = self.liste.currentItem()
        if item is None:
            return None
        seance_id = item.data(Qt.UserRole)
        for s in self.base.seances(self.oeil_courant):
            if s.id == seance_id:
                return s
        return None

    def _selection_changee(self, *_) -> None:
        seance = self._seance_selectionnee()
        if seance is None:
            return
        image = self.pipeline.lire_recale(self.oeil_courant, seance.date)
        etape = "recalee"
        if image is None:
            image = self.pipeline.lire_normalise(self.oeil_courant, seance.date)
            etape = "normalisee"
        if image is None and seance.cliches_actifs:
            image = self.biblio.lire(seance.cliches_actifs[0])
            etape = "brute"
        self.vue_biblio.afficher(image, ajuster=True)
        self.vue_traitement.afficher(image, ajuster=True)

        details = [f"{seance.date} · image {etape}"]
        for c in seance.cliches:
            q = self.base.recalage(c.id) if c.id else None
            ligne = f"{c.fichier} · {c.largeur}x{c.hauteur}"
            if c.nettete:
                ligne += f" · nettete {c.nettete:.0f}"
            if c.exclu:
                ligne += " · EXCLU"
            if q:
                ligne += f" · {q}"
            details.append(ligne)
        self.info_cliche.setText("\n".join(details))

    # -- actions bibliotheque ------------------------------------------------

    def _ajouter_cliches(self) -> None:
        fichiers, _ = QFileDialog.getOpenFileNames(
            self, "Choisir des cliches", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if not fichiers:
            return
        chemins = [Path(f) for f in fichiers]
        dialogue = DialogueImport(chemins, self)
        if dialogue.exec() != QDialog.Accepted:
            return
        jour, oeil, meme_seance = dialogue.valeurs()

        importes = doublons = 0
        for chemin in chemins:
            try:
                from ..stockage.nommage import analyser_nom

                jour_effectif = jour
                if not meme_seance:
                    analyse = analyser_nom(chemin.name, oeil_par_defaut=oeil)
                    if analyse:
                        jour_effectif = analyse.date
                resultat = self.biblio.importer_fichier(chemin, oeil=oeil, jour=jour_effectif)
                if resultat is None:
                    doublons += 1
                else:
                    importes += 1
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Import", f"{chemin.name} : {exc}")

        self._message(f"{importes} cliche(s) importe(s), {doublons} doublon(s) ignore(s).")
        self._rafraichir_bibliotheque()

    def _importer_dossier(self) -> None:
        dossier = QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        if not dossier:
            return
        oeil = self.oeil_courant
        self._lancer_generique(
            lambda rappel: self.biblio.importer_dossier(
                Path(dossier), oeil=oeil, rappel=lambda i, n, nom: rappel("import", i, n, nom)
            ),
            "import",
        )

    def _basculer_exclusion(self) -> None:
        seance = self._seance_selectionnee()
        if seance is None or not seance.cliches:
            return
        for c in seance.cliches:
            self.base.definir_exclusion(c.id, not c.exclu)
        self._rafraichir_bibliotheque()

    # -- reperes -------------------------------------------------------------

    def _saisir_repere(self, lequel: str) -> None:
        gab = self.pipeline.lire_gabarit(self.oeil_courant)
        if gab is None:
            QMessageBox.information(
                self,
                "Reperes",
                "Construisez d'abord le gabarit : les reperes y sont saisis "
                "une fois, puis propages a toute la serie.",
            )
            return
        self.vue_traitement.afficher(gab[0], ajuster=True)
        self._reperes_en_saisie = lequel
        self.vue_traitement.activer_clic(True)
        cible = "le centre du disque optique" if lequel == "disque" else "la fovea"
        self._message(f"Cliquez sur {cible}. Echap pour annuler.")

    def _clic_repere(self, x: float, y: float) -> None:
        if not self._reperes_en_saisie:
            return
        self._reperes_temp[self._reperes_en_saisie] = (x, y)
        self.vue_traitement.activer_clic(False)

        if self._reperes_en_saisie == "disque":
            self._message(
                "Disque enregistre. Pointez maintenant la macula "
                "(bouton « Pointer la macula »)."
            )
        self._reperes_en_saisie = None

        disque = self._reperes_temp.get("disque")
        macula = self._reperes_temp.get("macula")
        if disque and macula:
            gab = self.pipeline.lire_gabarit(self.oeil_courant)
            from ..pretraitement.champ import detecter_champ

            champ = detecter_champ(gab[0], self.pipeline.params.champ)
            rayon = max(12.0, champ.rayon / 6.0)
            self.base.enregistrer_reperes(
                Reperes(
                    oeil=self.oeil_courant,
                    disque_x=disque[0],
                    disque_y=disque[1],
                    disque_rayon_px=rayon,
                    macula_x=macula[0],
                    macula_y=macula[1],
                    manuel=True,
                )
            )
            self._reperes_temp = {}
            self._message("Reperes enregistres pour cet oeil.")
            self._rafraichir_reperes()

    # -- controles visuels ---------------------------------------------------

    def _deux_dates_recalees(self) -> list[np.ndarray]:
        seances = self.pipeline.seances_recalees(self.oeil_courant, valides_seulement=False)
        if len(seances) < 2:
            return []
        courante = self._seance_selectionnee()
        if courante is not None:
            index = next(
                (i for i, s in enumerate(seances) if s.id == courante.id), 0
            )
        else:
            index = 0
        suivant = min(index + 1, len(seances) - 1)
        if suivant == index:
            index = max(0, suivant - 1)
        return [
            self.pipeline.lire_recale(self.oeil_courant, seances[index].date),
            self.pipeline.lire_recale(self.oeil_courant, seances[suivant].date),
        ]

    def _comparateur_clignotant(self) -> None:
        if self.vue_traitement.bascule_active:
            self.vue_traitement.arreter_bascule()
            self._message("Comparateur arrete.")
            return
        images = [i for i in self._deux_dates_recalees() if i is not None]
        if len(images) < 2:
            QMessageBox.information(
                self, "Comparateur", "Il faut au moins deux seances recalees."
            )
            return
        self.vue_traitement.demarrer_bascule(images)
        self._message(
            "Comparateur actif : des vaisseaux immobiles signalent un bon recalage."
        )

    def _damier(self) -> None:
        from ..qualite import damier

        images = [i for i in self._deux_dates_recalees() if i is not None]
        if len(images) < 2:
            return
        self.vue_traitement.afficher(damier(images[0], images[1]), ajuster=True)
        self._message(
            "Damier : un vaisseau brise a la frontiere d'une case revele un defaut."
        )

    # -- analyses ------------------------------------------------------------

    def _choisir_reference(self) -> None:
        fichier, _ = QFileDialog.getOpenFileName(
            self,
            "Image de reference « saine » nettoyee",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if not fichier:
            return
        cible = self.biblio.definir_reference(self.oeil_courant, Path(fichier))
        self._message(f"Reference enregistree : {cible.name}")

    def _ajouter_evenement(self) -> None:
        libelle = self.champ_evt_libelle.text().strip()
        if not libelle:
            return
        self.base.ajouter_evenement(self.champ_evt_date.date().toPython(), libelle)
        self.champ_evt_libelle.clear()
        self._rafraichir_bibliotheque()
        if self.serie_courante is not None:
            self.vue_graphiques.charger(
                self.serie_courante, self.base.evenements(), self.plancher_bruit
            )

    def _exporter_csv(self) -> None:
        if self.serie_courante is None:
            QMessageBox.information(self, "Export", "Lancez d'abord une analyse.")
            return
        cible = self.pipeline.emp.exports / f"evolution_{self.oeil_courant}.csv"
        self.serie_courante.exporter_csv(cible)
        self._message(f"Export : {cible}")

    def _ouvrir_exports(self) -> None:
        import subprocess

        dossier = self.pipeline.emp.exports
        dossier.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(dossier)])
        except OSError:
            self._message(str(dossier))

    # -- execution des taches ------------------------------------------------

    def _lancer(self, quoi: str) -> None:
        if self.ordonnanceur.occupe:
            QMessageBox.information(
                self, "Traitement", "Un traitement est deja en cours."
            )
            return
        oeil = self.oeil_courant
        pipeline, base = self.pipeline, self.base

        if quoi == "normaliser":
            fonction = lambda rappel: pipeline.normaliser(oeil, rappel=rappel)
        elif quoi == "gabarit":
            fonction = lambda rappel: pipeline.construire_gabarit(oeil, rappel=rappel)
        elif quoi == "recaler":
            fonction = lambda rappel: pipeline.recaler(oeil, rappel=rappel)
        elif quoi == "controle":
            from ..analyse.controle import controle_negatif

            fonction = lambda rappel: controle_negatif(
                pipeline, base, oeil, rappel=rappel
            )
        elif quoi == "evolution":
            from ..analyse.execution import executer_evolution

            fonction = lambda rappel: executer_evolution(
                pipeline, base, oeil, rappel=rappel
            )
        elif quoi == "timelapse":
            from ..video.timelapse import generer_timelapse

            cible = pipeline.emp.exports / f"timelapse_{oeil}.mp4"
            fonction = lambda rappel: generer_timelapse(
                pipeline, base, oeil, cible, rappel=rappel
            )
        elif quoi == "evidence":
            from ..video.mise_en_evidence import generer_mise_en_evidence

            cible = pipeline.emp.exports / f"evolution_{oeil}.mp4"
            fonction = lambda rappel: generer_mise_en_evidence(
                pipeline, base, oeil, cible, rappel=rappel
            )
        else:
            return

        self._lancer_generique(fonction, quoi)

    def _lancer_generique(self, fonction, etiquette: str) -> None:
        self._activer_boutons(False)
        self.progression.setVisible(True)
        self.progression.setValue(0)

        self.ordonnanceur.lancer(
            lambda rappel: fonction(rappel),
            sur_progression=self._sur_progression,
            sur_fin=lambda r: self._sur_fin(etiquette, r),
            sur_echec=self._sur_echec,
            sur_message=self._message,
        )

    def _sur_progression(self, etiquette: str, i: int, n: int, detail: str) -> None:
        self.progression.setMaximum(max(n, 1))
        self.progression.setValue(i)
        self.etat.setText(f"{etiquette} · {i}/{n} · {detail}")

    def _sur_fin(self, etiquette: str, resultat) -> None:
        self._activer_boutons(True)
        self.progression.setVisible(False)

        if etiquette == "controle" and resultat is not None:
            self.rapport_controle.setPlainText(resultat.rendu_texte())
            self.plancher_bruit = resultat.plancher_objets()
            if self.serie_courante is not None:
                self.vue_graphiques.charger(
                    self.serie_courante, self.base.evenements(), self.plancher_bruit
                )
        elif etiquette == "evolution" and resultat is not None:
            self.serie_courante = resultat
            self.vue_graphiques.charger(
                resultat, self.base.evenements(), self.plancher_bruit
            )
            for m in resultat.messages:
                self.journal.appendPlainText(m)
        elif hasattr(resultat, "messages"):
            for m in resultat.messages[:20]:
                self.journal.appendPlainText(m)

        if resultat is not None:
            self.journal.appendPlainText(f"[{etiquette}] {resultat}")
        self.etat.setText(AVERTISSEMENT)
        self._rafraichir_bibliotheque()

    def _sur_echec(self, trace: str) -> None:
        self._activer_boutons(True)
        self.progression.setVisible(False)
        self.journal.appendPlainText(trace)
        QMessageBox.critical(self, "Erreur", trace.strip().splitlines()[-1])

    def _activer_boutons(self, actif: bool) -> None:
        for b in (
            self.bouton_normaliser,
            self.bouton_gabarit,
            self.bouton_recaler,
            self.bouton_controle,
            self.bouton_evolution,
            self.bouton_timelapse,
            self.bouton_evidence,
        ):
            b.setEnabled(actif)
        self.bouton_annuler.setEnabled(not actif)

    def _message(self, texte: str) -> None:
        self.etat.setText(texte)
        self.journal.appendPlainText(texte)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.vue_traitement.activer_clic(False)
            self.vue_traitement.arreter_bascule()
            self._reperes_en_saisie = None
        super().keyPressEvent(event)


def lancer(racine_data: str | None = None) -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    fenetre = FenetrePrincipale(racine_data)
    fenetre.show()
    return application.exec()
