import sys
import os
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QProgressBar, QMessageBox)
from PyQt6.QtCore import pyqtSignal, QObject

import align_images
import analyze_drusen
import create_video

class WorkerSignals(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

class MamaEyesGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mama-Eyes: Suivi de Druses")
        self.resize(800, 600)
        
        self.input_dir = ""
        self.output_dir = ""
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.log_message)
        self.signals.finished.connect(self.on_process_finished)
        self.signals.error.connect(self.on_process_error)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # File Selection Section
        file_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        self.lbl_input = QLabel("Dossier d'origine (images brutes) : Non sélectionné")
        btn_input = QPushButton("Parcourir...")
        btn_input.clicked.connect(self.select_input_dir)
        input_layout.addWidget(self.lbl_input)
        input_layout.addWidget(btn_input)
        file_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.lbl_output = QLabel("Dossier de sortie : Non sélectionné")
        btn_output = QPushButton("Parcourir...")
        btn_output.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.lbl_output)
        output_layout.addWidget(btn_output)
        file_layout.addLayout(output_layout)

        layout.addLayout(file_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_align = QPushButton("1. Aligner les images")
        self.btn_align.clicked.connect(lambda: self.start_process("align"))
        
        self.btn_analyze = QPushButton("2. Analyser l'évolution")
        self.btn_analyze.clicked.connect(lambda: self.start_process("analyze"))
        
        self.btn_video = QPushButton("3. Générer les vidéos")
        self.btn_video.clicked.connect(lambda: self.start_process("video"))
        
        self.btn_all = QPushButton("Tout exécuter (1+2+3)")
        self.btn_all.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_all.clicked.connect(lambda: self.start_process("all"))

        btn_layout.addWidget(self.btn_align)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_all)
        
        layout.addLayout(btn_layout)

        # Progress / Console
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")
        layout.addWidget(self.console)

    def select_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier d'images")
        if dir_path:
            self.input_dir = dir_path
            self.lbl_input.setText(f"Dossier d'origine : {self.input_dir}")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if dir_path:
            self.output_dir = dir_path
            self.lbl_output.setText(f"Dossier de sortie : {self.output_dir}")

    def log_message(self, message):
        self.console.append(message)

    def set_ui_enabled(self, enabled):
        self.btn_align.setEnabled(enabled)
        self.btn_analyze.setEnabled(enabled)
        self.btn_video.setEnabled(enabled)
        self.btn_all.setEnabled(enabled)

    def start_process(self, process_type):
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner les dossiers d'entrée et de sortie.")
            return

        self.set_ui_enabled(False)
        self.progress_bar.show()
        self.log_message(f"--- Démarrage du processus : {process_type} ---")

        # Run in a separate thread so GUI doesn't freeze
        thread = threading.Thread(target=self.run_scripts, args=(process_type,))
        thread.start()

    def run_scripts(self, process_type):
        try:
            aligned_dir = os.path.join(self.output_dir, "aligned_images")
            analysis_dir = os.path.join(self.output_dir, "analysis_results")
            
            if process_type in ["align", "all"]:
                self.signals.progress.emit("== Démarrage de l'alignement ==")
                align_images.align_images(self.input_dir, aligned_dir)
                
            if process_type in ["analyze", "all"]:
                self.signals.progress.emit("== Démarrage de l'analyse ==")
                analyze_drusen.analyze_drusen_regression(aligned_dir, analysis_dir)
                
            if process_type in ["video", "all"]:
                self.signals.progress.emit("== Génération des vidéos ==")
                output_neutral = os.path.join(self.output_dir, "evolution_neutre.mp4")
                output_enriched = os.path.join(self.output_dir, "evolution_enrichie.mp4")
                json_stats = os.path.join(analysis_dir, "evolution_stats.json")
                create_video.create_video_from_images(aligned_dir, output_neutral, output_enriched, json_stats=json_stats)
            
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def on_process_finished(self):
        self.progress_bar.hide()
        self.set_ui_enabled(True)
        self.log_message("--- Processus terminé avec succès ---")
        QMessageBox.information(self, "Succès", "Le traitement est terminé.")

    def on_process_error(self, error_msg):
        self.progress_bar.hide()
        self.set_ui_enabled(True)
        self.log_message(f"ERREUR : {error_msg}")
        QMessageBox.critical(self, "Erreur", f"Une erreur s'est produite :\n{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MamaEyesGUI()
    window.show()
    sys.exit(app.exec())
