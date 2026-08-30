"""Import et organisation des fichiers image.

`brut/` est immuable apres import. La deduplication se fait par SHA-256 du
contenu : reimporter deux fois le meme fichier est sans effet, meme s'il a
ete renomme entre-temps.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import Emplacements
from ..domain import Cliche
from .base import Base
from .nommage import EXTENSIONS, analyser_nom, formater_nom


def sha256_fichier(chemin: Path, taille_bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while bloc := f.read(taille_bloc):
            h.update(bloc)
    return h.hexdigest()


def indice_nettete(image_bgr: np.ndarray) -> float:
    """Variance du laplacien : plus c'est haut, plus l'image est nette.

    Sert a departager les cliches d'une meme seance et a reperer les
    comparaisons entre cliches de qualite tres differente (cf. §5.8 : le
    nombre de druses detectees derive avec la nettete autant qu'avec la
    maladie).
    """
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if max(gris.shape) > 1200:
        facteur = 1200 / max(gris.shape)
        gris = cv2.resize(gris, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


@dataclass
class ResultatImport:
    importes: int = 0
    doublons: int = 0
    ignores: list[str] = field(default_factory=list)
    erreurs: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.importes} importe(s)"]
        if self.doublons:
            parts.append(f"{self.doublons} doublon(s)")
        if self.ignores:
            parts.append(f"{len(self.ignores)} ignore(s)")
        if self.erreurs:
            parts.append(f"{len(self.erreurs)} erreur(s)")
        return ", ".join(parts)


class Bibliotheque:
    """Fait le lien entre les fichiers sur disque et la base."""

    def __init__(self, emplacements: Emplacements, base: Base):
        self.emp = emplacements
        self.base = base
        self.emp.creer()

    # -- import --------------------------------------------------------------

    def importer_fichier(
        self,
        source: Path,
        oeil: str | None = None,
        jour: date | None = None,
        copier: bool = True,
    ) -> Cliche | None:
        """Importe un cliche. Renvoie None si c'est un doublon.

        `oeil` et `jour` priment sur ce que dit le nom de fichier ; s'ils sont
        absents, ils en sont deduits. Aucune date n'est devinee autrement :
        les fichiers ne portent pas d'EXIF.
        """
        source = Path(source)
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"extension non geree : {source.name}")

        analyse = analyser_nom(source.name, oeil_par_defaut=oeil)
        if jour is None:
            if analyse is None:
                raise ValueError(
                    f"date introuvable dans le nom {source.name!r} et non fournie"
                )
            jour = analyse.date
        if oeil is None:
            if analyse is None:
                raise ValueError(f"oeil introuvable dans le nom {source.name!r}")
            oeil = analyse.oeil

        sha = sha256_fichier(source)
        if self.base.sha_connu(sha):
            return None

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"image illisible : {source}")
        hauteur, largeur = image.shape[:2]
        nettete = indice_nettete(image)

        seance_id = self.base.seance_id(oeil, jour, creer=True)
        index = self._prochain_index(seance_id)
        nom_cible = formater_nom(jour, oeil, index, source.suffix)
        dossier = self.emp.brut_oeil(oeil)
        dossier.mkdir(parents=True, exist_ok=True)
        cible = dossier / nom_cible

        # Ne jamais ecraser : le brut est immuable.
        n = index
        while cible.exists():
            n += 1
            cible = dossier / formater_nom(jour, oeil, n, source.suffix)

        if copier:
            shutil.copy2(source, cible)
        else:
            shutil.move(str(source), cible)

        cliche = Cliche(
            id=None,
            seance_id=seance_id,
            fichier=f"{oeil}/{cible.name}",
            sha256=sha,
            largeur=largeur,
            hauteur=hauteur,
            nettete=nettete,
            exclu=False,
            importe_le=datetime.now().isoformat(timespec="seconds"),
        )
        cliche.id = self.base.ajouter_cliche(cliche)
        return cliche

    def importer_dossier(
        self,
        dossier: Path,
        oeil: str | None = None,
        copier: bool = True,
        rappel=None,
    ) -> ResultatImport:
        """Importe recursivement un dossier de cliches."""
        dossier = Path(dossier)
        res = ResultatImport()
        fichiers = sorted(
            p
            for p in dossier.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        for i, chemin in enumerate(fichiers):
            try:
                cliche = self.importer_fichier(chemin, oeil=oeil, copier=copier)
                if cliche is None:
                    res.doublons += 1
                else:
                    res.importes += 1
            except ValueError as exc:
                res.ignores.append(f"{chemin.name} : {exc}")
            except OSError as exc:
                res.erreurs.append(f"{chemin.name} : {exc}")
            if rappel:
                rappel(i + 1, len(fichiers), chemin.name)
        return res

    def _prochain_index(self, seance_id: int) -> int:
        ligne = self.base.cx.execute(
            "SELECT COUNT(*) AS n FROM cliche WHERE seance_id = ?", (seance_id,)
        ).fetchone()
        return int(ligne["n"]) + 1

    # -- lecture -------------------------------------------------------------

    def chemin_cliche(self, cliche: Cliche) -> Path:
        return self.emp.brut / cliche.fichier

    def lire(self, cliche: Cliche) -> np.ndarray:
        chemin = self.chemin_cliche(cliche)
        image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"image illisible : {chemin}")
        return image

    def chemin_derive(self, oeil: str, etape: str, nom: str) -> Path:
        dossier = self.emp.derive_oeil(oeil, etape)
        dossier.mkdir(parents=True, exist_ok=True)
        return dossier / nom

    def image_reference(self, oeil: str) -> np.ndarray | None:
        """Image « saine » nettoyee manuellement, si l'utilisateur en a fourni une."""
        dossier = self.emp.reference_oeil(oeil)
        if not dossier.is_dir():
            return None
        for chemin in sorted(dossier.iterdir()):
            if chemin.suffix.lower() in EXTENSIONS:
                image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        return None

    def definir_reference(self, oeil: str, source: Path) -> Path:
        dossier = self.emp.reference_oeil(oeil)
        dossier.mkdir(parents=True, exist_ok=True)
        cible = dossier / f"reference_{oeil}{Path(source).suffix.lower()}"
        shutil.copy2(source, cible)
        return cible
