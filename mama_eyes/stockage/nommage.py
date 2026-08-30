"""Analyse et generation des noms de fichiers.

Les cliches ne portent aucune metadonnee EXIF : le nom de fichier est
l'unique source de date. Le format canonique est ``AAAA_MM_JJ_[DG][_n].ext``,
le suffixe ``_n`` distinguant plusieurs cliches d'une meme seance.

L'analyse reste tolerante aux formes historiques (espaces, ``OD``/``OG``,
mentions libres) afin de pouvoir ingerer une bibliotheque non normalisee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Date en tete : 2018_02_19, 2018-02-19 ou 2018 02 19
_MOTIF_DATE = re.compile(r"^(\d{4})[ _-](\d{2})[ _-](\d{2})")

# Lettre d'oeil isolee : D, G, OD, OG (evite de capturer le D de 'DROIT')
_MOTIF_OEIL = re.compile(r"(?:^|[ _-])(O?[DG])(?=[ _-]|\.|$|[a-z])")

# Suffixe d'index explicite : _2, -3
_MOTIF_INDEX = re.compile(r"[ _-](\d{1,2})(?=\.|$)")

EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageNommee:
    date: date
    oeil: str
    index: int
    extension: str

    def nom_canonique(self) -> str:
        base = f"{self.date:%Y_%m_%d}_{self.oeil}"
        if self.index > 1:
            base += f"_{self.index}"
        return base + self.extension


def analyser_nom(nom: str, oeil_par_defaut: str | None = None) -> ImageNommee | None:
    """Extrait date, oeil et index d'un nom de fichier.

    `oeil_par_defaut` sert de repli quand le nom ne porte pas la lettre
    (cas d'un fichier range dans un dossier par oeil).
    """
    chemin_nom = nom.rsplit("/", 1)[-1]
    tige, _, ext = chemin_nom.rpartition(".")
    if not tige:
        tige, ext = chemin_nom, ""
    extension = ("." + ext).lower() if ext else ".jpg"
    if extension not in EXTENSIONS:
        return None

    m_date = _MOTIF_DATE.match(tige)
    if not m_date:
        return None
    try:
        jour = date(int(m_date.group(1)), int(m_date.group(2)), int(m_date.group(3)))
    except ValueError:
        return None

    reste = tige[m_date.end() :]

    oeil = None
    m_oeil = _MOTIF_OEIL.search(reste)
    if m_oeil:
        oeil = m_oeil.group(1)[-1]  # 'OD' -> 'D', 'OG' -> 'G'
    elif oeil_par_defaut in ("D", "G"):
        oeil = oeil_par_defaut
    if oeil not in ("D", "G"):
        return None

    index = 1
    m_index = _MOTIF_INDEX.search(reste)
    if m_index:
        index = max(1, int(m_index.group(1)))

    return ImageNommee(date=jour, oeil=oeil, index=index, extension=extension)


def formater_nom(jour: date, oeil: str, index: int = 1, extension: str = ".jpg") -> str:
    return ImageNommee(jour, oeil, index, extension.lower()).nom_canonique()
