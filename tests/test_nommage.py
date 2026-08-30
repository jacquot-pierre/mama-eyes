"""Analyse des noms de fichiers, y compris les formes historiques."""

from datetime import date

import pytest

from mama_eyes.stockage.nommage import analyser_nom, formater_nom


@pytest.mark.parametrize(
    "nom,oeil_defaut,attendu",
    [
        ("2018_02_19_D.JPG", None, (date(2018, 2, 19), "D", 1)),
        ("2025_09_02_G.jpg", None, (date(2025, 9, 2), "G", 1)),
        # Formes historiques rencontrees dans la bibliotheque d'origine.
        ("2019 03 25 OG.JPG", None, (date(2019, 3, 25), "G", 1)),
        ("2024 05 11 OD.JPG", None, (date(2024, 5, 11), "D", 1)),
        ("2022 09 15 D resveratrol.JPG", None, (date(2022, 9, 15), "D", 1)),
        ("2019 08 26 Gb.JPG", None, (date(2019, 8, 26), "G", 1)),
        ("2020 01 20 G .JPG", None, (date(2020, 1, 20), "G", 1)),
        ("2018-02-19-D.JPG", None, (date(2018, 2, 19), "D", 1)),
        # Oeil absent du nom : repli sur le dossier.
        ("2023 11 15 O.JPG", "D", (date(2023, 11, 15), "D", 1)),
        # Plusieurs cliches pour une meme seance.
        ("2024_01_07_D_2.JPG", None, (date(2024, 1, 7), "D", 2)),
    ],
)
def test_analyse(nom, oeil_defaut, attendu):
    resultat = analyser_nom(nom, oeil_par_defaut=oeil_defaut)
    assert resultat is not None, f"{nom} n'a pas ete analyse"
    assert (resultat.date, resultat.oeil, resultat.index) == attendu


@pytest.mark.parametrize(
    "nom", ["notes.txt", "sans_date_D.jpg", "2018_13_45_D.jpg", "image.gif"]
)
def test_rejets(nom):
    assert analyser_nom(nom) is None


def test_format_canonique():
    assert formater_nom(date(2018, 2, 19), "D") == "2018_02_19_D.jpg"
    assert formater_nom(date(2018, 2, 19), "G", 3) == "2018_02_19_G_3.jpg"


def test_aller_retour():
    """Un nom canonique doit se rananalyser a l'identique."""
    original = formater_nom(date(2021, 6, 14), "G", 2, ".png")
    relu = analyser_nom(original)
    assert relu is not None
    assert relu.nom_canonique() == original
