"""Controle negatif : detection des paires et suites, dates eligibles.

Le plancher de faux positifs n'a de sens que s'il est rapporte aux dates ou la
regle de confirmation pouvait effectivement se declencher. Rapporter au
nombre *total* de dates sous-estime le plancher sur les suites courtes, ou
les bords dominent - et laisse alors passer des comptages qui ne sont que du
bruit de segmentation.
"""

from datetime import date, timedelta

import pytest

from mama_eyes.analyse.controle import (
    PaireControle,
    PaireDetection,
    RapportControle,
    ResultatSequence,
    paires_rapprochees,
    sequences_rapprochees,
)


def suite(depart: date, ecarts: list[int]) -> list[date]:
    dates = [depart]
    for e in ecarts:
        dates.append(dates[-1] + timedelta(days=e))
    return dates


# --- reperage des paires et suites -------------------------------------------


def test_paires_rapprochees():
    dates = suite(date(2019, 9, 9), [7, 7, 60, 7])
    couples = paires_rapprochees(dates, 10)
    assert [(i, j) for i, j, _ in couples] == [(0, 1), (1, 2), (3, 4)]


def test_paires_ignore_les_grands_ecarts():
    dates = suite(date(2020, 1, 1), [30, 45, 130])
    assert paires_rapprochees(dates, 10) == []


def test_sequence_hebdomadaire_reperee():
    """La suite de six cliches hebdomadaires doit etre reconnue d'un bloc."""
    dates = suite(date(2019, 9, 9), [7, 7, 7, 7, 7])
    sequences = sequences_rapprochees(dates, 10, longueur_min=3)
    assert len(sequences) == 1
    assert sequences[0] == [0, 1, 2, 3, 4, 5]


def test_sequences_separees_par_un_grand_ecart():
    dates = suite(date(2019, 1, 1), [7, 7, 200, 7, 7])
    sequences = sequences_rapprochees(dates, 10, longueur_min=3)
    assert [len(s) for s in sequences] == [3, 3]


def test_suite_trop_courte_ecartee():
    dates = suite(date(2019, 1, 1), [7])
    assert sequences_rapprochees(dates, 10, longueur_min=3) == []


# --- dates eligibles ---------------------------------------------------------


@pytest.mark.parametrize(
    "nb_dates,confirmations,attendu",
    [
        (6, 2, 3),  # la suite hebdomadaire reelle
        (4, 2, 1),
        (3, 2, 0),
        (5, 1, 4),
        (2, 2, 0),
    ],
)
def test_dates_eligibles(nb_dates, confirmations, attendu):
    """La regle exige k dates avant et apres : les bords sont exclus."""
    r = ResultatSequence(
        dates=suite(date(2019, 9, 9), [7] * (nb_dates - 1)),
        apparitions=0,
        disparitions=0,
        fusions=0,
        scissions=0,
        nombres=[0] * nb_dates,
        surfaces=[0] * nb_dates,
        confirmations=confirmations,
    )
    assert r.dates_eligibles == attendu


def test_faux_positifs_par_date_normalises_sur_les_eligibles():
    """Normaliser sur le total sous-estimerait le plancher d'un facteur 2."""
    r = ResultatSequence(
        dates=suite(date(2019, 9, 9), [7] * 5),  # 6 dates
        apparitions=110,
        disparitions=392,
        fusions=0,
        scissions=0,
        nombres=[700] * 6,
        surfaces=[300000] * 6,
        confirmations=2,
    )
    assert r.dates_eligibles == 3
    assert r.faux_positifs_par_date == pytest.approx(502 / 3)
    # La normalisation naive donnerait 502/6 = 84, soit deux fois moins.
    assert r.faux_positifs_par_date > 502 / len(r.dates)


def test_aucune_division_par_zero_sur_suite_trop_courte():
    r = ResultatSequence(
        dates=suite(date(2019, 9, 9), [7, 7]),
        apparitions=5,
        disparitions=5,
        fusions=0,
        scissions=0,
        nombres=[1, 1, 1],
        surfaces=[1, 1, 1],
        confirmations=2,
    )
    assert r.dates_eligibles == 0
    assert r.faux_positifs_par_date == 0.0


# --- rapport -----------------------------------------------------------------


def _paire(objets: int, surface: int = 100) -> PaireControle:
    return PaireControle(
        date_a=date(2019, 9, 9),
        date_b=date(2019, 9, 16),
        ecart_jours=7,
        surface_apparue=surface,
        surface_disparue=0,
        objets_apparus=objets,
        objets_disparus=0,
        gain=1.0,
        ecart_type_diff=1.0,
        p99_diff=10.0,
        p999_diff=26.0,
    )


def test_plancher_objets_sur_percentile():
    rapport = RapportControle(oeil="D", paires=[_paire(n) for n in (0, 1, 2, 5, 7)])
    assert rapport.plancher_objets(95.0) >= 5


def test_seuil_suggere_prend_le_pire_cas():
    """Le maximum, pas la mediane : c'est la queue qui produit les faux objets."""
    paires = [_paire(0) for _ in range(4)]
    paires[0].p999_diff = 40.0
    rapport = RapportControle(oeil="D", paires=paires)
    assert rapport.seuil_diff_suggere() == 40.0


def test_metrique_recommandee_designe_la_plus_stable():
    detections = [
        PaireDetection(
            date_a=date(2019, 9, 9),
            date_b=date(2019, 9, 16),
            ecart_jours=7,
            nombre_a=100,
            nombre_b=130,  # 30 % d'instabilite
            surface_a=1000,
            surface_b=1050,  # 5 % d'instabilite
            non_apparies_a=30,
            non_apparies_b=30,
        )
    ]
    rapport = RapportControle(oeil="D", paires=[_paire(0)], detections=detections)
    assert rapport.metrique_recommandee() == "surface"


def test_rapport_vide_ne_plante_pas():
    rapport = RapportControle(oeil="D")
    assert "Aucune paire" in rapport.rendu_texte()
    assert rapport.resume() == {}
    assert rapport.plancher_objets() == 0.0
