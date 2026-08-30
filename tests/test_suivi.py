"""Suivi temporel : appariement, regle de confirmation, fusions.

Ces tests encodent la difference de fond avec l'approche par masque cumulatif :
un decompte doit pouvoir *ne pas* augmenter, et une detection isolee ne doit
jamais etre validee.
"""

from datetime import date, timedelta

import pytest

from mama_eyes.config import ParamsSuivi
from mama_eyes.domain import Druse
from mama_eyes.analyse.suivi import (
    apparier,
    compter_par_date,
    construire_pistes,
    detecter_fusions_scissions,
    evenements_confirmes,
)


def druse(x, y, surface=100):
    return Druse(
        seance_id=0,
        piste_id=None,
        cx=float(x),
        cy=float(y),
        surface_px=surface,
        excentricite=0.3,
        solidite=0.9,
        intensite_moy=180.0,
    )


def dates(n):
    depart = date(2020, 1, 1)
    return [depart + timedelta(days=21 * i) for i in range(n)]


# --- appariement -------------------------------------------------------------


def test_appariement_simple():
    a = [druse(100, 100), druse(300, 300)]
    b = [druse(305, 302), druse(102, 98)]  # ordre inverse
    couples = dict(apparier(a, b))
    assert couples[0] == 1
    assert couples[1] == 0


def test_appariement_refuse_au_dela_de_la_distance_max():
    params = ParamsSuivi(distance_max_px=20.0)
    a = [druse(100, 100)]
    b = [druse(400, 400)]
    assert apparier(a, b, params) == []


def test_appariement_tient_compte_de_la_surface():
    """A distance egale, la druse de surface comparable doit gagner."""
    params = ParamsSuivi(distance_max_px=60.0, poids_surface=1.0)
    a = [druse(100, 100, surface=100)]
    b = [druse(120, 100, surface=5000), druse(80, 100, surface=105)]
    couples = dict(apparier(a, b, params))
    assert couples[0] == 1


# --- pistes ------------------------------------------------------------------


def test_piste_continue_recoit_une_identite_stable():
    detections = [[druse(100, 100)], [druse(102, 101)], [druse(101, 103)]]
    pistes = construire_pistes(detections)
    actives = [p for p in pistes if len(p.presences) == 3]
    assert len(actives) == 1
    assert actives[0].duree() == 3


def test_deux_druses_independantes_donnent_deux_pistes():
    detections = [
        [druse(100, 100), druse(500, 500)],
        [druse(101, 101), druse(502, 499)],
    ]
    pistes = construire_pistes(detections)
    assert len([p for p in pistes if len(p.presences) == 2]) == 2


# --- regle de confirmation ---------------------------------------------------


def test_disparition_confirmee_sur_deux_dates():
    """Presente en 0,1,2 puis absente en 3 et 4 : disparition confirmee en 3."""
    detections = [
        [druse(100, 100)],
        [druse(100, 100)],
        [druse(100, 100)],
        [],
        [],
    ]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(5))
    disparitions = [e for e in evenements if e.type == "disparition"]
    assert len(disparitions) == 1
    assert disparitions[0].index_date == 3


def test_absence_isolee_non_confirmee():
    """Une absence sur une seule date ne doit pas valider une disparition.

    C'est tout l'interet de la regle : un cliche flou ou mal expose ne doit
    pas produire un evenement.
    """
    detections = [
        [druse(100, 100)],
        [druse(100, 100)],
        [],  # absence isolee
        [druse(100, 100)],
        [druse(100, 100)],
    ]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(5))
    assert [e for e in evenements if e.type == "disparition"] == []


def test_apparition_confirmee():
    detections = [[], [], [druse(200, 200)], [druse(201, 201)], [druse(200, 202)]]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(5))
    apparitions = [e for e in evenements if e.type == "apparition"]
    assert len(apparitions) == 1
    assert apparitions[0].index_date == 2


def test_apparition_isolee_non_confirmee():
    detections = [[], [], [druse(200, 200)], [], []]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(5))
    assert [e for e in evenements if e.type == "apparition"] == []


def test_serie_stable_ne_produit_aucun_evenement():
    """Le controle negatif du suivi : rien ne bouge, rien n'est signale."""
    detections = [[druse(100, 100), druse(300, 300)] for _ in range(6)]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(6))
    assert evenements == []


def test_le_decompte_peut_diminuer():
    """Contrairement a un masque cumulatif, le decompte n'est pas monotone."""
    detections = [
        [druse(100, 100), druse(300, 300)],
        [druse(100, 100), druse(300, 300)],
        [druse(100, 100)],
        [druse(100, 100)],
    ]
    nombres = [len(d) for d in detections]
    assert nombres[-1] < nombres[0]

    pistes = construire_pistes(detections)
    comptes = compter_par_date(evenements_confirmes(pistes, dates(4)), 4)
    assert sum(comptes["disparition"]) == 1
    assert sum(comptes["apparition"]) == 0


# --- fusions et scissions ----------------------------------------------------


def test_fusion_signalee_separement():
    """Deux druses qui fusionnent ne doivent pas compter comme une disparition."""
    params = ParamsSuivi(distance_max_px=60.0)
    detections = [
        [druse(100, 100, 200), druse(140, 100, 200)],
        [druse(120, 100, 420)],
    ]
    evenements = detecter_fusions_scissions(detections, dates(2), params)
    assert any(e.type == "fusion" for e in evenements)


def test_scission_signalee_separement():
    params = ParamsSuivi(distance_max_px=60.0)
    detections = [
        [druse(120, 100, 420)],
        [druse(100, 100, 200), druse(140, 100, 200)],
    ]
    evenements = detecter_fusions_scissions(detections, dates(2), params)
    assert any(e.type == "scission" for e in evenements)


@pytest.mark.parametrize("confirmations", [1, 2, 3])
def test_nombre_de_confirmations_reglable(confirmations):
    """Une absence de longueur `k` valide si k >= confirmations requises."""
    params = ParamsSuivi(confirmations_requises=confirmations)
    detections = [[druse(100, 100)], [druse(100, 100)]] + [[] for _ in range(4)]
    pistes = construire_pistes(detections)
    evenements = evenements_confirmes(pistes, dates(6), params)
    assert len([e for e in evenements if e.type == "disparition"]) == 1
