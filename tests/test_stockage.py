"""Base et bibliotheque : deduplication, seances multi-cliches, cache."""

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from mama_eyes.config import Emplacements, ParamsDruses, Parametres
from mama_eyes.domain import Cliche, Reperes
from mama_eyes.stockage.base import Base
from mama_eyes.stockage.bibliotheque import Bibliotheque


@pytest.fixture
def bibliotheque(tmp_path):
    emp = Emplacements(tmp_path / "data")
    base = Base(emp.base)
    yield Bibliotheque(emp, base), base
    base.fermer()


def _ecrire_image(chemin: Path, teinte: int = 120) -> Path:
    image = np.full((240, 320, 3), teinte, dtype=np.uint8)
    cv2.circle(image, (160, 120), 90, (40, 90, 170), -1)
    cv2.imwrite(str(chemin), image)
    return chemin


# --- import ------------------------------------------------------------------


def test_import_cree_seance_et_cliche(bibliotheque, tmp_path):
    biblio, base = bibliotheque
    source = _ecrire_image(tmp_path / "2020_05_06_D.jpg")

    cliche = biblio.importer_fichier(source)
    assert cliche is not None
    assert cliche.fichier == "D/2020_05_06_D.jpg"

    seances = base.seances("D")
    assert len(seances) == 1
    assert seances[0].date == date(2020, 5, 6)
    assert len(seances[0].cliches) == 1


def test_deduplication_par_contenu(bibliotheque, tmp_path):
    """Reimporter le meme contenu sous un autre nom ne cree pas de doublon."""
    biblio, base = bibliotheque
    source = _ecrire_image(tmp_path / "2020_05_06_D.jpg")
    assert biblio.importer_fichier(source) is not None

    copie = tmp_path / "2020_05_06_D_copie.jpg"
    copie.write_bytes(source.read_bytes())
    assert biblio.importer_fichier(copie) is None

    assert len(base.seances("D")[0].cliches) == 1


def test_plusieurs_cliches_pour_une_meme_seance(bibliotheque, tmp_path):
    biblio, base = bibliotheque
    for i, teinte in enumerate((100, 140, 180), start=1):
        source = _ecrire_image(tmp_path / f"prise_{i}.jpg", teinte)
        biblio.importer_fichier(source, oeil="G", jour=date(2021, 3, 8))

    seances = base.seances("G")
    assert len(seances) == 1
    assert len(seances[0].cliches) == 3
    noms = sorted(c.fichier for c in seances[0].cliches)
    assert noms == [
        "G/2021_03_08_G.jpg",
        "G/2021_03_08_G_2.jpg",
        "G/2021_03_08_G_3.jpg",
    ]


def test_import_refuse_sans_date(bibliotheque, tmp_path):
    biblio, _ = bibliotheque
    source = _ecrire_image(tmp_path / "sans_date.jpg")
    with pytest.raises(ValueError, match="date"):
        biblio.importer_fichier(source, oeil="D")


def test_brut_immuable_pas_d_ecrasement(bibliotheque, tmp_path):
    """Deux contenus differents le meme jour cohabitent, sans ecrasement."""
    biblio, base = bibliotheque
    a = _ecrire_image(tmp_path / "a.jpg", 100)
    b = _ecrire_image(tmp_path / "b.jpg", 200)
    biblio.importer_fichier(a, oeil="D", jour=date(2022, 1, 10))
    biblio.importer_fichier(b, oeil="D", jour=date(2022, 1, 10))

    fichiers = sorted(p.name for p in biblio.emp.brut_oeil("D").iterdir())
    assert fichiers == ["2022_01_10_D.jpg", "2022_01_10_D_2.jpg"]


def test_nettete_calculee_a_l_import(bibliotheque, tmp_path):
    biblio, base = bibliotheque
    source = _ecrire_image(tmp_path / "2020_05_06_D.jpg")
    cliche = biblio.importer_fichier(source)
    assert cliche.nettete is not None and cliche.nettete >= 0


# --- exclusion et evenements -------------------------------------------------


def test_exclusion_retire_des_cliches_actifs(bibliotheque, tmp_path):
    biblio, base = bibliotheque
    source = _ecrire_image(tmp_path / "2020_05_06_D.jpg")
    cliche = biblio.importer_fichier(source)

    base.definir_exclusion(cliche.id, True)
    seance = base.seances("D")[0]
    assert seance.cliches_actifs == []
    assert len(seance.cliches) == 1


def test_evenements_tries(bibliotheque):
    _, base = bibliotheque
    base.ajouter_evenement(date(2023, 1, 1), "b")
    base.ajouter_evenement(date(2022, 9, 15), "resveratrol")
    libelles = [e.libelle for e in base.evenements()]
    assert libelles == ["resveratrol", "b"]


# --- reperes et calibration --------------------------------------------------


def test_reperes_aller_retour(bibliotheque):
    _, base = bibliotheque
    r = Reperes("D", 400.0, 300.0, 90.0, 850.0, 320.0, manuel=True)
    base.enregistrer_reperes(r)
    relu = base.reperes("D")
    assert relu is not None
    assert relu.disque_x == 400.0
    assert relu.manuel is True


def test_reperes_remplaces_et_non_dupliques(bibliotheque):
    _, base = bibliotheque
    base.enregistrer_reperes(Reperes("D", 1.0, 2.0, 10.0, 3.0, 4.0))
    base.enregistrer_reperes(Reperes("D", 9.0, 8.0, 20.0, 7.0, 6.0))
    assert base.reperes("D").disque_x == 9.0


def test_calibration_echelle():
    """1,85 mm de diametre papillaire donnent l'echelle px/mm."""
    r = Reperes("D", 0, 0, disque_rayon_px=185.0, macula_x=0, macula_y=0)
    assert r.px_par_mm == pytest.approx(370.0 / 1.85)
    # Une druse de 125 um doit faire une taille plausible en pixels.
    assert 20 < r.um_en_px(125.0) < 30
    assert r.px_en_um(r.um_en_px(63.0)) == pytest.approx(63.0)


# --- cache par empreinte -----------------------------------------------------


def test_empreinte_change_avec_les_parametres():
    a = Parametres()
    b = a.avec(druses=ParamsDruses(bloc_seuil_px=201))
    assert a.empreinte() != b.empreinte()


def test_empreinte_stable_pour_parametres_identiques():
    assert Parametres().empreinte() == Parametres().empreinte()


def test_parametres_serialisation_aller_retour():
    a = Parametres().avec(druses=ParamsDruses(offset_seuil=-9.5))
    b = Parametres.depuis_json(a.en_json())
    assert b.druses.offset_seuil == -9.5
    assert b.empreinte() == a.empreinte()


def test_chemin_derive_depend_de_l_empreinte(tmp_path):
    from mama_eyes.pipeline import Pipeline

    emp = Emplacements(tmp_path / "data")
    base = Base(emp.base)
    biblio = Bibliotheque(emp, base)

    p1 = Pipeline(biblio, base, Parametres())
    p2 = Pipeline(
        biblio, base, Parametres().avec(druses=ParamsDruses(bloc_seuil_px=201))
    )
    jour = date(2020, 5, 6)
    # Les parametres de druses n'interviennent pas dans la normalisation.
    assert p1.chemin_normalise("D", jour) == p2.chemin_normalise("D", jour)

    from mama_eyes.config import ParamsIllumination

    p3 = Pipeline(
        biblio,
        base,
        Parametres().avec(illumination=ParamsIllumination(sigma_fond_fraction=0.2)),
    )
    assert p1.chemin_normalise("D", jour) != p3.chemin_normalise("D", jour)
    base.fermer()
