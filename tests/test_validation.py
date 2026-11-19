import pytest

from bikeml.data.validate import DataValidationError, assert_valid, validate_batch


def failed(report):
    return {c.name for c in report.checks if not c.passed}


def test_lot_conforme(cfg, good_batch):
    report = assert_valid(good_batch, cfg)
    assert report.passed and report.n_rows == len(good_batch)


def test_colonne_manquante(cfg, good_batch):
    assert "schema" in failed(validate_batch(good_batch.drop(columns=["temp"]), cfg))


def test_valeurs_manquantes(cfg, good_batch):
    good_batch.loc[0, "hum"] = None
    assert "completude" in failed(validate_batch(good_batch, cfg))


def test_hors_bornes(cfg, good_batch):
    good_batch.loc[0, "hr"] = 25
    good_batch.loc[1, "cnt"] = -1
    assert "bornes" in failed(validate_batch(good_batch, cfg))


def test_doublons(cfg, good_batch):
    import pandas as pd
    dup = pd.concat([good_batch, good_batch.head(5)])
    assert "unicite" in failed(validate_batch(dup, cfg))


def test_incoherence_cible(cfg, good_batch):
    good_batch.loc[0, "cnt"] += 3
    assert "coherence" in failed(validate_batch(good_batch, cfg))


def test_lot_trop_petit(cfg, good_batch):
    assert "volume" in failed(validate_batch(good_batch.head(10), cfg))


def test_erreur_bloquante(cfg, good_batch):
    with pytest.raises(DataValidationError) as exc:
        assert_valid(good_batch.drop(columns=["cnt"]), cfg)
    assert "schema" in str(exc.value)


def test_valeur_non_numerique(cfg, good_batch):
    # "chaud" n'est ni manquant ni hors bornes : seul le contrôle de type le voit
    good_batch["temp"] = good_batch["temp"].astype(object)
    good_batch.loc[3, "temp"] = "chaud"
    assert failed(validate_batch(good_batch, cfg)) == {"types"}
