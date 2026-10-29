import pytest

from pgx_rl import elo


def test_even_match_is_zero_elo():
    est = elo.estimate(10, 20, 10)
    assert est.elo == pytest.approx(0)
    assert est.low < 0 < est.high


def test_known_score_maps_to_elo():
    # a 75% score is about +191 Elo
    assert elo.score_to_elo(0.75) == pytest.approx(190.85, abs=0.1)
    assert elo.estimate(75, 0, 25).elo == pytest.approx(190.85, abs=0.1)


def test_more_games_tighten_interval():
    small, large = elo.estimate(6, 4, 2), elo.estimate(60, 40, 20)
    assert (large.high - large.low) < (small.high - small.low)


def test_sweep_has_finite_interval():
    est = elo.estimate(10, 0, 0)
    assert est.low < est.elo and est.high < float("inf")


def test_no_games_rejected():
    with pytest.raises(ValueError):
        elo.estimate(0, 0, 0)
