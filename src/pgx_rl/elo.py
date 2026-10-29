"""Elo difference estimates with confidence intervals from match results."""

import math
from dataclasses import dataclass


@dataclass
class Estimate:
    elo: float
    low: float
    high: float
    score: float
    games: int

    def __str__(self):
        return (
            f"{self.elo:+.0f} Elo [{self.low:+.0f}, {self.high:+.0f}] 95% CI "
            f"(score {self.score:.3f} over {self.games} games)"
        )


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / score - 1)


def estimate(wins: int, draws: int, losses: int, z: float = 1.96) -> Estimate:
    """Elo of the first player relative to the opponent, normal approximation on the score.

    Draws shrink the variance, so the interval is computed from the per-game score
    variance rather than a binomial. With a perfect or zero score the interval is
    one-sided and clipped by a half-game continuity correction.
    """
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / n
    margin = z * math.sqrt(var / n)
    if wins == n or losses == n:
        margin = max(margin, 0.5 / n)
    lo, hi = max(score - margin, 0.5 / n), min(score + margin, 1 - 0.5 / n)
    return Estimate(score_to_elo(score), score_to_elo(lo), score_to_elo(hi), score, n)
