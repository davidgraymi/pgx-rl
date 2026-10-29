import shutil

import chess
import chess.engine
import jax
import pytest

from pgx_rl import chess as zero
from pgx_rl import evaluate

stockfish = shutil.which("stockfish")


def test_openings_are_legal_and_not_terminal():
    for line in evaluate.random_openings(5, 4, seed=1):
        board = chess.Board()
        for move in line:
            assert move in board.legal_moves
            board.push(move)
        assert not board.is_game_over()


@pytest.mark.skipif(stockfish is None, reason="stockfish not installed")
def test_tiny_match_against_stockfish():
    model = zero.Net(channels=8, blocks=1)
    key = jax.random.PRNGKey(0)
    params = model.init(
        key, jax.vmap(zero.env.init)(jax.random.split(key, 1)).observation
    )
    act = evaluate.make_agent(model, simulations=0, considered=2)
    engines = [chess.engine.SimpleEngine.popen_uci(stockfish) for _ in range(2)]
    try:
        w, d, l, truncated, games = evaluate.play_match(
            act,
            params,
            key,
            evaluate.random_openings(1, 2, seed=0),
            engines,
            chess.engine.Limit(nodes=50),
            max_plies=12,
        )
    finally:
        for engine in engines:
            engine.quit()
    assert w + d + l == 2 and len(games) == 2
    assert truncated <= 2
