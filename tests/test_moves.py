import random

import chess
import jax
import numpy as np
import pgx

from pgx_rl import moves

env = pgx.make("chess")
init, step = jax.jit(env.init), jax.jit(env.step)

PROMO_FEN = "8/P6k/8/8/8/8/p6K/8 w - - 0 1"


def test_random_games_match_pgx_legal_actions():
    rng = random.Random(0)
    for game in range(3):
        board = chess.Board()
        state = init(jax.random.PRNGKey(game))
        for _ in range(120):
            if board.is_game_over(claim_draw=False) or bool(state.terminated):
                break
            expected = moves.legal_action_mask(board)
            assert (expected == np.asarray(state.legal_action_mask)).all(), board.fen()
            move = rng.choice(list(board.legal_moves))
            action = moves.move_to_action(move, board.turn)
            assert moves.action_to_move(action, board) == move
            board.push(move)
            state = step(state, action)


def test_underpromotions_and_castling_round_trip():
    for fen in [
        PROMO_FEN,
        "7k/8/8/8/8/8/p6K/1R6 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ]:
        board = chess.Board(fen)
        for move in board.legal_moves:
            action = moves.move_to_action(move, board.turn)
            assert moves.action_to_move(action, board) == move
