"""Conversion between python-chess moves and pgx chess actions.

pgx squares are ``file * 8 + rank`` (a1 = 0, a2 = 1, ..., h8 = 63) in the frame of the
player to move: for black the board is mirrored vertically (rank -> 7 - rank). Actions
are ``from * 73 + plane`` (AlphaZero encoding); queen promotions are ordinary moves and
underpromotions use planes 0-8.
"""

import chess
import numpy as np
from pgx._src.games.chess import FROM_PLANE, TO_PLANE

UNDERPROMOTION = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
UNDERPROMOTION_PIECE = {v: k for k, v in UNDERPROMOTION.items()}
FILE_STEP = {0: 0, 1: 1, 2: -1}  # plane % 3: straight, capture right, capture left


def to_pgx_square(square: int, color: bool) -> int:
    file, rank = chess.square_file(square), chess.square_rank(square)
    return file * 8 + (rank if color == chess.WHITE else 7 - rank)


def from_pgx_square(square: int, color: bool) -> int:
    file, rank = square // 8, square % 8
    return chess.square(file, rank if color == chess.WHITE else 7 - rank)


def move_to_action(move: chess.Move, color: bool) -> int:
    """Action label for ``move`` played by ``color`` (the side to move)."""
    src = to_pgx_square(move.from_square, color)
    dst = to_pgx_square(move.to_square, color)
    if move.promotion in UNDERPROMOTION:
        step = {0: 0, 1: 1, -1: 2}[
            chess.square_file(move.to_square) - chess.square_file(move.from_square)
        ]
        return src * 73 + UNDERPROMOTION[move.promotion] * 3 + step
    return src * 73 + int(TO_PLANE[src, dst])


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Move for ``action`` on ``board``; castling is returned as python-chess expects."""
    color = board.turn
    src, plane = divmod(int(action), 73)
    dst = int(FROM_PLANE[src, plane])
    from_square = from_pgx_square(src, color)
    to_square = from_pgx_square(dst, color)
    promotion = None
    if plane < 9:
        promotion = UNDERPROMOTION_PIECE[plane // 3]
    elif board.piece_type_at(from_square) == chess.PAWN and chess.square_rank(
        to_square
    ) in (0, 7):
        promotion = chess.QUEEN
    # pgx encodes castling as a two-square king move; python-chess agrees (e1g1), but
    # normalise in case the board is Chess960-style
    return chess.Move(from_square, to_square, promotion)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(4672, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_action(move, board.turn)] = True
    return mask
