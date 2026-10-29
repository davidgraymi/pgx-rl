"""Stockfish value labels for dataset shards.

Positions come from replaying each stored game with python-chess (the same actions the
shard holds), so no FEN round trip through pgx is needed. A label is Stockfish's score
for the side to move, squashed to [-1, 1] with the Lichess win-probability curve, and
NaN where a position was skipped by ``--stride``. Labels live in sidecar files so the
raw shards keep their hashes; the manifest pins both, and ``dataset_id`` changes when
labels do.
"""

import argparse
import contextlib
import math
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np

from pgx_rl import data, moves, repro

LICHESS_CP_SLOPE = 0.00368208  # win% = 50 + 50 * (2 / (1 + exp(-slope * cp)) - 1)


def value_from_score(score: chess.engine.Score) -> float:
    """Side-to-move value in [-1, 1] from a relative Stockfish score."""
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    return math.tanh(LICHESS_CP_SLOPE / 2 * score.score())


def label_game(engine, limit, actions, gid, stride):
    """Values for one game; plies are labeled when ``t % stride`` equals a game-id offset."""
    out = np.full(len(actions), np.nan, dtype=np.float16)
    offset = int.from_bytes(gid, "big") % stride
    board = chess.Board()
    for t, action in enumerate(actions):
        move = moves.action_to_move(int(action), board)
        if t % stride == offset:
            info = engine.analyse(board, limit, game=(gid, t))
            out[t] = value_from_score(info["score"].relative)
        board.push(move)
    return out


def label_shard(shard, engines, limit, stride):
    """Label every game of a loaded shard, one game per engine at a time."""
    free = queue.Queue()
    for engine in engines:
        free.put(engine)

    def work(game):
        engine = free.get()
        try:
            return label_game(engine, limit, *game, stride)
        finally:
            free.put(engine)

    games = list(zip(data.game_slices(shard), map(bytes, shard["game_id"])))
    with ThreadPoolExecutor(len(engines)) as pool:
        values = list(pool.map(work, games))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float16)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Label dataset shards with Stockfish values."
    )
    p.add_argument("directory")
    p.add_argument("--stockfish", required=True)
    p.add_argument("--nodes", type=int, default=10_000, help="node limit per position")
    p.add_argument("--stride", type=int, default=4, help="label one in this many plies")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--splits", nargs="+", default=list(data.SPLITS))
    args = p.parse_args(argv)

    directory = Path(args.directory)
    manifest = data.read_manifest(directory)
    kept = [m for m in manifest["labels"] if m["split"] not in args.splits]
    labels = list(kept)
    limit = chess.engine.Limit(nodes=args.nodes)
    with contextlib.ExitStack() as stack:
        engines = []
        for _ in range(args.workers):
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            stack.callback(engine.quit)
            engine.configure({"Threads": 1, "Hash": 16})
            engines.append(engine)
        engine_name = engines[0].id["name"]
        for entry in manifest["shards"]:
            if entry["split"] not in args.splits:
                continue
            shard = data.load_shard(directory / entry["name"])
            values = label_shard(shard, engines, limit, args.stride)
            name = entry["name"].replace(".safetensors", ".labels.safetensors")
            repro.save_tensors(
                directory / name,
                {"value": values},
                {"format": "pgx-rl-labels-v1", "shard_sha256": entry["sha256"]},
            )
            labeled = int((~np.isnan(values)).sum())
            labels.append(
                {
                    "name": name,
                    "split": entry["split"],
                    "shard": entry["name"],
                    "shard_sha256": entry["sha256"],
                    "labeled": labeled,
                    "sha256": repro.sha256_file(directory / name),
                    "stockfish": engine_name,
                    "nodes": args.nodes,
                    "stride": args.stride,
                }
            )
            print(f"{name}: {labeled}/{len(values)} positions", flush=True)
    manifest["labels"] = sorted(labels, key=lambda m: m["name"])
    repro.write_json(directory / "manifest.json", data.seal(manifest))
    print(f"dataset_id {manifest['dataset_id']}")


if __name__ == "__main__":
    main()
