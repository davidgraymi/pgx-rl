"""Lichess PGN -> sharded, hash-pinned training data.

A shard stores whole games as pgx action sequences, not observations: the 119-plane
observation includes move history, so training replays games through pgx (``replay``).
Games are only written after pgx accepts every move, so a dataset never contains a
position that pgx and python-chess disagree about.

Shards are safetensors (byte-deterministic, no pickle). ``manifest.json`` records the
source files, filters, code commit and every shard's SHA-256; ``dataset_id`` is the
digest of everything that determines the data, and is what training runs cite.
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import zstandard
from safetensors.numpy import load_file

from pgx_rl import chess as zero
from pgx_rl import moves, repro

FORMAT = "pgx-rl-shard-v1"
SPLITS = ("train", "val", "test")
MAX_PLIES = 512  # pgx truncates chess at 512 plies
ID_BYTES = 8  # Lichess game ids
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}  # white's perspective
HEADER = re.compile(r'^\[(\w+) "(.*)"\]$')
SHARD_NAME = re.compile(r"(train|val|test)-\d{5}(\.labels)?\.safetensors")

step = jax.jit(jax.vmap(zero.env.step))
init = jax.jit(jax.vmap(zero.env.init))


@contextlib.contextmanager
def open_text(path):
    with open(path, "rb") as f:
        raw = (
            zstandard.ZstdDecompressor().stream_reader(f)
            if str(path).endswith(".zst")
            else f
        )
        yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def split_games(lines):
    """Yield ``(headers, text)`` for each game in a PGN line stream."""
    headers, buf = {}, []
    for line in lines:
        if line.startswith("[Event ") and buf:
            yield headers, "".join(buf)
            headers, buf = {}, []
        buf.append(line)
        if m := HEADER.match(line.rstrip("\n")):
            headers[m[1]] = m[2]
    if buf:
        yield headers, "".join(buf)


def reject_reason(h, min_elo, min_base_seconds):
    """Cheap header-only filter; returns a reason string or None to keep."""
    if h.get("Variant", "Standard") != "Standard" or "FEN" in h or "SetUp" in h:
        return "nonstandard"
    if h.get("Termination") != "Normal":
        return "termination"
    if h.get("Result") not in RESULTS:
        return "result"
    try:
        elos = int(h["WhiteElo"]), int(h["BlackElo"])
    except KeyError, ValueError:
        return "elo_missing"
    if min(elos) < min_elo:
        return "elo"
    if min_base_seconds:
        try:
            base = int(h["TimeControl"].split("+")[0])
        except KeyError, ValueError:
            return "time_control"
        if base < min_base_seconds:
            return "time_control"
    return None


def game_actions(text):
    """Parse movetext into pgx actions, or None if python-chess cannot read it."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors:
        return None
    board, actions = game.board(), []
    for move in game.mainline_moves():
        actions.append(moves.move_to_action(move, board.turn))
        board.push(move)
    return np.asarray(actions, dtype=np.int16)


def game_id(headers):
    site = headers.get("Site", "")
    return site.rsplit("/", 1)[-1].encode()[:ID_BYTES].ljust(ID_BYTES, b"\0")


def split_of(gid: bytes, val_permille: int, test_permille: int) -> str:
    """Deterministic split from the game id alone (independent of order and seed)."""
    bucket = int.from_bytes(hashlib.sha256(gid).digest()[:4], "big") % 1000
    if bucket < val_permille:
        return "val"
    if bucket < val_permille + test_permille:
        return "test"
    return "train"


@jax.jit
def _replay(actions, lengths):
    batch, steps = actions.shape
    state = init(jax.random.split(jax.random.PRNGKey(0), batch))

    def body(carry, t):
        state, ok = carry
        a = actions[:, t].astype(jnp.int32)
        active = t < lengths
        legal = ~state.terminated & state.legal_action_mask[jnp.arange(batch), a]
        new = step(state, jnp.where(active, a, 0))
        state = jax.tree.map(
            lambda n, o: jnp.where(active.reshape(-1, *[1] * (n.ndim - 1)), n, o),
            new,
            state,
        )
        return (state, ok & (legal | ~active)), None

    (state, ok), _ = jax.lax.scan(
        body, (state, jnp.ones(batch, dtype=bool)), jnp.arange(steps)
    )
    return ok, state


def replay(games, batch=256):
    """Replay action sequences through pgx; returns a bool array of fully-legal games."""
    ok = np.zeros(len(games), dtype=bool)
    for lo in range(0, len(games), batch):
        chunk = games[lo : lo + batch]
        lengths = np.array([len(g) for g in chunk])
        steps = -(-int(lengths.max()) // 64) * 64
        padded = np.zeros((batch, steps), dtype=np.int16)
        for i, g in enumerate(chunk):
            padded[i, : len(g)] = g
        lens = np.zeros(batch, dtype=np.int32)
        lens[: len(chunk)] = lengths
        ok[lo : lo + len(chunk)] = np.asarray(_replay(padded, lens)[0])[: len(chunk)]
    return ok


def write_shard(path, games):
    """``games``: list of dicts with actions, result, elo, gid. Returns (sha256, positions)."""
    lengths = np.array([len(g["actions"]) for g in games], dtype=np.int64)
    repro.save_tensors(
        path,
        {
            "actions": np.concatenate([g["actions"] for g in games]),
            "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            "result": np.array([g["result"] for g in games], dtype=np.int8),
            "elo": np.array([g["elo"] for g in games], dtype=np.int16),
            "game_id": np.stack(
                [np.frombuffer(g["gid"], dtype=np.uint8) for g in games]
            ),
        },
        {"format": FORMAT},
    )
    return repro.sha256_file(path), int(lengths.sum())


def load_shard(path):
    return load_file(str(path))


def game_slices(shard):
    """Per-game action arrays of a loaded shard."""
    o = shard["offsets"]
    return [shard["actions"][o[i] : o[i + 1]] for i in range(len(o) - 1)]


IDENTITY = ("format", "config", "sources", "shards", "labels")


def seal(manifest):
    """Set ``dataset_id`` to the digest of everything that determines the data."""
    manifest["dataset_id"] = repro.sha256_json({k: manifest[k] for k in IDENTITY})
    return manifest


def shard_path(directory, name):
    """Path of a manifest-listed file; manifests may come from the Hub, so names are strict."""
    if not SHARD_NAME.fullmatch(name):
        raise ValueError(f"unexpected file name in manifest: {name!r}")
    return Path(directory) / name


def read_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    manifest.setdefault("labels", [])
    return manifest


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "min_elo": args.min_elo,
        "min_base_seconds": args.min_base_seconds,
        "min_plies": args.min_plies,
        "max_plies": min(args.max_plies, MAX_PLIES),
        "val_permille": args.val_permille,
        "test_permille": args.test_permille,
        "shard_games": args.shard_games,
        "max_games": args.max_games,
    }
    stats, buffers, shards, counts = Counter(), {s: [] for s in SPLITS}, [], Counter()
    accepted = 0

    def flush(split):
        games = buffers[split]
        if not games:
            return
        keep = replay([g["actions"] for g in games])
        stats["drop_pgx_replay"] += int((~keep).sum())
        games = [g for g, k in zip(games, keep) if k]
        buffers[split] = []
        if not games:
            return
        name = f"{split}-{counts[split]:05d}.safetensors"
        counts[split] += 1
        sha, positions = write_shard(out / name, games)
        shards.append(
            {
                "name": name,
                "split": split,
                "games": len(games),
                "positions": positions,
                "sha256": sha,
            }
        )
        stats[f"kept_{split}"] += len(games)

    sources = []
    for pgn in args.pgn:
        sources.append(
            {
                "name": Path(pgn).name,
                "size": Path(pgn).stat().st_size,
                "sha256": repro.sha256_file(pgn),
            }
        )
        with open_text(pgn) as lines:
            for headers, text in split_games(lines):
                if args.max_games and accepted >= args.max_games:
                    break
                stats["seen"] += 1
                if reason := reject_reason(
                    headers, args.min_elo, args.min_base_seconds
                ):
                    stats[f"drop_{reason}"] += 1
                    continue
                actions = game_actions(text)
                if actions is None:
                    stats["drop_parse"] += 1
                    continue
                if not config["min_plies"] <= len(actions) <= config["max_plies"]:
                    stats["drop_length"] += 1
                    continue
                accepted += 1
                gid = game_id(headers)
                split = split_of(gid, args.val_permille, args.test_permille)
                buffers[split].append(
                    {
                        "actions": actions,
                        "result": RESULTS[headers["Result"]],
                        "elo": (int(headers["WhiteElo"]), int(headers["BlackElo"])),
                        "gid": gid,
                    }
                )
                if len(buffers[split]) >= args.shard_games:
                    flush(split)
    for split in SPLITS:
        flush(split)

    shards.sort(key=lambda s: s["name"])
    manifest = seal(
        {
            "format": FORMAT,
            "config": config,
            "sources": sources,
            "shards": shards,
            "labels": [],
            "stats": dict(sorted(stats.items())),
            "code": repro.git_info(),
        }
    )
    repro.write_json(out / "manifest.json", manifest)
    return manifest


def verify(directory, pgn=()):
    """Re-hash shards (and optionally source files) against the manifest; returns problems."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    problems = []
    if seal(dict(manifest))["dataset_id"] != manifest["dataset_id"]:
        problems.append("dataset_id does not match manifest contents")
    for shard in manifest["shards"] + manifest["labels"]:
        try:
            path = shard_path(directory, shard["name"])
            if "shard" in shard:
                shard_path(directory, shard["shard"])
        except ValueError as e:
            problems.append(str(e))
            continue
        if not path.exists():
            problems.append(f"missing {shard['name']}")
        elif repro.sha256_file(path) != shard["sha256"]:
            problems.append(f"hash mismatch {shard['name']}")
    known = {s["name"]: s["sha256"] for s in manifest["sources"]}
    for p in pgn:
        if known.get(Path(p).name) != repro.sha256_file(p):
            problems.append(f"source mismatch {Path(p).name}")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description="Build and verify Lichess training shards.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="PGN(.zst) files -> shards + manifest")
    b.add_argument("pgn", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--min-elo", type=int, default=2000)
    b.add_argument(
        "--min-base-seconds", type=int, default=180, help="drop bullet; 0 keeps all"
    )
    b.add_argument("--min-plies", type=int, default=10)
    b.add_argument("--max-plies", type=int, default=MAX_PLIES)
    b.add_argument("--val-permille", type=int, default=10)
    b.add_argument("--test-permille", type=int, default=10)
    b.add_argument("--shard-games", type=int, default=50_000)
    b.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="stop after this many accepted games (before replay); 0 = all",
    )
    v = sub.add_parser("verify", help="check shard (and optionally source) hashes")
    v.add_argument("directory")
    v.add_argument("--pgn", nargs="*", default=[])
    args = p.parse_args(argv)

    if args.command == "build":
        manifest = build(args)
        print(f"dataset_id {manifest['dataset_id']}")
        for key, value in manifest["stats"].items():
            print(f"  {key:22s} {value}")
        return
    problems = verify(args.directory, args.pgn)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
