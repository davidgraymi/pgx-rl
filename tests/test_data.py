import random

import chess
import chess.pgn
import pytest

from pgx_rl import data, moves, repro


def pgn_game(gid, san_or_moves, result="1-0", white=2100, black=2200, **headers):
    game = chess.pgn.Game()
    game.headers.update(
        Event="Rated Blitz game",
        Site=f"https://lichess.org/{gid}",
        Result=result,
        WhiteElo=str(white),
        BlackElo=str(black),
        TimeControl="300+0",
        Termination="Normal",
    )
    game.headers.update(headers)
    node = game
    for move in san_or_moves:
        node = node.add_variation(move)
    return (
        str(
            game,
        )
        + "\n\n"
    )


def random_moves(seed, plies=30):
    rng, board, line = random.Random(seed), chess.Board(), []
    while len(line) < plies and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        line.append(move)
        board.push(move)
    return line


REPETITION = [chess.Move.from_uci(u) for u in ["g1f3", "g8f6", "f3g1", "f6g8"] * 3] + [
    chess.Move.from_uci(u) for u in ["e2e4", "e7e5", "d2d4", "e5d4"]
]


@pytest.fixture
def pgn(tmp_path):
    text = ""
    for i in range(8):
        text += pgn_game(
            f"good{i:04d}", random_moves(i), result=["1-0", "0-1", "1/2-1/2"][i % 3]
        )
    text += pgn_game("lowelo01", random_moves(100), white=1500)
    text += pgn_game("timeout1", random_moves(101), Termination="Time forfeit")
    text += pgn_game("noresult", random_moves(102), result="*")
    text += pgn_game("tooshort", random_moves(103, plies=4))
    text += pgn_game("repeat01", REPETITION)
    path = tmp_path / "games.pgn"
    path.write_text(text)
    return path


def build(pgn, out, *extra):
    data.main(["build", str(pgn), "--out", str(out), "--min-base-seconds", "0", *extra])
    return data.read_manifest(out)


def test_filters_and_round_trip(pgn, tmp_path):
    manifest = build(
        pgn, tmp_path / "ds", "--val-permille", "0", "--test-permille", "0"
    )
    stats = manifest["stats"]
    assert stats["seen"] == 13
    assert stats["drop_elo"] == 1 and stats["drop_termination"] == 1
    assert stats["drop_result"] == 1 and stats["drop_length"] == 1
    assert stats["drop_pgx_replay"] == 1  # pgx ends the game at threefold repetition
    assert stats["kept_train"] == 8

    shard = data.load_shard(tmp_path / "ds" / manifest["shards"][0]["name"])
    games = data.game_slices(shard)
    assert len(games) == 8
    for i, actions in enumerate(games):
        board = chess.Board()
        for action, expected in zip(actions, random_moves(i)):
            move = moves.action_to_move(int(action), board)
            assert move == expected
            board.push(move)
        assert len(actions) == len(random_moves(i))
    assert shard["result"].tolist() == [1, -1, 0, 1, -1, 0, 1, -1]
    assert bytes(shard["game_id"][0]).decode() == "good0000"
    assert shard["elo"][0].tolist() == [2100, 2200]


def test_build_is_deterministic_and_verifiable(pgn, tmp_path):
    a = build(pgn, tmp_path / "a")
    b = build(pgn, tmp_path / "b")
    assert a["dataset_id"] == b["dataset_id"]
    assert [s["sha256"] for s in a["shards"]] == [s["sha256"] for s in b["shards"]]
    assert data.verify(tmp_path / "a", [pgn]) == []


def test_verify_detects_tampering(pgn, tmp_path):
    manifest = build(pgn, tmp_path / "ds")
    assert data.verify(tmp_path / "ds", [pgn]) == []
    pgn.write_text(pgn.read_text() + "\n")
    assert data.verify(tmp_path / "ds", [pgn]) == ["source mismatch games.pgn"]
    shard = tmp_path / "ds" / manifest["shards"][0]["name"]
    shard.write_bytes(shard.read_bytes()[:-1] + b"\x01")
    assert any("hash mismatch" in p for p in data.verify(tmp_path / "ds"))


def test_manifest_names_cannot_escape_directory(pgn, tmp_path):
    build(pgn, tmp_path / "ds")
    manifest = data.read_manifest(tmp_path / "ds")
    manifest["shards"][0]["name"] = "../evil.safetensors"
    repro.write_json(tmp_path / "ds" / "manifest.json", data.seal(manifest))
    problems = data.verify(tmp_path / "ds")
    assert any("unexpected file name" in p for p in problems)
    with pytest.raises(ValueError):
        data.shard_path(tmp_path, "sub/train-00000.safetensors")
    with pytest.raises(ValueError):
        data.shard_path(tmp_path, "train-00000.safetensors/../../x")
    assert data.shard_path(tmp_path, "val-00003.labels.safetensors").parent == tmp_path


def test_split_depends_only_on_game_id():
    ids = [f"game{i:04d}".encode() for i in range(2000)]
    splits = [data.split_of(g, 10, 10) for g in ids]
    assert splits == [data.split_of(g, 10, 10) for g in ids]
    assert 5 < splits.count("val") < 40 and 5 < splits.count("test") < 40


def test_sampling_is_a_deterministic_subset(pgn, tmp_path):
    full = build(pgn, tmp_path / "full")
    half = build(pgn, tmp_path / "half", "--sample-permille", "500")
    again = build(pgn, tmp_path / "again", "--sample-permille", "500")
    assert half["dataset_id"] == again["dataset_id"] != full["dataset_id"]
    kept = lambda m: sum(s["games"] for s in m["shards"])
    assert 0 < kept(half) < kept(full)
    assert half["stats"]["drop_sample"] > 0
