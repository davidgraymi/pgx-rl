import numpy as np

from pgx_rl import data, pretrain, repro
from tests.test_data import build, pgn  # noqa: F401  (pytest fixture)


def test_replayed_masks_contain_the_played_move(pgn, tmp_path):  # noqa: F811
    build(pgn, tmp_path / "ds", "--val-permille", "0", "--test-permille", "0")
    _, games = pretrain.load_split(tmp_path / "ds", "train")
    rng = np.random.default_rng(0)
    (pos,) = pretrain.positions(games, len(games), 64, rng, shuffle=False)
    assert pos["obs"].shape[1:] == (8, 8, 119)
    assert pos["mask"][np.arange(len(pos["action"])), pos["action"]].all()
    assert len(pos["action"]) == sum(min(len(g["actions"]), 64) for g in games)
    # outcome is from the mover's view: ply 0 is white's
    assert pos["outcome"][0] == games[0]["result"]


def test_pretrain_end_to_end_is_reproducible(pgn, tmp_path):  # noqa: F811
    build(pgn, tmp_path / "ds", "--val-permille", "500", "--test-permille", "0")
    hashes = []
    for name in "ab":
        pretrain.main(
            [str(tmp_path / "ds"), "--channels", "8", "--blocks", "1", "--chunk", "4"]
            + [
                "--eval-every",
                "3",
                "--eval-games",
                "2",
                "--patience",
                "1",
                "--minibatch",
                "16",
                "--max-plies",
                "64",
                "--save",
                str(tmp_path / name),
            ]
            + ["--record", str(tmp_path / f"{name}.json")]
        )
        hashes.append(repro.sha256_file(tmp_path / name))
    assert hashes[0] == hashes[1]
    record = repro.json.loads((tmp_path / "a.json").read_text())
    assert record["dataset_id"] == data.read_manifest(tmp_path / "ds")["dataset_id"]
    assert 0 <= record["val"]["top1"] <= 1
