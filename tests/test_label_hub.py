import shutil

import chess
import chess.engine
import jax
import numpy as np
import pytest

from pgx_rl import chess as zero
from pgx_rl import data, hub, label, repro
from tests.test_data import build, pgn  # noqa: F401  (pytest fixture)

stockfish = shutil.which("stockfish")


def test_value_from_score():
    cp = chess.engine.Cp
    assert label.value_from_score(cp(0)) == 0
    assert 0.17 < label.value_from_score(cp(100)) < 0.19
    assert label.value_from_score(cp(-100)) == -label.value_from_score(cp(100))
    assert label.value_from_score(chess.engine.Mate(3)) == 1.0
    assert label.value_from_score(chess.engine.Mate(-2)) == -1.0


@pytest.mark.skipif(stockfish is None, reason="stockfish not installed")
def test_label_updates_manifest(pgn, tmp_path):  # noqa: F811
    ds = tmp_path / "ds"
    before = build(pgn, ds, "--val-permille", "0", "--test-permille", "0")
    label.main(
        [
            str(ds),
            "--stockfish",
            stockfish,
            "--nodes",
            "20",
            "--stride",
            "5",
            "--workers",
            "1",
        ]
    )
    after = data.read_manifest(ds)
    assert after["dataset_id"] != before["dataset_id"]
    assert data.verify(ds) == []

    entry = after["labels"][0]
    values = repro.load_file(str(ds / entry["name"]))["value"]
    shard = data.load_shard(ds / entry["shard"])
    assert values.shape == shard["actions"].shape
    labeled = ~np.isnan(values)
    assert labeled.sum() == entry["labeled"] > 0
    assert np.abs(values[labeled]).max() <= 1
    # one in `stride` plies per game, so each game of 30 plies has 6 labels
    assert labeled.sum() == sum(len(g) // 5 for g in data.game_slices(shard))


def test_hub_dry_run_stages_cards(pgn, tmp_path, capsys):  # noqa: F811
    ds = tmp_path / "ds"
    manifest = build(pgn, ds)
    assert hub.push_dataset(ds, "me/chess-data", dry_run=True) is None
    card = (ds / "README.md").read_text()
    assert manifest["dataset_id"] in card and "cc0-1.0" in card

    model = zero.Net(channels=8, blocks=1)
    key = jax.random.PRNGKey(0)
    params = model.init(
        key, jax.vmap(zero.env.init)(jax.random.split(key, 1)).observation
    )
    ckpt = tmp_path / "net.safetensors"
    repro.save_checkpoint(
        ckpt,
        params,
        net={"channels": 8, "blocks": 1},
        record={"run": {"seed": 5, "dataset_id": manifest["dataset_id"]}},
    )
    assert hub.push_model(ckpt, "me/chess-net", dry_run=True) is None
    out = capsys.readouterr().out
    assert (
        "net.safetensors.json" in out and "README.md" in out and "would upload" in out
    )


def test_push_refuses_unverified_dataset(pgn, tmp_path):  # noqa: F811
    ds = tmp_path / "ds"
    manifest = build(pgn, ds)
    shard = ds / manifest["shards"][0]["name"]
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="verify failed"):
        hub.push_dataset(ds, "me/chess-data", dry_run=True)
