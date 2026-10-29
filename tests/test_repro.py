import jax
import numpy as np
import pytest

from pgx_rl import chess as zero
from pgx_rl import repro


@pytest.fixture(scope="module")
def params():
    model = zero.Net(channels=8, blocks=1)
    key = jax.random.PRNGKey(0)
    obs = jax.vmap(zero.env.init)(jax.random.split(key, 1)).observation
    return model.init(key, obs)


def test_seed_everything_is_deterministic():
    assert (repro.seed_everything(7) == repro.seed_everything(7)).all()
    first = np.random.rand()
    repro.seed_everything(7)
    assert np.random.rand() == first


def test_checkpoint_round_trip(tmp_path, params):
    path = tmp_path / "net.safetensors"
    repro.save_checkpoint(
        path,
        params,
        net={"channels": 8, "blocks": 1},
        record={"run": {"seed": 3, "dataset_id": "abc"}},
    )
    loaded, net = repro.load_checkpoint(path)
    assert net == {"channels": 8, "blocks": 1}
    assert repro.hash_params(loaded) == repro.hash_params(params)
    out = zero.Net(**net).apply(loaded, np.zeros((1, 8, 8, 119), np.float32))
    ref = zero.Net(**net).apply(params, np.zeros((1, 8, 8, 119), np.float32))
    assert np.allclose(out[0], ref[0])

    record = repro.json.loads((tmp_path / "net.safetensors.json").read_text())
    assert record["run"] == {"seed": 3, "dataset_id": "abc"}
    assert record["params_sha256"] == repro.hash_params(params)
    assert record["file_sha256"] == repro.sha256_file(path)
    assert "jax" in record["environment"]["packages"]


def test_checkpoint_is_byte_deterministic(tmp_path, params):
    for name in "ab":
        repro.save_checkpoint(tmp_path / name, params, net={"channels": 8, "blocks": 1})
    assert repro.sha256_file(tmp_path / "a") == repro.sha256_file(tmp_path / "b")


def test_load_rejects_wrong_hash(tmp_path, params):
    path = tmp_path / "net.safetensors"
    repro.save_checkpoint(path, params, net={"channels": 8, "blocks": 1})
    flat = repro.load_file(str(path))
    key = min(flat)
    flat[key] = flat[key] + 1
    repro.save_tensors(
        path,
        flat,
        {
            "format": repro.CHECKPOINT_FORMAT,
            "net": {},
            "params_sha256": repro.hash_params(params),
        },
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        repro.load_checkpoint(path)
