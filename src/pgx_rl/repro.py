"""Reproducibility helpers: seeds, environment records, hashing and safetensors checkpoints.

Every checkpoint is written next to a ``.json`` run record holding the seed, arguments,
code commit, library versions and the dataset id it was trained on, so a result can be
traced back to the exact data and code. JAX randomness is explicit (``PRNGKey(seed)``),
so :func:`seed_everything` only needs to cover Python and NumPy.
"""

import hashlib
import json
import platform
import random
import subprocess
from importlib import metadata

import jax
import numpy as np
from flax import traverse_util
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

CHECKPOINT_FORMAT = "pgx-rl-params-v1"
PACKAGES = ("jax", "jaxlib", "flax", "optax", "pgx", "mctx", "chess", "safetensors")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write("\n")


def git_info() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}


def environment_info() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "packages": versions,
        "git": git_info(),
    }


def save_tensors(path, tensors, meta: dict):
    """Write safetensors with one canonical-JSON metadata entry.

    safetensors serialises its metadata dict in hash-map order, so files with several
    metadata keys are not byte-reproducible; a single key is.
    """
    save_file(tensors, str(path), metadata={"meta": canonical_json(meta)})


def read_meta(path) -> dict:
    with safe_open(str(path), framework="numpy") as f:
        return json.loads((f.metadata() or {}).get("meta", "{}"))


def hash_params(params) -> str:
    """Digest of parameter names, dtypes, shapes and values (independent of file layout)."""
    digest = hashlib.sha256()
    for name, value in sorted(traverse_util.flatten_dict(params, sep="/").items()):
        value = np.ascontiguousarray(value)
        digest.update(f"{name}|{value.dtype}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_checkpoint(path, params, net: dict, record: dict | None = None):
    """Write params as safetensors plus ``<path>.json`` with the full run record."""
    flat = {
        name: np.asarray(value)
        for name, value in traverse_util.flatten_dict(
            jax.device_get(params), sep="/"
        ).items()
    }
    params_sha = hash_params(traverse_util.unflatten_dict(flat, sep="/"))
    save_tensors(
        path,
        flat,
        {"format": CHECKPOINT_FORMAT, "net": net, "params_sha256": params_sha},
    )
    write_json(
        f"{path}.json",
        {
            **(record or {}),
            "net": net,
            "params_sha256": params_sha,
            "file_sha256": sha256_file(path),
            "environment": environment_info(),
        },
    )


def load_checkpoint(path):
    """Return ``(params, net_config)``; the stored parameter hash is re-checked."""
    meta = read_meta(path)
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    params = traverse_util.unflatten_dict(load_file(str(path)), sep="/")
    if hash_params(params) != meta["params_sha256"]:
        raise ValueError(f"{path}: parameter hash mismatch, file is corrupt")
    return params, meta["net"]
