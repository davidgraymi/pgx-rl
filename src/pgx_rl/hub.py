"""Publish and re-fetch datasets and models on the Hugging Face Hub.

Pushes upload exactly what the manifest or run record pins, plus a generated card with
the ``dataset_id``, seeds and commands needed to reproduce. Repos are created private
unless ``--public``. Authentication comes from ``HF_TOKEN`` or ``hf auth login``.
``pull`` re-downloads at a pinned revision and re-verifies every hash.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from pgx_rl import data, repro

DATASET_CARD = """\
---
license: cc0-1.0
tags: [chess, lichess, reinforcement-learning, pgx]
---

# {repo}

Chess games from the [Lichess open database](https://database.lichess.org/) (CC0),
filtered and stored as [pgx](https://github.com/sotetsuk/pgx) action sequences for
[pgx-rl](https://github.com/davidgraymi/pgx-rl).

**dataset_id:** `{dataset_id}`

## Contents
{shard_table}

## Sources
{sources}

## Filters
```json
{config}
```

## Build stats
```json
{stats}
```

Built from code commit `{commit}`. Reproduce or verify:

```bash
uv run pgx-rl-data verify <dir> --pgn <source files>
uv run pgx-rl-hub pull {repo} --type dataset --revision <commit> --dest <dir>
```
"""

MODEL_CARD = """\
---
tags: [chess, reinforcement-learning, pgx, jax]
---

# {repo}

Chess policy/value network from [pgx-rl](https://github.com/davidgraymi/pgx-rl).

| | |
|---|---|
| net | `{net}` |
| params sha256 | `{params_sha256}` |
| seed | `{seed}` |
| dataset_id | `{dataset_id}` |
| code commit | `{commit}` (dirty: {dirty}) |

`{name}` is a safetensors file; `{name}.json` is the full run record (arguments,
library versions, devices). Load with `pgx_rl.repro.load_checkpoint`, which re-checks
the parameter hash.
"""


def dataset_card(manifest, repo):
    shards = manifest["shards"]
    split_rows = [
        f"- {s}: {sum(m['games'] for m in shards if m['split'] == s)} games, "
        f"{sum(m['positions'] for m in shards if m['split'] == s)} positions"
        for s in data.SPLITS
    ]
    if manifest["labels"]:
        label = manifest["labels"][0]
        split_rows.append(
            f"- Stockfish value labels: {label['stockfish']}, {label['nodes']} nodes, "
            f"1 in {label['stride']} plies"
        )
    sources = "\n".join(
        f"- `{s['name']}` sha256 `{s['sha256']}`" for s in manifest["sources"]
    )
    return DATASET_CARD.format(
        repo=repo,
        dataset_id=manifest["dataset_id"],
        shard_table="\n".join(split_rows),
        sources=sources,
        config=json.dumps(manifest["config"], indent=2),
        stats=json.dumps(manifest["stats"], indent=2),
        commit=manifest["code"]["commit"],
    )


def model_card(record, repo, name):
    run = record.get("run", {})
    git = record["environment"]["git"]
    return MODEL_CARD.format(
        repo=repo,
        net=json.dumps(record["net"], sort_keys=True),
        params_sha256=record["params_sha256"],
        seed=run.get("seed"),
        dataset_id=run.get("dataset_id"),
        commit=git["commit"],
        dirty=git["dirty"],
        name=name,
    )


def push(folder, repo, repo_type, public, dry_run):
    folder = Path(folder)
    print(
        f"{'would upload' if dry_run else 'uploading'} to {repo_type} {repo} "
        f"({'public' if public else 'private'}):"
    )
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        print(f"  {path.relative_to(folder)} ({path.stat().st_size} bytes)")
    if dry_run:
        return None
    api = HfApi()
    api.create_repo(repo, repo_type=repo_type, private=not public, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder), repo_id=repo, repo_type=repo_type
    )
    print(f"revision {commit.oid}")
    return commit.oid


def push_dataset(directory, repo, public=False, dry_run=False):
    directory = Path(directory)
    if problems := data.verify(directory):
        raise SystemExit("refusing to push, verify failed:\n" + "\n".join(problems))
    (directory / "README.md").write_text(
        dataset_card(data.read_manifest(directory), repo)
    )
    return push(directory, repo, "dataset", public, dry_run)


def push_model(checkpoint, repo, public=False, dry_run=False):
    checkpoint = Path(checkpoint)
    repro.load_checkpoint(checkpoint)  # re-checks the parameter hash before publishing
    record = json.loads(Path(f"{checkpoint}.json").read_text())
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        for src in (checkpoint, Path(f"{checkpoint}.json")):
            shutil.copy(src, staging / src.name)
        (staging / "README.md").write_text(model_card(record, repo, checkpoint.name))
        return push(staging, repo, "model", public, dry_run)


def pull(repo, repo_type, revision, dest):
    path = Path(
        snapshot_download(repo, repo_type=repo_type, revision=revision, local_dir=dest)
    )
    if repo_type == "dataset":
        if problems := data.verify(path):
            raise SystemExit("verify failed:\n" + "\n".join(problems))
        print(f"verified dataset_id {data.read_manifest(path)['dataset_id']}")
    else:
        for file in path.glob("*.safetensors"):
            _, net = repro.load_checkpoint(file)
            print(f"verified {file.name} net {net}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Push/pull datasets and models on the HF Hub."
    )
    sub = p.add_subparsers(dest="command", required=True)
    for kind, arg in (("push-dataset", "directory"), ("push-model", "checkpoint")):
        s = sub.add_parser(kind)
        s.add_argument(arg)
        s.add_argument("--repo", required=True, help="user/name")
        s.add_argument(
            "--public", action="store_true", help="default is a private repo"
        )
        s.add_argument(
            "--dry-run", action="store_true", help="stage and list files, no network"
        )
    s = sub.add_parser("pull")
    s.add_argument("repo")
    s.add_argument("--type", choices=["dataset", "model"], default="dataset")
    s.add_argument("--revision", required=True, help="commit sha printed by push")
    s.add_argument("--dest", required=True)
    args = p.parse_args(argv)

    if args.command == "push-dataset":
        push_dataset(args.directory, args.repo, args.public, args.dry_run)
    elif args.command == "push-model":
        push_model(args.checkpoint, args.repo, args.public, args.dry_run)
    else:
        pull(args.repo, args.type, args.revision, args.dest)


if __name__ == "__main__":
    main()
