"""Stage 1: supervised pretraining on dataset shards.

Shards store pgx actions, so each chunk of games is replayed through pgx to get the
observations and legal-action masks. Policy target is the move played; value target blends
the game outcome (mover's view) with the Stockfish label where one exists.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import load_file

from pgx_rl import chess as zero
from pgx_rl import data, repro


@partial(jax.jit, static_argnames="steps")
def replay_obs(actions, steps):
    """Observations and legal masks before each ply; actions are [batch, steps]."""
    state = data.init(jax.random.split(jax.random.PRNGKey(0), actions.shape[0]))

    def body(state, a):
        out = (state.observation, state.legal_action_mask)
        return data.step(state, a), out

    _, (obs, mask) = jax.lax.scan(body, state, actions.T.astype(jnp.int32))
    return obs.swapaxes(0, 1)[:, :steps], mask.swapaxes(0, 1)[:, :steps]


def load_split(directory, split):
    """Games of a split as dicts of numpy arrays; ``value`` is NaN where unlabeled."""
    manifest = data.read_manifest(directory)
    sidecar = {m["shard"]: m["name"] for m in manifest["labels"]}
    games = []
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        shard = data.load_shard(data.shard_path(directory, entry["name"]))
        label = sidecar.get(entry["name"])
        values = (
            load_file(str(data.shard_path(directory, label)))["value"]
            if label
            else np.full(len(shard["actions"]), np.nan, np.float16)
        )
        o = shard["offsets"]
        for i, actions in enumerate(data.game_slices(shard)):
            games.append(
                {
                    "actions": actions,
                    "result": int(shard["result"][i]),
                    "value": values[o[i] : o[i + 1]],
                }
            )
    return manifest["dataset_id"], games


def chunk_batch(games, steps):
    """Pad a chunk of games to ``steps`` plies; returns numpy arrays plus a validity mask."""
    n = len(games)
    actions = np.zeros((n, steps), np.int16)
    sf = np.full((n, steps), np.nan, np.float32)
    outcome = np.zeros((n, steps), np.float32)
    valid = np.zeros((n, steps), bool)
    for i, g in enumerate(games):
        t = len(g["actions"])
        actions[i, :t] = g["actions"]
        sf[i, :t] = g["value"]
        sign = np.where(np.arange(t) % 2 == 0, 1.0, -1.0)  # ply 0 is white to move
        outcome[i, :t] = g["result"] * sign
        valid[i, :t] = True
    return actions, sf, outcome, valid


def flat(x, keep):
    return np.asarray(x).reshape(-1, *x.shape[2:])[keep]


def positions(games, chunk, steps, rng, shuffle):
    """Yield dicts of flat position arrays, one per chunk of games."""
    order = rng.permutation(len(games)) if shuffle else np.arange(len(games))
    for lo in range(0, len(order), chunk):
        sel = [games[i] for i in order[lo : lo + chunk]]
        longest = max(len(g["actions"]) for g in sel)
        t = min(steps, -(-longest // 64) * 64)
        actions, sf, outcome, valid = chunk_batch(
            [{**g, "actions": g["actions"][:t], "value": g["value"][:t]} for g in sel],
            t,
        )
        actions = np.pad(actions, ((0, chunk - len(sel)), (0, 0)))
        obs, mask = replay_obs(actions, t)
        pad = chunk - len(sel)
        sf = np.pad(sf, ((0, pad), (0, 0)), constant_values=np.nan)
        outcome = np.pad(outcome, ((0, pad), (0, 0)))
        valid = np.pad(valid, ((0, pad), (0, 0)))
        keep = valid.reshape(-1)
        yield {
            "obs": flat(obs, keep),
            "mask": flat(mask, keep),
            "action": flat(actions[:, :t], keep).astype(np.int32),
            "sf": flat(sf, keep),
            "outcome": flat(outcome, keep),
        }


def make_step(model, opt, blend):
    def target(b):
        has = ~jnp.isnan(b["sf"])
        sf = jnp.where(has, b["sf"], 0.0)
        return jnp.where(has, blend * sf + (1 - blend) * b["outcome"], b["outcome"])

    def loss_fn(params, b):
        logits, value = model.apply(params, b["obs"])
        logp = jax.nn.log_softmax(zero.mask_logits(logits, b["mask"]))
        pol = -jnp.take_along_axis(logp, b["action"][:, None], -1).squeeze(-1)
        acc = (logp.argmax(-1) == b["action"]).mean()
        val = (value - target(b)) ** 2
        return pol.mean() + val.mean(), (pol.mean(), val.mean(), acc)

    @jax.jit
    def train(params, opt_state, b):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, (loss, *aux)

    @jax.jit
    def evaluate(params, b):
        logits, value = model.apply(params, b["obs"])
        logp = jax.nn.log_softmax(zero.mask_logits(logits, b["mask"]))
        pol = -jnp.take_along_axis(logp, b["action"][:, None], -1).squeeze(-1)
        has = ~jnp.isnan(b["sf"])
        sf_err = jnp.where(has, (value - jnp.where(has, b["sf"], 0.0)) ** 2, 0.0)
        return {
            "n": jnp.float32(len(pol)),
            "policy_ce": pol.sum(),
            "top1": (logp.argmax(-1) == b["action"]).sum(),
            "value_mse_outcome": ((value - b["outcome"]) ** 2).sum(),
            "n_labeled": has.sum(),
            "value_mse_stockfish": sf_err.sum(),
        }

    return train, evaluate


def run_eval(evaluate, params, games, chunk, steps, minibatch, rng):
    total = {}
    for pos in positions(games, chunk, steps, rng, shuffle=False):
        for lo in range(0, len(pos["action"]), minibatch):
            b = {k: v[lo : lo + minibatch] for k, v in pos.items()}
            for k, v in evaluate(params, b).items():
                total[k] = total.get(k, 0.0) + float(v)
    n, nl = max(total["n"], 1), max(total["n_labeled"], 1)
    return {
        "positions": int(total["n"]),
        "policy_ce": total["policy_ce"] / n,
        "top1": total["top1"] / n,
        "value_mse_outcome": total["value_mse_outcome"] / n,
        "labeled": int(total["n_labeled"]),
        "value_mse_stockfish": total["value_mse_stockfish"] / nl,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 1 supervised pretraining.")
    p.add_argument("directory", help="dataset directory (manifest.json and shards)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--chunk", type=int, default=32, help="games replayed at once")
    p.add_argument("--max-plies", type=int, default=256, help="plies kept per game")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--blend", type=float, default=0.5, help="weight of the Stockfish label"
    )
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--save", default="pretrained.safetensors")
    p.add_argument("--record", help="write the final validation metrics here (json)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    key = repro.seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset_id, train_games = load_split(args.directory, "train")
    _, val_games = load_split(args.directory, "val")
    if not train_games:
        raise SystemExit("no training games in the dataset")

    model = zero.Net(channels=args.channels, blocks=args.blocks)
    key, k = jax.random.split(key)
    params = model.init(k, jax.vmap(zero.env.init)(jax.random.split(k, 1)).observation)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    train, evaluate = make_step(model, opt, args.blend)
    net = {"channels": args.channels, "blocks": args.blocks}

    start, steps = time.time(), 0
    for epoch in range(1, args.epochs + 1):
        for pos in positions(train_games, args.chunk, args.max_plies, rng, True):
            perm = rng.permutation(len(pos["action"]))
            for lo in range(0, len(perm) - args.minibatch + 1, args.minibatch):
                b = {k: v[perm[lo : lo + args.minibatch]] for k, v in pos.items()}
                params, opt_state, (loss, pol, val, acc) = train(params, opt_state, b)
                steps += 1
                if steps % 50 == 0:
                    print(
                        f"epoch {epoch} step {steps} loss {loss:.3f} policy {pol:.3f} "
                        f"value {val:.3f} top1 {acc:.3f} ({time.time() - start:.0f}s)",
                        flush=True,
                    )
        metrics = (
            run_eval(
                evaluate,
                params,
                val_games,
                args.chunk,
                args.max_plies,
                args.minibatch,
                rng,
            )
            if val_games
            else {}
        )
        print(f"epoch {epoch} val {metrics}", flush=True)

    repro.save_checkpoint(
        args.save,
        params,
        net=net,
        record={
            "run": {
                "seed": args.seed,
                "args": vars(args),
                "steps": steps,
                "dataset_id": dataset_id,
                "val": metrics,
            }
        },
    )
    if args.record:
        repro.write_json(Path(args.record), {"dataset_id": dataset_id, "val": metrics})


if __name__ == "__main__":
    main()
