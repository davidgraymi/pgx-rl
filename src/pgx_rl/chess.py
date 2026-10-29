"""AlphaZero-style self-play training for chess with pgx, mctx and JAX.

Self-play uses Gumbel MuZero search (mctx) with the real pgx chess dynamics as the
model. The network is a small residual CNN with policy and value heads. Games that
exceed ``--max-steps`` plies are scored as draws.
"""

import argparse
import pickle
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx

env = pgx.make("chess")


class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        y = nn.relu(nn.GroupNorm(num_groups=4)(nn.Conv(self.channels, (3, 3))(x)))
        y = nn.GroupNorm(num_groups=4)(nn.Conv(self.channels, (3, 3))(y))
        return nn.relu(x + y)


class Net(nn.Module):
    channels: int = 64
    blocks: int = 4

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32)
        x = nn.relu(nn.GroupNorm(num_groups=4)(nn.Conv(self.channels, (3, 3))(x)))
        for _ in range(self.blocks):
            x = ResBlock(self.channels)(x)
        p = nn.relu(nn.Conv(32, (1, 1))(x)).reshape(x.shape[0], -1)
        logits = nn.Dense(env.num_actions)(p)
        v = nn.relu(nn.Conv(8, (1, 1))(x)).reshape(x.shape[0], -1)
        value = jnp.tanh(nn.Dense(1)(nn.relu(nn.Dense(64)(v)))).squeeze(-1)
        return logits, value


def mask_logits(logits, mask):
    logits = logits - logits.max(-1, keepdims=True)
    return jnp.where(mask, logits, jnp.finfo(logits.dtype).min)


def safe_mask(state):
    # terminated states can have no legal actions; keep search well-defined
    return jnp.where(
        state.legal_action_mask.any(-1, keepdims=True), state.legal_action_mask, True
    )


def make_search(model, num_simulations, considered):
    def recurrent_fn(params, key, action, state):
        player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = model.apply(params, state.observation)
        reward = state.rewards[jnp.arange(action.shape[0]), player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0)
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=mask_logits(logits, safe_mask(state)),
            value=value,
        )
        return out, state

    def search(params, key, state):
        logits, value = model.apply(params, state.observation)
        mask = safe_mask(state)
        root = mctx.RootFnOutput(
            prior_logits=mask_logits(logits, mask), value=value, embedding=state
        )
        return mctx.gumbel_muzero_policy(
            params,
            key,
            root,
            recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~mask,
            max_num_considered_actions=considered,
        )

    return search


def make_selfplay(model, search, max_steps):
    @partial(jax.jit, static_argnames="batch")
    def selfplay(params, key, batch):
        key_init, key_scan = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(key_init, batch))

        def step(state, key):
            out = search(params, key, state)
            valid = ~state.terminated
            rec = {
                "obs": state.observation,
                "weights": out.action_weights,
                "player": state.current_player,
                "valid": valid,
            }
            state = jax.vmap(env.step)(state, out.action)
            return state, (rec, state.rewards)

        _, (rec, rewards) = jax.lax.scan(
            step, state, jax.random.split(key_scan, max_steps)
        )
        outcome = rewards.sum(0)  # (B, 2): final reward per player, 0 if truncated
        rec["value"] = jnp.take_along_axis(
            jnp.broadcast_to(outcome, (max_steps, *outcome.shape)),
            rec["player"][..., None],
            axis=-1,
        ).squeeze(-1)
        return rec

    return selfplay


def make_update(model, opt):
    def loss_fn(params, mb):
        logits, value = model.apply(params, mb["obs"])
        w = mb["valid"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        pol = -(mb["weights"] * jax.nn.log_softmax(logits)).sum(-1)
        pol = (pol * w).sum() / n
        val = (((value - mb["value"]) ** 2) * w).sum() / n
        return pol + val, (pol, val)

    @partial(jax.jit, static_argnames="minibatch")
    def update(params, opt_state, rec, key, minibatch):
        flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), rec)
        n = (flat["valid"].shape[0] // minibatch) * minibatch
        perm = jax.random.permutation(key, flat["valid"].shape[0])[:n]
        mbs = jax.tree.map(lambda x: x[perm].reshape(-1, minibatch, *x.shape[1:]), flat)

        def body(carry, mb):
            params, opt_state = carry
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), (loss, *aux)

        (params, opt_state), metrics = jax.lax.scan(body, (params, opt_state), mbs)
        return params, opt_state, jax.tree.map(jnp.mean, metrics)

    return update


def make_eval(model, search, max_steps):
    """Network + search (as player 0 or 1 at random) vs a uniformly random mover."""

    @partial(jax.jit, static_argnames="batch")
    def evaluate(params, key, batch):
        key_init, key_side, key_scan = jax.random.split(key, 3)
        state = jax.vmap(env.init)(jax.random.split(key_init, batch))
        agent = jax.random.randint(key_side, (batch,), 0, 2)

        def step(state, key):
            k_search, k_rand = jax.random.split(key)
            a_net = search(params, k_search, state).action
            a_rand = jax.random.categorical(
                k_rand,
                mask_logits(jnp.zeros((batch, env.num_actions)), safe_mask(state)),
            )
            action = jnp.where(state.current_player == agent, a_net, a_rand)
            state = jax.vmap(env.step)(state, action)
            return state, state.rewards

        _, rewards = jax.lax.scan(step, state, jax.random.split(key_scan, max_steps))
        r = rewards.sum(0)[jnp.arange(batch), agent]
        return (r > 0).mean(), (r == 0).mean(), (r < 0).mean()

    return evaluate


def main():
    p = argparse.ArgumentParser(
        description="AlphaZero-style chess training with pgx + JAX."
    )
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--selfplay-batch", type=int, default=128)
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument(
        "--considered", type=int, default=16, help="actions considered at the root"
    )
    p.add_argument(
        "--max-steps", type=int, default=200, help="plies before a game is a draw"
    )
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--save", default="chess_params.pkl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = Net(channels=args.channels, blocks=args.blocks)
    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    params = model.init(k, jax.vmap(env.init)(jax.random.split(k, 1)).observation)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    search = make_search(model, args.simulations, args.considered)
    selfplay = make_selfplay(model, search, args.max_steps)
    update = make_update(model, opt)
    evaluate = make_eval(model, search, args.max_steps)

    start = time.time()
    for i in range(1, args.iterations + 1):
        key, k1, k2 = jax.random.split(key, 3)
        rec = selfplay(params, k1, args.selfplay_batch)
        params, opt_state, (loss, pol, val) = update(
            params, opt_state, rec, k2, args.minibatch
        )
        msg = (
            f"iter {i:4d} loss {loss:.3f} policy {pol:.3f} value {val:.3f} "
            f"decisive {(rec['value'] != 0)[0].mean():.2f} ({time.time() - start:.0f}s)"
        )
        if i % args.eval_every == 0 or i == args.iterations:
            key, k = jax.random.split(key)
            w, d, l = evaluate(params, k, args.eval_batch)
            msg += f" | vs random: win {w:.2f} draw {d:.2f} loss {l:.2f}"
            with open(args.save, "wb") as f:
                pickle.dump(jax.device_get(params), f)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
