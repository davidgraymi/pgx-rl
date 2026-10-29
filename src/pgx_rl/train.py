"""Actor-critic (REINFORCE with a value baseline) on a pgx game, fully jitted in JAX.

The agent plays as player 0 against a uniformly random opponent.
"""

import argparse
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pgx

ENV_ID = "tic_tac_toe"
MAX_STEPS = 9  # upper bound on plies per episode for tic_tac_toe


class ActorCritic(nn.Module):
    num_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, obs):
        x = obs.reshape(obs.shape[0], -1).astype(jnp.float32)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.num_actions)(x), nn.Dense(1)(x).squeeze(-1)


def masked_logits(logits, mask):
    return jnp.where(mask, logits, jnp.finfo(logits.dtype).min)


def make_rollout(env, model):
    """Returns rollout(params, key, batch) -> trajectory dict, one full episode per env."""

    def step(carry, key):
        state, params = carry
        key_act, key_env = jax.random.split(key)
        is_agent = (state.current_player == 0) & ~state.terminated
        logits, _ = model.apply(params, state.observation)
        agent_logits = masked_logits(logits, state.legal_action_mask)
        random_logits = masked_logits(jnp.zeros_like(logits), state.legal_action_mask)
        logits = jnp.where(is_agent[:, None], agent_logits, random_logits)
        action = jax.random.categorical(key_act, logits)
        out = dict(
            obs=state.observation,
            mask=state.legal_action_mask,
            action=action,
            active=is_agent,
        )
        keys = jax.random.split(key_env, action.shape[0])
        state = jax.vmap(env.step)(state, action, keys)
        return (state, params), (out, state.rewards[:, 0])

    @partial(jax.jit, static_argnames="batch")
    def rollout(params, key, batch):
        key_init, key_scan = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(key_init, batch))
        _, (traj, rewards) = jax.lax.scan(
            step, (state, params), jax.random.split(key_scan, MAX_STEPS)
        )
        traj["ret"] = jnp.broadcast_to(rewards.sum(0), traj["active"].shape)  # (T, B)
        return traj

    return rollout


def make_loss(model, entropy_coef=0.01):
    def loss_fn(params, traj):
        t, b = traj["active"].shape
        obs = traj["obs"].reshape(t * b, *traj["obs"].shape[2:])
        logits, value = model.apply(params, obs)
        logits = masked_logits(logits, traj["mask"].reshape(t * b, -1))
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(
            logp_all, traj["action"].reshape(-1, 1), axis=1
        ).squeeze(-1)
        entropy = -(
            jnp.exp(logp_all) * jnp.where(jnp.isfinite(logp_all), logp_all, 0)
        ).sum(-1)
        ret = traj["ret"].reshape(-1)
        active = traj["active"].reshape(-1).astype(jnp.float32)
        adv = jax.lax.stop_gradient(ret - value)
        n = jnp.maximum(active.sum(), 1.0)
        pg = -(logp * adv * active).sum() / n
        vf = 0.5 * (((ret - value) ** 2) * active).sum() / n
        ent = (entropy * active).sum() / n
        return pg + vf - entropy_coef * ent, dict(pg=pg, vf=vf, entropy=ent)

    return loss_fn


def main():
    p = argparse.ArgumentParser(description="Train an RL agent with pgx + JAX.")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = pgx.make(ENV_ID)
    model = ActorCritic(num_actions=env.num_actions)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jax.vmap(env.init)(jax.random.split(init_key, 1)).observation
    params = model.init(init_key, dummy)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    rollout = make_rollout(env, model)
    loss_fn = make_loss(model)

    @jax.jit
    def update(params, opt_state, traj):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, traj)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    start = time.time()
    for i in range(1, args.iterations + 1):
        key, k = jax.random.split(key)
        traj = rollout(params, k, args.batch)
        params, opt_state, loss, aux = update(params, opt_state, traj)
        if i % 50 == 0 or i == 1:
            ret = traj["ret"][0]
            print(
                f"iter {i:4d} loss {loss:7.4f} entropy {aux['entropy']:.3f} "
                f"win {(ret > 0).mean():.2f} draw {(ret == 0).mean():.2f} "
                f"loss {(ret < 0).mean():.2f} ({time.time() - start:.1f}s)"
            )


if __name__ == "__main__":
    main()
