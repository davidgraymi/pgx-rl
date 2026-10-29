# pgx-rl

Reinforcement learning on [pgx](https://github.com/sotetsuk/pgx) game environments with [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax) and [Optax](https://github.com/google-deepmind/optax). Environments, rollouts and updates are all jitted and batched on-device.

Currently: an actor-critic agent learning tic-tac-toe against a random opponent.

## Usage

```bash
uv sync
uv run pgx-rl --iterations 300
uv run pytest
```
