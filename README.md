# pgx-rl

Reinforcement learning on [pgx](https://github.com/sotetsuk/pgx) game environments with [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax) and [Optax](https://github.com/google-deepmind/optax). Environments, rollouts and updates are all jitted and batched on-device.

Currently: an actor-critic agent learning tic-tac-toe against a random opponent.

## Usage

```bash
uv sync
uv run pgx-rl --iterations 300
uv run pytest
```

## Roadmap

The target is the strongest chess engine that is free to train and run on a laptop plus free cloud tiers. The plan: supervised pretraining on the Lichess open database with Stockfish-labeled values, then Gumbel AlphaZero self-play fine-tuning, served through a UCI wrapper. A Decision Transformer baseline is included for comparison; diffusion world models and GRPO were evaluated and rejected. Details, risks and milestones are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); agent guidance is in [AGENTS.md](AGENTS.md).

Status: only the self-play loop below exists so far (milestone 0).

## Chess (AlphaZero-style)

Self-play with Gumbel MuZero search ([mctx](https://github.com/google-deepmind/mctx)) over the real pgx chess dynamics, training a residual CNN policy/value network. Periodically evaluates against a random mover and saves parameters to `chess_params.pkl`.

```bash
uv run pgx-rl-chess --iterations 100 --selfplay-batch 128 --simulations 32
```

Games longer than `--max-steps` plies are scored as draws. Real strength needs a GPU/TPU and many iterations; on CPU use smaller settings (see `--help`).
