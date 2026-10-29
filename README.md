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

Status: self-play loop (milestone 0), evaluation harness and benchmark (milestone 1) and the data and reproducibility tooling (milestone 2) exist. Stage 1 training (milestone 3) is next.

## Chess (AlphaZero-style)

Self-play with Gumbel MuZero search ([mctx](https://github.com/google-deepmind/mctx)) over the real pgx chess dynamics, training a residual CNN policy/value network. Periodically evaluates against a random mover and saves parameters to `chess_params.safetensors`.

```bash
uv run pgx-rl-chess --iterations 100 --selfplay-batch 128 --simulations 32
```

Games longer than `--max-steps` plies are scored as draws. Real strength needs a GPU/TPU and many iterations; on CPU use smaller settings (see `--help`).

## Evaluation and benchmarks

```bash
brew install stockfish   # or any Stockfish binary; pass --stockfish PATH
uv run pgx-rl-eval --params chess_params.safetensors --simulations 32 --sf-elo 1350 --pairs 16
uv run pgx-rl-bench      # per-machine throughput of net, pgx step, search, self-play, update
```

`pgx-rl-eval` reports W/D/L against Stockfish and an Elo difference with a 95% interval. Without `--params` it evaluates a random net.

## Data and reproducibility

```bash
# Lichess dumps: https://database.lichess.org/ (CC0)
uv run pgx-rl-data build lichess_db_standard_rated_2024-01.pgn.zst --out data/lichess-2024-01
uv run pgx-rl-label data/lichess-2024-01 --stockfish $(which stockfish) --workers 4
uv run pgx-rl-data verify data/lichess-2024-01 --pgn lichess_db_standard_rated_2024-01.pgn.zst
uv run pgx-rl-hub push-dataset data/lichess-2024-01 --repo USER/chess-lichess-2024-01   # private unless --public
uv run pgx-rl-hub push-model chess_params.safetensors --repo USER/chess-net
```

Checkpoints are safetensors with a `.json` run record (seed, arguments, dataset id, library versions, git commit). See "Data and reproducibility" in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
