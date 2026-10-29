# AGENTS.md

Guidance for AI coding agents working in this repo.

## Project goal
Build the strongest chess engine possible that is **free to train and run**: MacBook Pro / Mac mini plus free cloud tiers only (Kaggle, Colab). Secondary goal: learn modern RL/sequence-model techniques via baselines. The plan and decisions live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); read it before making design changes and update it when a decision changes.

## Hard constraints
- No paid services, APIs, or compute. Do not add dependencies that need a paid key or account.
- Everything must run on a laptop at small scale (small configs via CLI flags); larger runs may use free cloud notebooks.
- Rejected approaches (do not reintroduce without discussion): diffusion world models, GRPO. Decision Transformer is a baseline only, not the main path.

## Stack and commands
Python >=3.14, managed with `uv`. JAX, Flax, Optax, pgx, mctx.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pgx-rl-chess --iterations 2 --selfplay-batch 8 --simulations 4 --max-steps 20 --minibatch 32 --eval-batch 8   # smoke test
```

## Layout
- `src/pgx_rl/train.py`: tic-tac-toe actor-critic (original example).
- `src/pgx_rl/chess.py`: network, Gumbel MuZero self-play, update, eval, CLI.
- `src/pgx_rl/moves.py`: python-chess <-> pgx move conversion.
- `src/pgx_rl/elo.py`, `evaluate.py`: Elo estimates and the Stockfish match harness (`pgx-rl-eval`).
- `src/pgx_rl/bench.py`: throughput benchmark (`pgx-rl-bench`).
- `tests/`: pytest; keep tests tiny (few simulations, small batches) so they run on CPU in seconds.
- `docs/ARCHITECTURE.md`: design and milestones.

## Devices
Default to CPU (`jax>=0.12`). jax-mps is faster for net-heavy work but requires `jax<0.12` and is slower for pgx/mctx; see "Measured performance" in the architecture doc before switching. Re-measure rather than assume when changing net size or batch size.

## Conventions
- Keep environments, rollouts, and updates jitted and batched; avoid Python loops over games or plies.
- Match the surrounding style: terse functions, minimal comments, `make_*` factories that close over the model. Run `ruff` before finishing.
- New training entry points get a `pgx-rl-*` script in `pyproject.toml` and CLI flags with small defaults.
- Add tests for any new data conversion (PGN/UCI to pgx actions, FEN handling) as round-trip checks against python-chess.

## Verification
- Never claim strength improvements without an evaluation result (match vs a fixed opponent, with sample size). Win-rate vs the random mover is only a sanity check.
- Report what was run and what was not. If a change was not executed (for example, no GPU available), say so.

## Do not
- Commit data dumps, checkpoints (`*.pkl`), or Stockfish binaries; keep them gitignored.
- Run long training jobs in the foreground; run them in the background and log to a file.
