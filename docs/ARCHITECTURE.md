# Chess architecture

**Goal:** maximize playing strength of a chess engine whose training and inference cost nothing. It must be trainable on a MacBook Pro / Mac mini and use only free cloud tiers. A secondary goal is to learn modern techniques, so some stages are research baselines rather than the strength path.

## Decisions

| Candidate | Decision | Why |
|---|---|---|
| Policy+value net with Gumbel MuZero search (mctx) | **Core** | Works with 16-64 simulations per move, which is what the budget allows. Already in `chess.py`. |
| Supervised pretraining on Lichess + Stockfish labels | **Core** | Largest strength gain per unit of compute. Avoids millions of from-scratch self-play games. |
| Self-play RL fine-tuning | **Core** | Pushes past the teacher once the net is competent. |
| Decision Transformer | **Baseline (secondary goal)** | Return-conditioned sequence model on the same data. Expected to lose to the searched net; kept for comparison. |
| Diffusion world model | **Rejected** | pgx is an exact simulator. A learned model only adds error. |
| GRPO | **Rejected** | Critic-free methods target LLMs. A value head is cheap on a board game and search targets are stronger. |

## System

```
Lichess PGN dump (CC0) ──► data pipeline ──► shards (obs, action, outcome, sf_value)
                                                   │
                                  Stage 1: supervised pretrain (policy CE + value MSE)
                                                   │
                                           pretrained params
                                                   │
                  Stage 2: Gumbel AlphaZero self-play ◄──► replay buffer ──► update
                                                   │
                                            final params
                                                   │
                     Inference: mctx search ─► UCI wrapper ─► Lichess bot / local GUI / ONNX demo
```

### Network
- Input: pgx chess observation (8x8x119, AlphaZero-style with history). Output: 4672-way policy logits (pgx action encoding) and a scalar value in [-1, 1].
- Start from the current residual CNN (`Net`) scaled to roughly 6-10 blocks at 128 channels (about 3-10M parameters). Treat a square-token transformer as an ablation, not the default.
- Size is limited by laptop inference + self-play throughput, not by training memory. Scale only after profiling.

### Stage 1: supervised pretraining
- Data: Lichess open database, filtered to both players at or above roughly 2000 Elo, no abandoned games.
- Policy target: the move played. Value target: a blend of game outcome and Stockfish eval (win-probability mapped to [-1, 1]) on a labeled subset. Stockfish runs on CPU (Mac mini is the labeling box).
- Exit criterion: policy top-1 accuracy and value MSE on a held-out set, plus strength vs the stage-0 baselines below.

### Stage 2: self-play RL
- Reuse `make_search` / `make_selfplay` / `make_update` from `chess.py`, initialized from stage-1 params.
- Needed additions: a replay buffer (the current loop trains only on the latest batch), a resign/adjudication rule instead of fixed-length draws, and checkpoint gating (a new net must beat the previous one before it generates data).
- Keep a fraction of supervised batches mixed in to prevent forgetting.

### Inference
- Search with 50-200 simulations. Expose through a UCI wrapper so any GUI or the Lichess bot API (free) can drive it.
- Optional: export the net to ONNX for a static browser demo on Cloudflare Pages.

## Evaluation (must exist before training claims)
- Baselines: random mover, then Stockfish at fixed low skill levels / depth limits. Report Elo estimates from match results with confidence intervals, not win-rate alone.
- Fixed-seed suites so runs are comparable; log net-only and net+search strength separately.
- Secondary-goal comparison: Decision Transformer vs policy/value net, with and without search.

## Compute plan
- Free cloud: Kaggle (weekly GPU/TPU quota; JAX runs natively on TPU) and Colab free tier for stage 1 and heavy self-play. Laptops for development, Stockfish labeling, and inference.
- Expected ceiling is an estimate, not a measurement: from-scratch self-play likely stalls at club level; pretraining plus search plausibly reaches strong club to low expert.

## Risks and open spikes
1. **JAX on Apple Metal** is experimental and mctx's control flow may not run well on it. Benchmark CPU vs Metal before choosing local defaults; MLX is a fallback for the net only.
2. **PGN to pgx conversion.** The 119-plane observation includes move history, so games must be replayed through pgx (or a faithful encoder) rather than converted position by position. Verify pgx's action encoding against python-chess with a round-trip test over many games.
3. **pgx to FEN for Stockfish labels.** Confirm pgx can produce or accept FEN; otherwise label from python-chess positions generated during the same replay.
4. **Python 3.14 + JAX/pgx/mctx compatibility** is pinned in `pyproject.toml`; any dependency added for data work (python-chess, zstandard) must be checked against it.

## Milestones
1. Evaluation harness (Stockfish matches, Elo estimate) and benchmarks of current self-play throughput on each machine.
2. Data pipeline: Lichess dump to sharded training data, with round-trip tests.
3. Stage 1 training + metrics; first strength number.
4. Stage 2 self-play with replay buffer and gating.
5. UCI wrapper and Lichess bot.
6. Decision Transformer baseline and comparison write-up.
