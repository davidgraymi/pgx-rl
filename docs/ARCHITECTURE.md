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

## Evaluation harness (milestone 1)
- `pgx-rl-eval` plays the net (with `--simulations N` search, or `0` for the raw policy) against Stockfish. Openings are random legal lines (`--opening-plies`), each played with both colors, so `--pairs P` gives `2P` games. Stockfish is limited by `--sf-elo` (UCI_Elo) or `--sf-skill`, plus a per-move `--sf-nodes` limit.
- pgx is the arbiter of game end; python-chess only supplies Stockfish's board. `moves.py` converts between the two (tested against pgx's legal-action mask over random games, plus castling, en passant and underpromotion). Games unfinished after `--max-plies` count as draws and are reported as truncated, which flatters weak engines, so check that count.
- Elo comes from the match score with a normal-approximation 95% interval (`elo.py`). The absolute figure is on Stockfish's UCI_Elo scale, not Lichess or FIDE; use it to compare our own checkpoints, not as a rating claim. Intervals are wide below about 100 games.
- Cost: the agent searches all games every ply and keeps only its own moves, so about half the search work is discarded.

## Compute plan
- Free cloud: Kaggle (weekly GPU/TPU quota; JAX runs natively on TPU) and Colab free tier for stage 1 and heavy self-play. Laptops for development, Stockfish labeling, and inference.
- Expected ceiling is an estimate, not a measurement: from-scratch self-play likely stalls at club level; pretraining plus search plausibly reaches strong club to low expert.

## Measured performance (Apple M3, one run each, batch 128, 16 simulations)
Backend: [jax-mps](https://github.com/tillahoffmann/jax-mps) 0.11.1 (needs `jax<0.12`; the project currently pins `jax>=0.12`, so it was tested in a separate venv with jax 0.11.2).

| Component | CPU | MPS |
|---|---|---|
| Net forward, 64ch x 4 blocks | 19 ms | 6 ms |
| Net forward, 128ch x 8 blocks | 97 ms | 34 ms |
| pgx chess step | 4 ms | 11 ms |
| mctx tree bookkeeping (16 sims) | 153 ms | 300 ms |
| Full search, 64x4 net | 594 ms | 614 ms |
| Full search, 128x8 net | 1.94 s | 1.12 s |

- The net is 3x faster on MPS; pgx stepping and mctx tree ops are 2-3x slower. Per-search time is roughly tree + 16 x (env step + net forward), which matched measurements.
- Policy: develop and evaluate on CPU. Use MPS for stage 1 pretraining (pure net work) and for self-play once the net is about 128x8 or larger. Adopting it means resolving the `jax<0.12` pin.
- Roofline ([neural-cost](https://github.com/davidgraymi/neural-cost)) on the net forward: 64x4 reaches 69% of the memory-bandwidth bound on MPS, 128x8 only 39%, both memory-bound (GroupNorm and activations). Headroom is in fusion and bf16, not more FLOPs. neural-cost does not see pgx or mctx work, which is most of the self-play time. Its hardware peak is the GPU's, so its CPU efficiency figures are not meaningful.

## Risks and open spikes
1. **JAX on Apple Metal**: resolved to a hybrid policy above. Open: why pgx step and mctx tree ops are slower on MPS, and whether a CPU-search / MPS-net split can beat either alone.
2. **PGN to pgx conversion.** The 119-plane observation includes move history, so games must be replayed through pgx (or a faithful encoder) rather than converted position by position. Verify pgx's action encoding against python-chess with a round-trip test over many games.
3. **pgx to FEN for Stockfish labels.** Confirm pgx can produce or accept FEN; otherwise label from python-chess positions generated during the same replay.
4. **Python 3.14 + JAX/pgx/mctx compatibility** is pinned in `pyproject.toml`; any dependency added for data work (python-chess, zstandard) must be checked against it.

## Milestones
1. **Done:** evaluation harness (`pgx-rl-eval`: Stockfish matches, Elo estimate) and throughput benchmark (`pgx-rl-bench`). Benchmark numbers for the other machines are still to be collected.
2. Data pipeline: Lichess dump to sharded training data, with round-trip tests.
3. Stage 1 training + metrics; first strength number.
4. Stage 2 self-play with replay buffer and gating.
5. UCI wrapper and Lichess bot.
6. Decision Transformer baseline and comparison write-up.

## Measured throughput: benchmark baseline (`pgx-rl-bench`, defaults)
Apple CPU (`cpu:0`), jax 0.12.0.1, 64x4 net (9.97M parameters, mostly the 4672-way policy Dense), batch 64, 16 simulations, one run.

| Component | Time |
|---|---|
| Net forward | 14.6 ms |
| pgx step | 2.9 ms |
| Search, one ply | 363 ms |
| Self-play, 2048 positions | 14.0 s (147 positions/s) |
| Update, 8 minibatches of 256 | 30.4 s |

The update step costs about twice as much as generating its data, so measure it on MPS before assuming self-play is the bottleneck. Not yet measured on MPS or on the other machine.
