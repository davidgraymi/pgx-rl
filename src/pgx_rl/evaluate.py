"""Match the chess network (optionally with Gumbel MuZero search) against Stockfish.

Games run as one batch: the jitted agent proposes a move for every game each ply and
Stockfish answers for the games where it is to move. Each opening is played twice with
colors swapped. pgx decides when a game ends; games still running after ``--max-plies``
are scored as draws and counted separately.
"""

import argparse
import contextlib
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np

from pgx_rl import chess as zero
from pgx_rl import elo, moves, repro

step = jax.jit(jax.vmap(zero.env.step))
init = jax.jit(jax.vmap(zero.env.init))


def make_agent(model, simulations, considered):
    """Jitted ``act(params, key, state) -> action``; ``simulations=0`` plays the net's argmax."""
    if simulations > 0:
        search = zero.make_search(model, simulations, considered)
        return jax.jit(lambda params, key, state: search(params, key, state).action)

    @jax.jit
    def act(params, key, state):
        logits, _ = model.apply(params, state.observation)
        return zero.mask_logits(logits, zero.safe_mask(state)).argmax(-1)

    return act


def random_openings(pairs, plies, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < pairs:
        board, line = chess.Board(), []
        for _ in range(plies):
            move = rng.choice(list(board.legal_moves))
            line.append(move)
            board.push(move)
        if not board.is_game_over():
            out.append(line)
    return out


def configure(engine, elo_rating, skill):
    options = {"Threads": 1, "Hash": 16}
    if elo_rating is not None:
        spec = engine.options["UCI_Elo"]
        if not spec.min <= elo_rating <= spec.max:
            raise SystemExit(
                f"--sf-elo must be within [{spec.min}, {spec.max}] for this Stockfish"
            )
        options.update({"UCI_LimitStrength": True, "UCI_Elo": elo_rating})
    elif skill is not None:
        options["Skill Level"] = skill
    engine.configure(options)


def play_match(
    act, params, key, openings, engines, limit, max_plies, agent_label="agent"
):
    """Play ``2 * len(openings)`` games; returns (wins, draws, losses, truncated, games)."""
    batch = 2 * len(openings)
    lines = openings + openings
    agent_white = np.arange(batch) < len(openings)
    boards = [chess.Board() for _ in range(batch)]
    state = init(jax.random.split(jax.random.fold_in(key, 0), batch))
    white_id = np.asarray(state.current_player)
    agent_id = np.where(agent_white, white_id, 1 - white_id)
    totals = np.zeros((batch, 2))
    pool = ThreadPoolExecutor(max_workers=len(engines))

    def sf_move(g):
        return engines[g].play(boards[g], limit).move

    for ply in range(max_plies):
        done = np.asarray(state.terminated)
        if done.all():
            break
        live = ~done
        agent_turn = np.array([b.turn == w for b, w in zip(boards, agent_white)])
        in_book = ply < len(lines[0])
        actions = np.zeros(batch, dtype=np.int32)
        chosen = {}
        if in_book:
            for g in np.flatnonzero(live):
                chosen[g] = lines[g][ply]
        else:
            if (live & agent_turn).any():
                net = np.asarray(act(params, jax.random.fold_in(key, ply + 1), state))
                for g in np.flatnonzero(live & agent_turn):
                    chosen[g] = moves.action_to_move(net[g], boards[g])
            sf_games = np.flatnonzero(live & ~agent_turn)
            for g, move in zip(sf_games, pool.map(sf_move, sf_games)):
                chosen[g] = move
        for g, move in chosen.items():
            if move not in boards[g].legal_moves:
                raise RuntimeError(f"illegal move {move} in {boards[g].fen()}")
            actions[g] = moves.move_to_action(move, boards[g].turn)
            boards[g].push(move)
        state = step(state, jnp.asarray(actions))
        totals += np.asarray(state.rewards)
        for g in chosen:
            if boards[g].is_game_over() and not bool(state.terminated[g]):
                raise RuntimeError(f"pgx/python-chess disagree at {boards[g].fen()}")
    pool.shutdown()

    truncated = int((~np.asarray(state.terminated)).sum())
    result = totals[np.arange(batch), agent_id]
    games = []
    for g in range(batch):
        game = chess.pgn.Game.from_board(boards[g])
        names = (
            (agent_label, "Stockfish") if agent_white[g] else ("Stockfish", agent_label)
        )
        game.headers.update(White=names[0], Black=names[1])
        game.headers["Result"] = (
            "1/2-1/2"
            if result[g] == 0
            else "1-0"
            if (result[g] > 0) == agent_white[g]
            else "0-1"
        )
        games.append(game)
    return (
        int((result > 0).sum()),
        int((result == 0).sum()),
        int((result < 0).sum()),
        truncated,
        games,
    )


def main():
    p = argparse.ArgumentParser(description="Evaluate the chess net against Stockfish.")
    p.add_argument(
        "--params", help="safetensors written by pgx-rl-chess; omit for a random net"
    )
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--simulations", type=int, default=32, help="0 plays the raw policy")
    p.add_argument("--considered", type=int, default=16)
    p.add_argument("--stockfish", default=shutil.which("stockfish"))
    p.add_argument(
        "--sf-elo", type=int, help="UCI_Elo for Stockfish (strength-limited)"
    )
    p.add_argument(
        "--sf-skill", type=int, help="Skill Level 0-20 (ignored if --sf-elo)"
    )
    p.add_argument(
        "--sf-nodes", type=int, default=5000, help="node limit per Stockfish move"
    )
    p.add_argument(
        "--pairs",
        type=int,
        default=16,
        help="openings; each is played with both colors",
    )
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--pgn", help="write games here")
    p.add_argument(
        "--record", help="write a JSON result record (seed, hashes, counts) here"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.stockfish:
        raise SystemExit("Stockfish not found: install it or pass --stockfish")

    key = repro.seed_everything(args.seed)
    if args.params:
        params, net = repro.load_checkpoint(args.params)
        model, label = zero.Net(**net), args.params
    else:
        print("no --params: evaluating a randomly initialised net")
        net = {"channels": args.channels, "blocks": args.blocks}
        model, label = zero.Net(**net), "random-net"
        params = model.init(key, init(jax.random.split(key, 1)).observation)
    act = make_agent(model, args.simulations, args.considered)
    openings = random_openings(args.pairs, args.opening_plies, args.seed)

    start = time.time()
    with contextlib.ExitStack() as stack:
        engines, engine_name = [], None
        for _ in range(2 * args.pairs):
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            stack.callback(engine.quit)
            configure(engine, args.sf_elo, args.sf_skill)
            engines.append(engine)
            engine_name = engine.id["name"]
        limit = chess.engine.Limit(nodes=args.sf_nodes)
        w, d, l, truncated, games = play_match(
            act, params, key, openings, engines, limit, args.max_plies, label
        )

    if args.pgn:
        with open(args.pgn, "w") as f:
            for game in games:
                print(game, file=f, end="\n\n")
    opponent = (
        f"UCI_Elo {args.sf_elo}"
        if args.sf_elo
        else f"skill {args.sf_skill}"
        if args.sf_skill is not None
        else "full strength"
    )
    print(
        f"{label} ({args.simulations} sims) vs Stockfish ({opponent}, {args.sf_nodes} nodes): "
        f"+{w} ={d} -{l}, {truncated} truncated at {args.max_plies} plies, "
        f"{time.time() - start:.0f}s"
    )
    est = elo.estimate(w, d, l)
    print(f"relative to Stockfish: {est}")
    if args.record:
        repro.write_json(
            args.record,
            {
                "args": vars(args),
                "net": net,
                "params_sha256": repro.hash_params(params),
                "stockfish": engine_name,
                "result": {"wins": w, "draws": d, "losses": l, "truncated": truncated},
                "elo": {"diff": est.elo, "low": est.low, "high": est.high},
                "environment": repro.environment_info(),
            },
        )
    if args.sf_elo:
        print(
            f"approximate absolute Elo (Stockfish UCI_Elo scale): {args.sf_elo + est.elo:.0f}"
        )


if __name__ == "__main__":
    main()
