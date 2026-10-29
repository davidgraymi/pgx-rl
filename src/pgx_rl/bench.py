"""Per-machine throughput benchmark for the chess self-play loop.

Times the pieces (net forward, pgx step, one search, a full self-play batch, one update)
after a warm-up call, so the numbers exclude compilation. Run it once per machine and
device and paste the output into docs/ARCHITECTURE.md.
"""

import argparse
import json
import platform
import time

import jax
import optax

from pgx_rl import chess as zero


def timed(fn, *args, repeats):
    jax.block_until_ready(fn(*args))  # compile + warm up
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / repeats


def main():
    p = argparse.ArgumentParser(description="Benchmark chess self-play throughput.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--considered", type=int, default=16)
    p.add_argument(
        "--max-steps", type=int, default=32, help="plies per self-play batch"
    )
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", action="store_true", help="print one JSON line")
    args = p.parse_args()

    env = zero.env
    model = zero.Net(channels=args.channels, blocks=args.blocks)
    key = jax.random.PRNGKey(0)
    state = jax.vmap(env.init)(jax.random.split(key, args.batch))
    params = model.init(key, state.observation)
    n_params = sum(x.size for x in jax.tree.leaves(params))

    search = zero.make_search(model, args.simulations, args.considered)
    selfplay = zero.make_selfplay(model, search, args.max_steps)
    opt = optax.adam(1e-3)
    update = zero.make_update(model, opt)

    forward = jax.jit(lambda p, o: model.apply(p, o))
    step = jax.jit(jax.vmap(env.step))
    search_j = jax.jit(search)
    action = state.legal_action_mask.argmax(-1)

    r = args.repeats
    t_forward = timed(forward, params, state.observation, repeats=r)
    t_step = timed(step, state, action, repeats=r)
    t_search = timed(search_j, params, key, state, repeats=r)
    t_selfplay = timed(lambda: selfplay(params, key, args.batch), repeats=1)
    rec = selfplay(params, key, args.batch)
    t_update = timed(
        lambda: update(params, opt.init(params), rec, key, args.minibatch), repeats=r
    )

    positions = args.batch * args.max_steps
    result = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "net": f"{args.channels}x{args.blocks}",
        "params": int(n_params),
        "batch": args.batch,
        "simulations": args.simulations,
        "forward_ms": 1e3 * t_forward,
        "pgx_step_ms": 1e3 * t_step,
        "search_ms": 1e3 * t_search,
        "selfplay_s": t_selfplay,
        "selfplay_positions_per_s": positions / t_selfplay,
        "update_ms": 1e3 * t_update,
    }
    if args.json:
        print(json.dumps(result))
        return
    print(
        f"{result['device']} | jax {result['jax']} | net {result['net']} "
        f"({n_params / 1e6:.2f}M params) | batch {args.batch} | {args.simulations} sims"
    )
    print(f"  net forward      {result['forward_ms']:9.1f} ms")
    print(f"  pgx step         {result['pgx_step_ms']:9.1f} ms")
    print(f"  search (1 ply)   {result['search_ms']:9.1f} ms")
    print(
        f"  self-play        {t_selfplay:9.2f} s for {positions} positions "
        f"({result['selfplay_positions_per_s']:.0f} positions/s)"
    )
    print(f"  update           {result['update_ms']:9.1f} ms")


if __name__ == "__main__":
    main()
