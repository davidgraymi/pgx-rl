import jax
import optax

from pgx_rl import chess


def test_selfplay_and_update_run():
    model = chess.Net(channels=8, blocks=1)
    key = jax.random.PRNGKey(0)
    params = model.init(
        key, jax.vmap(chess.env.init)(jax.random.split(key, 1)).observation
    )
    search = chess.make_search(model, num_simulations=4, considered=2)
    rec = chess.make_selfplay(model, search, max_steps=4)(params, key, 2)
    assert rec["obs"].shape[:2] == (4, 2)
    opt = optax.adam(1e-3)
    _, _, (loss, *_) = chess.make_update(model, opt)(
        params, opt.init(params), rec, key, 4
    )
    assert jax.numpy.isfinite(loss)
