import jax
import pgx

from pgx_rl.train import ActorCritic, make_loss, make_rollout


def test_rollout_and_loss_are_finite():
    env = pgx.make("tic_tac_toe")
    model = ActorCritic(num_actions=env.num_actions)
    key = jax.random.PRNGKey(0)
    obs = jax.vmap(env.init)(jax.random.split(key, 2)).observation
    params = model.init(key, obs)
    traj = make_rollout(env, model)(params, key, 16)
    loss, _ = make_loss(model)(params, traj)
    assert jax.numpy.isfinite(loss)
