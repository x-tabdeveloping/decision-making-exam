from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy import stats

# numpyro.set_host_device_count(4)


def inv_probit(x):
    return stats.norm.cdf(x, loc=0.0, scale=1.0)


def utility_function(outcome, u_aversion, u_shape):
    return jnp.where(
        outcome > 0,
        jnp.power(outcome, u_shape),
        -u_aversion * jnp.power(jnp.abs(outcome), u_shape),
    )


def update_qs(qs, data, learning_rates, u_aversion, u_shape):
    n_subjects, n_arms = qs.shape
    choice = data["c"]
    reward = data["r"]
    regret = (
        utility_function(reward, u_aversion, u_shape)
        - qs[jnp.arange(n_subjects), choice]
    )
    qs = qs.at[jnp.arange(n_subjects), choice].add(learning_rates * regret)
    return qs, qs


def trace_qs(c, r, q0s, learning_rates, u_aversion, u_shape):
    _, states = jax.lax.scan(
        partial(
            update_qs,
            learning_rates=learning_rates,
            u_aversion=u_aversion,
            u_shape=u_shape,
        ),
        q0s,
        xs={
            "c": c,
            "r": r,
        },
    )
    return states


def pvl_model(
    stop_condition, n_arms, n_subjects, rewards=None, choices=None, prior=False
):
    lr_loc = numpyro.sample("lr_loc", dist.Normal(loc=0, scale=1))
    lr_scale = numpyro.sample("lr_scale", dist.Uniform(0, 1))
    stop_lr_effect = numpyro.sample("stop_lr_effect", dist.Normal(loc=0, scale=1))
    probit_lr = numpyro.sample(
        "probit_lr",
        dist.Normal(
            lr_loc + stop_lr_effect * stop_condition, jnp.full(n_subjects, lr_scale)
        ),
    )
    lr = numpyro.deterministic("lr", inv_probit(probit_lr))
    inv_t_loc = numpyro.sample("inv_t_loc", dist.Normal(loc=0, scale=1))
    inv_t_scale = numpyro.sample("inv_t_scale", dist.Uniform(0, 1))
    stop_inv_t_effect = numpyro.sample("stop_inv_t_effect", dist.Normal(loc=0, scale=1))
    probit_inv_t = numpyro.sample(
        "probit_inv_t",
        dist.Normal(
            inv_t_loc + stop_inv_t_effect * stop_condition,
            jnp.full(n_subjects, inv_t_scale),
        ),
    )
    inv_t = numpyro.deterministic("inv_t", 5 * inv_probit(probit_inv_t))
    u_shape_loc = numpyro.sample("u_shape_loc", dist.Normal(loc=0, scale=1))
    u_shape_scale = numpyro.sample("u_shape_scale", dist.Uniform(0, 1))
    stop_u_shape_effect = numpyro.sample(
        "stop_u_shape_effect", dist.Normal(loc=0, scale=1)
    )
    probit_u_shape = numpyro.sample(
        "probit_u_shape",
        dist.Normal(
            u_shape_loc + stop_u_shape_effect * stop_condition,
            jnp.full(n_subjects, u_shape_scale),
        ),
    )
    u_shape = numpyro.deterministic("u_shape", inv_probit(probit_u_shape))
    u_aversion_loc = numpyro.sample("u_aversion_loc", dist.Normal(loc=0, scale=1))
    u_aversion_scale = numpyro.sample("u_aversion_scale", dist.Uniform(0, 1))
    stop_u_aversion_effect = numpyro.sample(
        "stop_u_aversion_effect", dist.Normal(loc=0, scale=1)
    )
    probit_u_aversion = numpyro.sample(
        "probit_u_aversion",
        dist.Normal(
            u_aversion_loc + stop_u_aversion_effect * stop_condition,
            jnp.full(n_subjects, u_aversion_scale),
        ),
    )
    u_aversion = numpyro.deterministic("u_aversion", 5 * inv_probit(probit_u_aversion))
    if not prior:
        q0 = np.zeros((n_subjects, n_arms), dtype=jnp.float64)
        q = trace_qs(choices, rewards, q0, lr, u_aversion, u_shape)
        logits = q * inv_t[:, None]
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=choices)
