from operator import attrgetter
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import joblib
import numpyro
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam

from pvl_delta import (interaction_model, inv_probit, pvl_delta_model,
                       utility_function, vanilla_model)

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

MODELS = {
    "full": pvl_delta_model,
    "interactions_only": interaction_model,
    "vanilla": vanilla_model,
}


def diagnostics_summary(mcmc, exclude_deterministic=True, prob=0.95):
    sites = mcmc._states[mcmc._sample_field]
    if isinstance(sites, dict) and exclude_deterministic:
        state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mcmc._states[mcmc._sample_field].items()
                if k in state_sample_field
            }
    return summary(sites, prob=prob)


def simulate_outcomes(
    params,
    stop_condition,
    n_trials_per_subject,
    p_win,
    reward_arm,
    random_state=42,
):
    n_subjects = params["probit_lr"].shape[0]
    key = jax.random.key(random_state)
    n_arms = p_win.shape[0]
    q0s = jnp.zeros((n_subjects, n_arms))
    q_ts = q0s
    choices = []
    rewards = []
    key, subkey = jax.random.split(key)
    # Randomly generating load for first trial
    load_blocks = [jax.random.binomial(subkey, 1, jnp.full(n_subjects, 0.5))]
    for i_trial in range(n_trials_per_subject):
        _choice = []
        _reward = []
        load = load_blocks[-1]
        inv_t = 5 * inv_probit(
            params["probit_inv_t"]
            + params["stop_inv_t_effect"] * stop_condition
            + params["load_inv_t_effect"] * load
            + params["interaction_inv_t_effect"] * load * stop_condition
        )
        theta = jnp.power(3, inv_t) - 1
        u_aversion = 5 * inv_probit(
            params["probit_u_aversion"]
            + params["stop_u_aversion_effect"] * stop_condition
            + params["load_u_aversion_effect"] * load
            + +params["interaction_u_aversion_effect"] * load * stop_condition
        )
        u_shape = inv_probit(
            params["probit_u_shape"]
            + params["stop_u_shape_effect"] * stop_condition
            + params["load_u_shape_effect"] * load
            + params["interaction_u_shape_effect"] * load * stop_condition
        )
        lr = inv_probit(
            params["probit_lr"]
            + params["stop_lr_effect"] * stop_condition
            + params["load_lr_effect"] * load
            + params["interaction_lr_effect"] * load * stop_condition
        )
        for i_subject in range(n_subjects):
            logits = theta[i_subject] * q_ts[i_subject]
            key, subkey = jax.random.split(key)
            choice = jax.random.categorical(subkey, logits=logits)
            key, subkey = jax.random.split(key)
            success = jax.random.binomial(subkey, 1, p_win[choice])
            key, subkey = jax.random.split(key)
            stake = jax.random.categorical(subkey, jnp.ones(3))
            # Calculating reward based on stake.
            # For each step in stake we multiply by two
            # reward_arm contains the lowest stakes
            potential_reward = reward_arm[choice] * jnp.power(2, stake)
            if success == 1:
                reward = potential_reward
            else:
                # Subjects lose half the amount on non-succesful trials
                reward = -potential_reward / 2
            regret = (
                utility_function(
                    reward,
                    u_aversion[i_subject],
                    u_shape[i_subject],
                )
                - q_ts[i_subject, choice]
            )
            q_ts = q_ts.at[i_subject, choice].add(lr[i_subject] * regret)
            _reward.append(reward)
            _choice.append(choice)
        # Alternating load-no_load
        load_blocks.append(jnp.abs(load - 1))
        choices.append(_choice)
        rewards.append(_reward)
    return jnp.array(rewards), jnp.array(choices), jnp.stack(load_blocks[:-1])


def main():
    n_trials = 202
    n_subjects = 64
    n_experiments = 10
    n_arms = 6
    p_win = jnp.array([0.2, 0.25, 0.33, 0.47, 0.61, 0.87])
    # We denote them in the lowest stake
    win_reward = jnp.array([64, 32, 16, 8, 4, 2])
    # We calculate joint EV for all stakes
    ev = win_reward / 3 * (10.5 * p_win - 3.5)
    experiments_dir = Path("simulation_results/")
    experiments_dir.mkdir(exist_ok=True)
    keys = [jax.random.key(n) for n in range(40, 50)]
    for i_experiment in range(n_experiments):
        experiment = dict()
        print(f"=============[Experiment {i_experiment}]==============")
        print("Sampling parameters from prior")
        key = keys[i_experiment]
        experiment["key"] = keys[i_experiment]
        key, subkey = jax.random.split(key)
        stop_condition = jax.random.bernoulli(subkey, p=0.5, shape=n_subjects)
        experiment["stop_condition"] = stop_condition
        # We always generate from the full model:
        prior = Predictive(
            pvl_delta_model,
            num_samples=1,
        )
        key, subkey = jax.random.split(key)
        params = prior(
            subkey,
            prior=True,
            n_subjects=n_subjects,
            n_arms=n_arms,
            stop_condition=stop_condition,
        )
        params = {key: value[0] for key, value in params.items()}
        print("True parameters: ", params)
        experiment["params"] = params
        experiment["stop_condition"] = stop_condition
        print("Simulating rewards")
        experiment["p_win"] = p_win
        experiment["win_reward"] = win_reward
        experiment["ev"] = ev
        print("True expected values: ", ev)
        print("Simulating outcomes: ")
        rewards, choices, load_blocks = simulate_outcomes(
            params, stop_condition, n_trials, p_win, win_reward
        )
        experiment["load_blocks"] = load_blocks
        experiment["rewards"] = rewards
        experiment["choices"] = choices
        for model_name, model in MODELS.items():
            experiment[model_name] = dict()
            print("Recovering parameters with model:", model_name)
            config = {
                "probit_lr": LocScaleReparam(centered=0),
                "probit_inv_t": LocScaleReparam(centered=0),
                "probit_u_shape": LocScaleReparam(centered=0),
                "probit_u_aversion": LocScaleReparam(centered=0),
            }
            # Reparameterizing model before running
            reparam_model = numpyro.handlers.reparam(model, config=config)
            nuts_kernel = NUTS(reparam_model, target_accept_prob=0.9, max_tree_depth=15)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=1000,
                num_warmup=5000,
                num_chains=4,
            )
            key, subkey = jax.random.split(key)
            mcmc.run(
                subkey,
                choices=choices,
                rewards=rewards,
                stop_condition=stop_condition,
                load_blocks=load_blocks,
                n_arms=n_arms,
                n_subjects=n_subjects,
            )
            print("Sampling summary:")
            mcmc.print_summary(prob=0.95)
            samples = mcmc.get_samples()
            experiment[model_name]["summary"] = diagnostics_summary(samples, prob=0.95)
            idata = az.from_numpyro(mcmc)
            print("Sampling posterior predictive")
            predictive = Predictive(reparam_model, mcmc.get_samples())
            key, subkey = jax.random.split(key)
            posterior_predictive = predictive(
                subkey,
                choices=choices,
                rewards=rewards,
                stop_condition=stop_condition,
                n_arms=n_arms,
                n_subjects=n_subjects,
                load_blocks=load_blocks,
            )
            idata.extend(az.from_numpyro(posterior_predictive=posterior_predictive))
            experiment[model_name]["idata"] = idata
            print("Saving experiment data")
            joblib.dump(
                experiment,
                experiments_dir.joinpath(f"experiment_{i_experiment}.joblib"),
            )


if __name__ == "__main__":
    main()
