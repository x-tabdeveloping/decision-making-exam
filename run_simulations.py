from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import joblib
from numpyro.infer import MCMC, NUTS, Predictive

from pvl_delta import pvl_delta_model


def simulate_outcomes(
    params,
    stop_condition,
    n_trials_per_subject,
    p_win,
    reward_arm,
    random_state=42,
):
    n_subjects = params["lr"].shape[0]
    key = jax.random.key(random_state)
    n_arms = p_win.shape[0]
    q0s = jnp.zeros((n_subjects, n_arms))
    q_ts = q0s
    choices = []
    rewards = []
    for i_trial in range(n_trials_per_subject):
        _choice = []
        _reward = []
        for i_subject in range(n_subjects):
            logits = params["inv_t"][i_subject] * q_ts[i_subject]
            key, subkey = jax.random.split(key)
            choice = jax.random.categorical(subkey, logits=logits)
            key, subkey = jax.random.split(key)
            reward = jax.random.binomial(key, 1, p_win[choice]) * reward_arm[choice]
            regret = reward - q_ts[i_subject, choice]
            q_ts = q_ts.at[i_subject, choice].add(params["lr"][i_subject] * regret)
            _reward.append(reward)
            _choice.append(choice)
        choices.append(_choice)
        rewards.append(_reward)
    return jnp.array(rewards), jnp.array(choices)


def main():
    n_trials = 100
    n_subjects = 4
    n_experiments = 10
    n_arms = 4
    key = jax.random.key(42)
    experiments_dir = Path("/experiments")
    experiments_dir.mkdir(exist_ok=True)

    for i_experiment in range(n_experiments):
        experiment = dict()
        print(f"=============[Experiment {i_experiment}]==============")
        print("Sampling parameters from prior")
        key, subkey = jax.random.split(key)
        stop_condition = jax.random.bernoulli(subkey, p=0.5, shape=n_subjects)
        experiment["stop_condition"] = stop_condition
        prior = Predictive(
            pvl_model,
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
        key, subkey = jax.random.split(key)
        p_win = jax.random.uniform(subkey, shape=n_arms, minval=0, maxval=1)
        key, subkey = jax.random.split(key)
        win_reward = jax.random.gamma(subkey, a=1.0, shape=n_arms)
        ev = p_win * win_reward
        experiment["p_win"] = p_win
        experiment["win_reward"] = win_reward
        experiment["ev"] = ev
        print("True expected values: ", ev)
        print("Simulating outcomes: ")
        rewards, choices = simulate_outcomes(
            params, stop_condition, n_trials, p_win, win_reward
        )
        experiment["rewards"] = rewards
        experiment["choices"] = choices
        print("Recovering parameters with model:")
        nuts_kernel = NUTS(pvl_delta_model, target_accept_prob=0.9)
        mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=3000, num_chains=4)
        key, subkey = jax.random.split(key)
        mcmc.run(
            subkey,
            choices=choices,
            rewards=rewards,
            stop_condition=stop_condition,
            n_arms=n_arms,
            n_subjects=n_subjects,
        )
        print("Sampling summary:")
        mcmc.print_summary()
        idata = az.from_numpyro(mcmc)
        print("Sampling posterior predictive")
        predictive = Predictive(pvl_delta_model, mcmc.get_samples())
        key, subkey = jax.random.split(key)
        posterior_predictive = predictive(
            subkey,
            choices=choices,
            rewards=rewards,
            stop_condition=stop_condition,
            n_arms=n_arms,
            n_subjects=n_subjects,
        )
        idata.extend(az.from_numpyro(posterior_predictive=posterior_predictive))
        experiment["idata"] = idata
        print("Saving experiment data")
        joblib.dump(
            experiment, experiments_dir.joinpath(f"experiment_{i_experiment}.joblib")
        )


if __name__ == "__main__":
    main()
