"""Bayesian baseline model that learns choice frequencies per subject."""

import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


def baseline_model(
    n_arms,
    n_subjects,
    choices=None,
):
    """
    Bayesian baseline model: each subject has a Dirichlet-Categorical distribution.

    This model learns the empirical choice frequencies per subject with a
    symmetric Dirichlet prior. 

    Parameters
    ----------
    n_arms : int
        Number of choice options.
    n_subjects : int
        Number of subjects.
    choices : np.ndarray, optional
        Shape (n_trials, n_subjects). Observed choices.
    """
    # Symmetric Dirichlet prior concentration parameter
    alpha = numpyro.sample("alpha", dist.Exponential(1.0))

    # Sample choice probabilities for each subject
    # probs shape: (n_subjects, n_arms)
    with numpyro.plate("subjects", n_subjects):
        probs = numpyro.sample(
            "probs",
            dist.Dirichlet(jnp.ones(n_arms) * alpha)
        )

    if choices is not None:
        n_trials = choices.shape[0]

        # Flatten choices to (n_trials * n_subjects,)
        # and create subject indices to select correct probs
        choices_flat = choices.T.ravel()  # (n_subjects * n_trials,)
        subject_idx = jnp.repeat(jnp.arange(n_subjects), n_trials)

        # Select probs for each observation: probs[subject_idx] has shape (n_obs, n_arms)
        probs_expanded = probs[subject_idx]

        with numpyro.plate("obs_plate", n_trials * n_subjects):
            numpyro.sample(
                "obs",
                dist.Categorical(probs=probs_expanded),
                obs=choices_flat,
            )

