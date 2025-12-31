from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score

from pvl_delta import trace_qs

experiments_path = Path("experiments")
experiment_files = [file for file in experiments_path.glob("*.joblib")]


def calculate_f1s(idata, choices):
    pred_choices = idata.posterior_predictive["obs"].squeeze()
    c = np.ravel(choices)
    f1s = []
    for draw, p_choice in pred_choices.groupby("draw"):
        p_c = np.ravel(p_choice)
        f1s.append(f1_score(p_c, c, average="macro"))
    return np.array(f1s)


# Plotting subject level parameter recovery
subject_level_params = ["lr", "inv_t", "u_shape", "u_aversion"]
fig = make_subplots(rows=len(subject_level_params), cols=len(experiment_files))
color_scheme = px.colors.qualitative.Bold
records = []
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    idata = data["idata"]
    stop_condition = data["stop_condition"]
    for i_param, param in enumerate(subject_level_params):
        vals = idata.posterior[param].values
        vals = vals.reshape(-1, vals.shape[-1])
        for i_subject, (subject_vals, subject_stop) in enumerate(
            zip(vals.T, stop_condition)
        ):
            fig.add_trace(
                go.Box(
                    name=f"Subject {i_subject}",
                    showlegend=False,
                    y=subject_vals,
                    line_color=color_scheme[int(subject_stop)],
                ),
                col=i_experiment + 1,
                row=i_param + 1,
            )
fig.show()


records = []
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    idata = data["idata"]
    c = data["choices"]
    f1s = calculate_f1s(idata, c)
    for f1 in f1s:
        records.append(dict(f1=f1, experiment=i_experiment))
df = pd.DataFrame.from_records(records)
print(df.groupby("experiment").agg(["mean", "std"]))


# Effect and population level parameter recovery
fig = make_subplots(rows=2, cols=5)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    idata = data["idata"]
    param_names = [
        "stop_lr_effect",
        "stop_inv_t_effect",
        "lr_loc",
        "inv_t_loc",
        "u_aversion_loc",
        "u_shape_loc",
    ]
    color_scheme = px.colors.qualitative.Pastel
    col = (i_experiment % 5) + 1
    row = (i_experiment // 5) + 1
    for i_effect, effect in enumerate(param_names):
        effect_name = " ".join(effect.split("_")).title()
        fig.add_trace(
            go.Violin(
                x0=effect_name,
                y=np.ravel(idata.posterior[effect]),
                meanline_visible=True,
                name=effect_name,
                fillcolor=color_scheme[i_effect],
                line_color="black",
                opacity=0.5,
                showlegend=False,
            ),
            col=col,
            row=row,
        )
        fig.add_scatter(
            x=[effect_name],
            y=[data["params"][effect]],
            marker=dict(
                size=22,
                symbol="diamond-wide-dot",
                color=color_scheme[i_effect],
                line=dict(width=2, color="black"),
            ),
            showlegend=False,
            col=col,
            row=row,
        )
fig.show()

data["ev"]

# Plotting EV recovery
n_experiments = len(experiment_files)
n_subjects = len(data["params"]["lr"])
fig = make_subplots(rows=n_subjects + 1, cols=n_experiments)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    idata = data["idata"]
    lr = data["params"]["lr"]
    n_subjects = lr.shape[0]
    n_arms = len(np.unique(data["choices"]))
    color_scheme = px.colors.qualitative.Bold
    # trials, subjects, arms
    q_true = trace_qs(
        c=data["choices"],
        r=data["rewards"],
        q0s=np.zeros((n_subjects, n_arms)),
        learning_rates=lr,
        u_aversion=data["params"]["u_aversion"],
        u_shape=data["params"]["u_shape"],
    )
    # subjects, arms, trials
    q_true = np.transpose(q_true, (1, 2, 0))
    mean_posterior = idata.posterior.mean(dim=["chain", "draw"])
    # trials, subjects, arms
    q_pred = trace_qs(
        c=data["choices"],
        r=data["rewards"],
        q0s=np.zeros((n_subjects, n_arms)),
        learning_rates=mean_posterior["lr"].values,
        u_aversion=mean_posterior["u_aversion"].values,
        u_shape=mean_posterior["u_shape"].values,
    )
    # subjects, arms, trials
    q_pred = np.transpose(q_pred, (1, 2, 0))
    for i_subject, q_subj in enumerate(q_pred):
        for arm, pred in enumerate(q_subj):
            true = q_true[i_subject, arm, :]
            fig.add_scatter(
                x=np.arange(len(pred)),
                y=pred,
                line=dict(dash="dash", color=color_scheme[arm]),
                showlegend=False,
                col=i_experiment + 1,
                row=i_subject + 1,
            )
            fig.add_scatter(
                x=np.arange(len(pred)),
                y=true,
                line=dict(color=color_scheme[arm]),
                showlegend=False,
                col=i_experiment + 1,
                row=i_subject + 1,
            )
            fig.add_scatter(
                x=[len(pred) - 1],
                y=[data["ev"][arm]],
                mode="markers",
                marker=dict(
                    color=color_scheme[arm], size=12, line=dict(color="black", width=2)
                ),
                showlegend=False,
                col=i_experiment + 1,
                row=i_subject + 1,
            )
    # Last row is for effect recovery
    param_names = [
        "stop_lr_effect",
        "stop_inv_t_effect",
        "lr_loc",
        "inv_t_loc",
        "u_aversion_loc",
        "u_shape_loc",
    ]
    for i_effect, effect in enumerate(param_names):
        effect_name = " ".join(effect.split("_")).title()
        fig.add_trace(
            go.Box(
                y=np.ravel(idata.posterior[effect]),
                name=effect_name,
                line_color=color_scheme[i_effect],
                opacity=0.7,
                showlegend=False,
            ),
            col=i_experiment + 1,
            row=n_subjects + 1,
        )
        fig.add_scatter(
            x=[effect_name],
            y=[data["params"][effect]],
            marker=dict(
                size=12,
                symbol="diamond-wide-dot",
                color=color_scheme[i_effect],
                line=dict(width=2, color="black"),
            ),
            showlegend=False,
            col=i_experiment + 1,
            row=n_subjects + 1,
        )
fig = fig.update_xaxes(matches=None)
fig = fig.update_layout(template="plotly_white", margin=dict(r=0, l=0, t=0, b=0))
fig.show()
