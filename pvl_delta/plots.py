import arviz as az
import jax
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pvl_delta import inv_probit, utility_function


def plot_utilities(idata, load_blocks, stop_condition):
    fig = make_subplots(
        cols=2, rows=1, subplot_titles=["No-load blocks", "Load blocks"]
    )
    r = np.linspace(-5, 5, 200)
    for load in [0, 1]:
        for stop in [0, 1]:
            shape = np.ravel(
                idata.posterior["u_shape"].values[
                    :, :, (stop_condition == stop)[None, :] & (load_blocks == load)
                ]
            )
            aversion = np.ravel(
                idata.posterior["u_aversion"].values[
                    :, :, (stop_condition == stop)[None, :] & (load_blocks == load)
                ]
            )
            u = utility_function(r, u_aversion=aversion.mean(), u_shape=shape.mean())
            color = "red" if bool(stop) else "blue"
            fig.add_scatter(
                name="Stop" if bool(stop) else "Non-stop",
                showlegend=bool(load),
                x=r,
                y=u,
                line=dict(color=color),
                mode="lines",
                col=load + 1,
                row=1,
            )
    fig = fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
    )
    fig = fig.update_annotations(
        font=dict(family="Times New Roman", size=18, color="black")
    )
    fig = fig.update_yaxes(matches="y", title="Utility")
    fig = fig.update_xaxes(title="Reward")
    return fig


def plot_choices(idata, choices, prior=False):
    if prior:
        pred_choices = idata.prior["obs"].mean(dim=["chain", "draw"]).values
    else:
        pred_choices = (
            idata.posterior_predictive["obs"].mean(dim=["chain", "draw"]).values
        )
    n_trials, n_subjects = pred_choices.shape
    trials = np.arange(n_trials)
    fig = go.Figure()
    color_scheme = px.colors.diverging.Portland
    subject_colors = px.colors.sample_colorscale(
        color_scheme, np.arange(n_subjects) / n_subjects
    )
    for i_subject in range(n_subjects):
        true = choices[:, i_subject]
        pred = pred_choices[:, i_subject]
        fig.add_scatter(
            legendgroup=i_subject,
            legendgrouptitle_text=f"Subject {i_subject}",
            name="Choices",
            x=trials,
            y=true,
            line=dict(color=subject_colors[i_subject], shape="hv"),
            opacity=0.5,
        )
        fig.add_scatter(
            legendgroup=i_subject,
            legendgrouptitle_text=f"Subject {i_subject}",
            name="Predictions",
            x=trials,
            y=pred,
            line=dict(color=subject_colors[i_subject], dash="dash", shape="hv"),
            opacity=0.5,
        )
    fig = fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
    )
    fig = fig.update_annotations(
        font=dict(family="Times New Roman", size=18, color="black")
    )
    fig = fig.update_yaxes(matches="y", title="Utility")
    fig = fig.update_xaxes(title="Reward")
    return fig


def plot_effects(idata):
    vars = {"u_shape": "A'", "u_aversion": "w'", "lr": "a'", "inv_t": "c'"}
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i_var, var in enumerate(vars):
        for i_cond, cond in enumerate(["load", "stop"]):
            i_effect = i_var + i_cond * len(vars)
            effect_name = f"{cond}_{var}_effect"
            lower, higher = az.hdi(idata.posterior, var_names=effect_name)[
                effect_name
            ].values
            mean = idata.posterior[effect_name].values.mean()
            proper_name = vars[var]
            effect_title = "\\beta" if cond == "stop" else "\\pi"
            effect_title = f"${effect_title}_{proper_name}$"
            color = colors[i_effect]
            data = np.ravel(idata.posterior[effect_name].values)
            fig.add_trace(
                go.Violin(
                    x=data,
                    y0=effect_title,
                    line_color=color,
                    name=effect_name,
                    showlegend=False,
                    orientation="h",
                    points=False,
                    side="positive",
                    width=1.5,
                    opacity=0.7,
                )
            )
            fig.add_scatter(
                x=[mean],
                y=[effect_title],
                marker=dict(color=color, line=dict(width=2, color="black")),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[higher - mean],
                    arrayminus=[mean - lower],
                ),
                showlegend=False,
            )
    fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=True)
    fig = fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    fig = fig.update_annotations(
        font=dict(family="Times New Roman", size=18, color="black")
    )
    fig = fig.update_xaxes(title="Effect size")
    return fig


def plot_params(idata, stop_condition):
    vars = {"u_shape": "A", "u_aversion": "w", "lr": "a", "inv_t": "c"}
    fig = make_subplots(
        cols=2,
        rows=4,
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
        subplot_titles=["No-load blocks", "Load blocks"],
    )
    for i_param, param in enumerate(vars):
        probit_intercept = idata.posterior[f"probit_{param}"].values
        probit_intercept = probit_intercept.reshape(-1, probit_intercept.shape[-1])
        effect = np.ravel(idata.posterior[f"load_{param}_effect"].values)
        for i_subj, subject_int in enumerate(probit_intercept.T):
            noload = np.array(inv_probit(subject_int))
            load = np.array(inv_probit(subject_int + effect))
            lower, higher = az.hdi(noload)
            mean = noload.mean()
            load_lower, load_higher = az.hdi(load)
            load_mean = load.mean()
            color = "red" if stop_condition[i_subj] else "blue"
            fig.add_violin(
                orientation="h",
                x=noload,
                y0=i_subj,
                line_color=color,
                showlegend=False,
                col=1,
                row=i_param + 1,
                points=False,
                side="positive",
                width=1.5,
                opacity=0.7,
            )
            fig.add_scatter(
                x=[mean],
                y=[i_subj],
                marker=dict(color=color, line=dict(width=2, color="black")),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[higher - mean],
                    arrayminus=[mean - lower],
                ),
                col=1,
                row=i_param + 1,
                showlegend=False,
            )
            fig.update_xaxes(row=i_param + 1, col=1, title=vars[param])
            fig.add_violin(
                orientation="h",
                x=load,
                y0=i_subj,
                line_color=color,
                showlegend=False,
                col=2,
                row=i_param + 1,
                points=False,
                side="positive",
                width=1.5,
                opacity=0.7,
            )
            fig.add_scatter(
                x=[load_mean],
                y=[i_subj],
                marker=dict(color=color, line=dict(width=2, color="black")),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[load_higher - load_mean],
                    arrayminus=[load_mean - load_lower],
                ),
                showlegend=False,
                col=2,
                row=i_param + 1,
            )
            fig.update_xaxes(row=i_param + 1, col=2, title=vars[param])
    fig = fig.update_xaxes(matches="x")
    fig = fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
        margin=dict(t=30, l=10, r=10, b=10),
    )
    fig = fig.update_annotations(
        font=dict(family="Times New Roman", size=18, color="black")
    )
    fig = fig.update_yaxes(title="Subject")
    return fig
