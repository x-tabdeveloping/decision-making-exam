from pathlib import Path

import arviz as az
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from tqdm import tqdm

from pvl_delta import trace_qs


def format_param_name(param_name):
    vars = {"u_shape": "A'", "u_aversion": "w'", "lr": "a'", "inv_t": "c'"}
    conds = {"load": "\\pi", "stop": "\\beta", "interaction": "\\zeta"}
    if param_name.endswith("_effect"):
        param_name = param_name.removesuffix("_effect")
        cond, var = param_name.split("_", 1)
        cond, var = conds[cond], vars[var]
        return f"${cond}_{{{var}}}$"
    if param_name.startswith("probit_"):
        param_name = param_name.removeprefix("probit_").removesuffix("_decentered")
        var = vars[param_name]
        return f"${var}$"
    if param_name.endswith("_loc"):
        param_name = param_name.removesuffix("_loc")
        var = vars[param_name]
        return f"$\\mu_{{{var}}}$"
    if param_name.endswith("_scale"):
        param_name = param_name.removesuffix("_scale")
        var = vars[param_name]
        return f"$\\sigma_{{{var}}}$"


experiments_path = Path("simulation_results")
experiment_files = [
    experiments_path.joinpath(f"experiment_{i}.joblib") for i in [0, 1, 2, 3, 4]
]

comp_dfs = []
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    comp_df = az.compare(
        {
            model_name: data[model_name]["idata"]
            for model_name in ["vanilla", "interactions_only", "full"]
        },
        ic="waic",
    )
    print(comp_df)
    comp_dfs.append(comp_df)

for experiment_file, comp_df in zip(experiment_files, comp_dfs):
    comp_df["experiment"] = experiment_file.stem.split("_")[-1]
overall_comp = pd.concat([df.reset_index() for df in comp_dfs])

fig = px.scatter(
    overall_comp,
    y="index",
    color="index",
    x="elpd_waic",
    facet_col="experiment",
    error_x="se",
    color_discrete_sequence=px.colors.qualitative.Dark24,
)
fig = fig.update_xaxes(matches=None)
fig = fig.update_layout(
    width=800,
    height=300,
    template="plotly_white",
    font=dict(family="Times New Roman", size=16, color="black"),
    margin=dict(r=10, l=10, t=30, b=10),
)
fig = fig.update_traces(showlegend=False)
fig = fig.update_traces(marker=dict(line=dict(width=2, color="black")))
fig = fig.update_yaxes(title="")
fig.show()

fig = make_subplots(cols=len(experiment_files), rows=2)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    for i_model, model in enumerate(["interactions_only", "full"]):
        if model not in data:
            print(model, "not in data, skipping")
            continue
        idata = data[model]["idata"]
        params = data["params"]
        vars = {"u_shape": "A'", "u_aversion": "w'", "lr": "a'", "inv_t": "c'"}
        colors = px.colors.qualitative.Dark24
        for i_var, var in enumerate(vars):
            for i_cond, cond in enumerate(["load", "stop", "interaction"]):
                i_effect = i_var + i_cond * len(vars)
                effect_name = f"{cond}_{var}_effect"
                if effect_name not in idata.posterior:
                    continue
                lower, higher = az.hdi(idata.posterior, var_names=effect_name)[
                    effect_name
                ].values
                mean = idata.posterior[effect_name].values.mean()
                proper_name = vars[var]
                effect_title = (
                    "\\beta"
                    if cond == "stop"
                    else "\\pi" if cond == "load" else "\\zeta"
                )
                effect_title = f"${effect_title}_{{{proper_name}}}$"
                color = colors[i_effect]
                vals = np.ravel(idata.posterior[effect_name].values)
                fig.add_trace(
                    go.Violin(
                        x=vals,
                        y0=effect_title,
                        line_color=color,
                        name=effect_name,
                        showlegend=False,
                        orientation="h",
                        points=False,
                        side="positive",
                        width=0.5,
                        opacity=0.7,
                    ),
                    col=i_experiment + 1,
                    row=i_model + 1,
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
                    col=i_experiment + 1,
                    row=i_model + 1,
                )
                fig.add_scatter(
                    x=[params[effect_name]],
                    y=[effect_title],
                    marker=dict(color="white", line=dict(width=2, color="black")),
                    showlegend=False,
                    col=i_experiment + 1,
                    row=i_model + 1,
                )
fig.add_vline(x=0, line_width=2, line_color="black")
fig = fig.update_xaxes(range=(-6, 6))
fig = fig.update_layout(
    template="plotly_white",
    font=dict(family="Times New Roman", size=16, color="black"),
    margin=dict(t=10, l=10, r=10, b=10),
)
fig = fig.update_annotations(
    font=dict(family="Times New Roman", size=18, color="black")
)
fig = fig.update_xaxes(title="Effect size")
fig.show()

fig = make_subplots(cols=len(experiment_files), rows=3)
n_subjects = data["full"]["idata"].posterior["probit_lr"].shape[-1]
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    for i_model, model in enumerate(["vanilla", "interactions_only", "full"]):
        if model not in data:
            print(model, "not in data, skipping")
            continue
        idata = data[model]["idata"]
        params = data["params"]
        vars = {
            "probit_u_shape": "A'",
            "probit_u_aversion": "w'",
            "probit_lr": "a'",
            "probit_inv_t": "c'",
        }
        colors = px.colors.qualitative.Dark24
        for i_var, var in enumerate(vars):
            fig.add_scatter(
                x=params[var],
                y=params[var],
                mode="lines",
                col=i_experiment + 1,
                row=i_model + 1,
                line=dict(width=2, color="black"),
                showlegend=False,
            )
            for i_subject in range(n_subjects):
                vals = np.ravel(idata.posterior[var][:, :, i_subject].values)
                lower, higher = az.hdi(vals)
                mean = np.mean(vals)
                proper_name = vars[var]
                color = colors[i_var]
                fig.add_scatter(
                    name=vars[var],
                    y=[mean],
                    x=[params[var][i_subject]],
                    marker=dict(color=color, line=dict(width=2, color="black")),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[higher - mean],
                        arrayminus=[mean - lower],
                    ),
                    showlegend=(i_model == 0)
                    and (i_experiment == 0)
                    and (i_subject == 0),
                    col=i_experiment + 1,
                    row=i_model + 1,
                )
fig = fig.update_layout(
    template="plotly_white",
    font=dict(family="Times New Roman", size=16, color="black"),
    margin=dict(t=10, l=10, r=10, b=10),
)
fig = fig.update_annotations(
    font=dict(family="Times New Roman", size=18, color="black")
)
fig = fig.update_xaxes(title="")
fig.show()


# Effect and population level parameter recovery
fig = make_subplots(rows=2, cols=5)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    idata = data["full"]["idata"]
    param_names = [
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

records = []
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    for model in ["vanilla", "interactions_only", "full"]:
        summary = data[model]["summary"]
        divergences = np.sum(data["vanilla"]["idata"].sample_stats["diverging"].values)
        for param in summary.keys():
            n_eff = np.mean(summary[param]["n_eff"])
            rhat = np.mean(summary[param]["r_hat"])
            records.append(
                {
                    "Experiment": i_experiment,
                    "Model": model,
                    "Parameter": param,
                    "R^": rhat,
                    "N_eff": n_eff,
                    "Divergences": divergences,
                }
            )
sampling_df = pd.DataFrame.from_records(records)
sampling_df["Parameter"] = sampling_df["Parameter"].map(format_param_name)

fig = px.bar(
    sampling_df, y="N_eff", x="Parameter", facet_row="Model", facet_col="Experiment"
)
fig = fig.add_hline(y=400)
fig = fig.update_xaxes(
    matches=None,
)
fig = fig.update_yaxes(title="Effective Sample Size", col=1)
fig = fig.update_layout(
    template="plotly_white", font=dict(family="Times New Roman", size=16)
)
fig = fig.update_traces(marker=dict(color="white", line=dict(color="black", width=2)))
fig.show()

fig = px.bar(
    sampling_df, y="R^", x="Parameter", facet_row="Model", facet_col="Experiment"
)
fig = fig.add_hline(y=1.01)
fig = fig.update_xaxes(
    matches=None,
)
fig = fig.update_yaxes(title="", col=1)
fig = fig.update_layout(
    template="plotly_white", font=dict(family="Times New Roman", size=16)
)
fig = fig.update_traces(marker=dict(color="white", line=dict(color="black", width=2)))
fig.show()

sampling_df.groupby(["Experiment", "Model"])["Divergences"].mean().reset_index()

set(data["vanilla"]["idata"].posterior.keys())

fig = make_subplots(
    cols=len(experiment_files), rows=3, horizontal_spacing=0.02, vertical_spacing=0.05
)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    for i_model, model in enumerate(["vanilla", "interactions_only", "full"]):
        idata = data[model]["idata"]
        # We remove deterministic trial-level parameters and decentered
        params = set(idata.posterior.keys()) - {
            "theta",
            "lr",
            "inv_t",
            "u_aversion",
            "u_shape",
            "probit_lr",
            "probit_inv_t",
            "probit_u_shape",
            "probit_u_aversion",
        }
        params = {param for param in params if not param.endswith("_decentered")}
        posterior_vars = {}
        for param in params:
            proper_name = format_param_name(param)
            vals = idata.posterior[param].values
            if len(vals.shape) == 3:
                # if subject level, break it up
                for i_subj, subj_vals in enumerate(vals.transpose((-1, 0, 1))):
                    posterior_vars[f"{proper_name}[{i_subj}]"] = np.ravel(subj_vals)
            else:
                posterior_vars[proper_name] = np.ravel(vals)
        params = list(posterior_vars.keys())
        corr_mat = np.eye(len(params))
        i_s, j_s = np.tril_indices(len(params), k=-1)
        for i, j in tqdm(
            list(zip(i_s, j_s)),
            desc="Calculating correlation matrix.",
        ):
            r = spearmanr(posterior_vars[params[i]], posterior_vars[params[j]])[0]
            corr_mat[i, j] = r
            corr_mat[j, i] = r
        fig.add_heatmap(
            z=corr_mat,
            x=params,
            y=params,
            coloraxis="coloraxis1",
            col=i_experiment + 1,
            row=i_model + 1,
        )
fig.show()


fig = make_subplots(
    cols=len(experiment_files), rows=3, horizontal_spacing=0.02, vertical_spacing=0.05
)
for i_experiment, experiment_file in enumerate(experiment_files):
    print(experiment_file)
    data = joblib.load(experiment_file)
    for i_model, model in enumerate(["vanilla", "interactions_only", "full"]):
        idata = data[model]["idata"]
        # We remove deterministic trial-level parameters and decentered
        params = [
            "probit_lr",
            "probit_inv_t",
            "probit_u_shape",
            "probit_u_aversion",
        ]
        n_subjects = idata.posterior["probit_lr"].values.shape[-1]
        corrs = np.stack([np.eye(len(params))] * n_subjects)
        i_s, j_s = np.tril_indices(len(params), k=-1)
        for i, j in tqdm(
            list(zip(i_s, j_s)),
            desc="Calculating correlation matrix.",
        ):
            param_i = params[i]
            param_j = params[j]
            vals_i = idata.posterior[param_i].values
            vals_j = idata.posterior[param_j].values
            for subj in range(n_subjects):
                a = np.ravel(vals_i[:, :, subj])
                b = np.ravel(vals_j[:, :, subj])
                r = spearmanr(a, b)[0]
                corrs[subj, i, j] = r
                corrs[subj, j, i] = r
        corr_mat = np.mean(corrs, axis=0)
        names = [format_param_name(param) for param in params]
        fig.add_heatmap(
            z=corr_mat,
            x=names,
            y=names,
            coloraxis="coloraxis1",
            col=i_experiment + 1,
            row=i_model + 1,
        )
fig = fig.update_coloraxes(cmid=0)
fig.show()
