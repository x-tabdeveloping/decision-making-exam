def plot_effects(idata, true_params):
    fig = go.Figure()
    color_scheme = px.colors.qualitative.Pastel
    for i_effect, effect in enumerate(["stop_lr_effect", "stop_inv_t_effect"]):
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
        )
        fig.add_scatter(
            x=[effect_name],
            y=[true_params[effect]],
            marker=dict(
                size=22,
                symbol="diamond-wide-dot",
                color=color_scheme[i_effect],
                line=dict(width=2, color="black"),
            ),
            showlegend=False,
        )
    return fig


plot_effects(idata, params)


q_true = trace_qs(
    c=choices,
    r=rewards,
    q0s=np.zeros((n_subjects, n_arms)),
    learning_rates=true_params["lr"],
)
records = []
for trial, q_trial in enumerate(q_true):
    for subject, (q_sub, stop) in enumerate(zip(q_trial, stop_condition)):
        for arm, q in enumerate(q_sub):
            records.append(
                dict(
                    expected_value=q,
                    arm=arm,
                    subject=subject,
                    trial=trial,
                    stop=stop,
                    gold=True,
                )
            )
q_pred = trace_qs(
    c=choices,
    r=rewards,
    q0s=np.zeros((n_subjects, n_arms)),
    learning_rates=mean_posterior["lr"],
)
for trial, q_trial in enumerate(q_pred):
    for subject, q_sub in enumerate(q_trial):
        for arm, q in enumerate(q_sub):
            records.append(
                dict(
                    expected_value=q, arm=arm, subject=subject, trial=trial, gold=False
                )
            )
df = pd.DataFrame.from_records(records)
fig = px.line(
    df,
    x="trial",
    y="expected_value",
    color="arm",
    facet_row="subject",
    facet_col="gold",
)
fig.show()
