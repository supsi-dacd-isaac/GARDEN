"""Compare frozen-checkpoint recharge diagnostics without promoting oracles."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .buffer_recharge_diagnostics import (
    DHW, PEL, QROOM, REQUEST, TIN, SETPOINT, eligible, load_signals,
)


def paired_bootstrap(raw: pd.DataFrame, candidate: pd.DataFrame, *, seed: int = 13) -> list[dict]:
    joined = raw.merge(candidate, on=["building", "event"], suffixes=("_raw", "_candidate"), validate="one_to_one")
    if len(joined) != len(raw) or len(joined) != len(candidate):
        raise ValueError("Arms must contain identical events")
    rows = []
    rng = np.random.default_rng(seed)
    for regime, mask in (("all", np.ones(len(joined), dtype=bool)),
                         ("dhw", joined.dhw_first_hour_raw),
                         ("no_dhw", ~joined.dhw_first_hour_raw)):
        group = joined.loc[mask]
        for metric in ("drop_crps", "brier_gt_0.5", "drop_coverage90"):
            delta = (group[f"{metric}_candidate"].astype(float) - group[f"{metric}_raw"].astype(float))
            means = delta.groupby(group.building).mean().to_numpy()
            draws = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
            rows.append({"regime": regime, "metric": metric, "buildings": len(means),
                         "difference": means.mean(), "lower95": np.quantile(draws, .025),
                         "upper95": np.quantile(draws, .975)})
    return rows


def summaries(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for regime, group in (("all", frame), ("dhw", frame[frame.dhw_first_hour]),
                          ("no_dhw", frame[~frame.dhw_first_hour])):
        rows.append({
            "regime": regime, "events": len(group), "buildings": group.building.nunique(),
            **{key: group[key].mean() for key in (
                "observed_drop_c", "mean_trace_drop_c", "mean_particle_drop_c", "drop_crps",
                "drop_coverage90", "pre_energy_wh_m2", "minimum_energy_wh_m2",
                "fraction_particles_energy_below_1", "p_drop_gt_0.5", "brier_gt_0.5",
            )},
            "observed_drop_rate_gt_0.5": (group.observed_drop_c > .5).mean(),
            "drop_width90": (group.drop_upper90 - group.drop_lower90).mean(),
        })
    return rows


def write_trace(dataset: Path, arms: dict[str, Path], building: int, output: Path, events: pd.DataFrame) -> None:
    signals, metadata = load_signals(dataset, building)
    signals = signals.iloc[1:]
    dates = signals.index
    values = metadata["gardenCompatibility"]["values"]
    area = values["floor_area"] * values["totalFloors"]
    targets = np.column_stack([
        signals[TIN], signals[QROOM], signals[PEL] * (1 - signals[DHW]) / area,
    ])
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
                           subplot_titles=("Indoor temperature [C]", "Room heat [W/m2]", "SH electricity [W/m2]", "Observed operating states (diagnostic labels)"))
    colors = {"raw": "#c62828", "oracle_dhw": "#00897b", "oracle_qroom": "#7b1fa2"}
    labels = {"raw": "Raw checkpoint", "oracle_dhw": "Observed-DHW oracle (not deployable)",
              "oracle_qroom": "Observed-Qroom oracle (not deployable)"}
    for channel in range(3):
        figure.add_trace(go.Scatter(x=dates, y=targets[:, channel], name="EnergyPlus",
                                    legendgroup="sim", showlegend=channel == 0,
                                    line=dict(color="#1565c0", width=1.5)), row=channel + 1, col=1)
    figure.add_trace(go.Scatter(x=dates, y=signals[SETPOINT], name="Room setpoint",
                                line=dict(color="#616161", dash="dot")), row=1, col=1)
    for name, folder in arms.items():
        with np.load(folder / "ensemble_cache" / f"{building}.npz") as saved:
            paths = saved["predictions"]
        mean = paths.mean(axis=0)
        lower, upper = np.quantile(paths, [.05, .95], axis=0)
        for channel in range(3):
            if name == "raw":
                figure.add_trace(go.Scatter(x=dates, y=lower[:, channel], mode="lines",
                                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                                            legendgroup=name), row=channel + 1, col=1)
                figure.add_trace(go.Scatter(x=dates, y=upper[:, channel], mode="lines", fill="tonexty",
                                            fillcolor="rgba(198,40,40,0.15)", line=dict(width=0),
                                            name="Raw 90% interval", showlegend=channel == 0,
                                            legendgroup=name, hoverinfo="skip"), row=channel + 1, col=1)
            figure.add_trace(go.Scatter(x=dates, y=mean[:, channel], name=labels[name],
                                        line=dict(color=colors[name], width=1), legendgroup=name,
                                        showlegend=channel == 0), row=channel + 1, col=1)
        if name == "raw":
            for particle in range(min(5, len(paths))):
                figure.add_trace(go.Scatter(x=dates, y=paths[particle, :, 0], name=f"Raw particle {particle}",
                                            visible="legendonly", line=dict(width=.7)), row=1, col=1)
    figure.add_trace(go.Scatter(x=dates, y=signals[REQUEST], name="Requested lockout",
                                line=dict(color="#455a64", shape="hv")), row=4, col=1)
    figure.add_trace(go.Scatter(x=dates, y=(signals[DHW] > .5) & (signals[PEL] > 10),
                                name="Observed active DHW", line=dict(color="#ef6c00", shape="hv")), row=4, col=1)
    buttons = [dict(label="Annual", method="relayout", args=[{f"xaxis{suffix}.autorange": True for suffix in ("", "2", "3", "4")}])]
    for _, event in events[events.building == building].nlargest(3, "drop_c").iterrows():
        start = signals.index[max(0, int(event.start) - 25)]
        stop = signals.index[min(len(signals) - 1, int(event.end) + 23)]
        buttons.append(dict(label=str(event.datetime)[:16], method="relayout",
                            args=[{f"xaxis{suffix}.range": [str(start), str(stop)] for suffix in ("", "2", "3", "4")}]))
    figure.update_layout(title=f"EGID {building}: frozen buffer diagnosis; oracle arms are NOT forecasts",
                         height=1100, template="plotly_white", hovermode="x unified",
                         legend=dict(orientation="h", y=1.12), margin=dict(t=180),
                         updatemenus=[dict(buttons=buttons, x=1, y=1.07, xanchor="right")])
    figure.write_html(output / f"profile_{building}.html", include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--oracle-dhw", type=Path, required=True)
    parser.add_argument("--oracle-qroom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-ids", type=int, nargs="+", default=[11152319, 11230131, 191952443])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = {"raw": args.raw, "oracle_dhw": args.oracle_dhw, "oracle_qroom": args.oracle_qroom}
    frames = {name: pd.read_csv(path / "ensemble_events.csv") for name, path in arms.items()}
    requests = {name: json.loads((path / "request.json").read_text()) for name, path in arms.items()}
    for name, request in requests.items():
        for key in ("dataset", "split_manifest", "artifact", "particles", "seed", "ensemble_splits"):
            if request[key] != requests["raw"][key]:
                raise ValueError(f"Unmatched {key} for {name}")
    summary = pd.DataFrame([{**row, "arm": name} for name, frame in frames.items() for row in summaries(frame)])
    summary.to_csv(args.output / "ensemble_summary.csv", index=False)
    paired = pd.DataFrame([{**row, "arm": name} for name in ("oracle_dhw", "oracle_qroom")
                           for row in paired_bootstrap(frames["raw"], frames[name])])
    paired.to_csv(args.output / "paired_building_bootstrap.csv", index=False)
    events = eligible(pd.read_csv(args.raw / "events.csv"))
    for building in args.plot_ids:
        write_trace(Path(requests["raw"]["dataset"]), arms, building, args.output, events)
        print(f"Wrote trace {building}", flush=True)
    columns = ["arm", "regime", "events", "observed_drop_c", "mean_trace_drop_c", "mean_particle_drop_c",
               "drop_crps", "drop_coverage90", "p_drop_gt_0.5", "observed_drop_rate_gt_0.5", "brier_gt_0.5"]
    conditional = pd.read_csv(args.raw / "conditional_summary.csv")
    matched = pd.read_csv(args.raw / "matched_building_bootstrap.csv")
    links = " ".join(f'<a href="profile_{building}.html">EGID {building}</a>' for building in args.plot_ids)
    scores = summary[summary.regime == "all"].set_index("arm").drop_crps
    dhw_improvement = 100 * (1 - scores["oracle_dhw"] / scores["raw"])
    qroom_improvement = 100 * (1 - scores["oracle_qroom"] / scores["raw"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Buffer recharge diagnosis</title>
    <style>body{{font:15px system-ui;margin:32px;max-width:1400px}}table{{border-collapse:collapse;font-size:13px}}
    th,td{{padding:7px;border:1px solid #ddd;text-align:right}}h2{{margin-top:32px}}a{{margin-right:18px}}</style></head><body>
    <h1>Buffer recharge diagnosis</h1><p>Frozen minimum-global-capacity checkpoint. {requests['raw']['particles']} continuous annual paths per held-out building;
    identical random keys across arms. No retraining or ex-post calibration. Oracle arms use simulator outputs and are not deployable.</p>
    <p>Events are complete requested interruptions in November-March with constant room setpoint and two uncensored post-release hours.
    Drop = preceding one-hour mean temperature minus minimum temperature through those two hours, clipped at zero.
    Successes and failures are both retained. Mode groups use actual DHW operation in the first post-release hour: diagnostic, not forecast covariates.</p>
    <p>Matched strata: building, month, exact duration, and within-building preceding-load tertile. These are associations, not a causal intervention.
    Bootstrap intervals resample buildings. Particle-level drops and the drop of the ensemble mean are different quantities.</p>
    <h2>Decision</h2><p>DHW recovery is strongly associated with simulator cooling, including after matching observable event conditions.
    It is not primarily a disappearance of sharp individual drops when averaging particles. However, correct-DHW gating improves overall
    event CRPS by only {dhw_improvement:.2f}%, versus {qroom_improvement:.2f}% when observed room heat drives the thermal dynamics.
    A recharge-only model change is therefore deferred: the evidence points more strongly to usable heat delivery, while leaving
    capacity, stratification and delivery-control contributions unresolved. No production model or checkpoint was changed.</p>
    <h2>Annual traces with raw 90% bands</h2><p>{links}</p>
    <h2>Simulator conditional behavior</h2>{conditional.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <h2>Matched DHW-minus-no-DHW associations</h2>{matched.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <h2>Event forecast scores</h2>{summary[columns].to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <h2>Candidate minus raw (equal building weights)</h2>{paired.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <h2>Limits</h2><p>The DHW oracle only suppresses wrongly allocated SH generation; it does not force correct SH restart or retrain the controller.
    The Qroom oracle supplies observed room heat to the existing thermal dynamics; its Qroom/Pel predictions remain autonomous.
    Neither oracle is an upper bound on what a retrained model could achieve. Tank-node temperatures are absent from the retained parquet,
    so total storage, stratification and delivery thresholds remain incompletely identified.</p>
    <h2>Provenance</h2><pre>{html.escape(json.dumps(requests, indent=2))}</pre></body></html>"""
    (args.output / "index.html").write_text(page)


if __name__ == "__main__":
    main()
