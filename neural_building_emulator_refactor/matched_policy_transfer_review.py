"""Interpret a completed frozen transfer benchmark alongside excitation audits."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .generic_response_report import native_scores
from .kpi_policy_transfer import _html_table
from .matched_policy_transfer import METHODS, SOURCE, digest, read_json, verify_hashes, write_json
from .probabilistic_calibration import summarize_joint_distribution_tables


LABELS = {"structured": "Structured 0.5h", "continuous": "Vanilla LSTM",
          "hurdle_markov": "Markov LSTM", "lightgbm": "LightGBM"}


def run(output):
    completion = read_json(output / "completion.json")
    verify_hashes(completion["hashes"])
    request = read_json(output / "request.json")
    verify_hashes(request["hashes"])
    reports = output / "reports"
    macro = pd.read_csv(reports / "macro_metrics.csv")
    source = pd.read_csv(reports / "source_calibration_distributions.csv")
    distributions = {name: pd.read_csv(reports / f"{name}_test_distributions.csv")
                     for name in completion["branches"]}
    original_test = pd.read_csv(reports / "source_test_distributions.csv")
    keys = ["profile_id", "kpi_kind", "horizon_hours", "direction", "sample_index"]
    transfer = original_test[original_test.method.eq("lightgbm")].merge(
        distributions["raw"][distributions["raw"].method.eq("lightgbm")],
        on=keys, suffixes=("_source", "_target"), validate="one_to_one")
    transfer["prediction_change"] = (transfer.prediction_target - transfer.prediction_source).abs()
    prediction_changes = transfer.groupby("profile_id").prediction_change.max().reset_index()
    prediction_changes.to_csv(reports / "lightgbm_prediction_changes.csv", index=False)
    unchanged_predictions = int(prediction_changes.prediction_change.lt(1e-10).sum())
    joint = summarize_joint_distribution_tables(source, distributions, kpi_kinds=("f",))
    joint.to_csv(reports / "direct_f_joint_metrics.csv", index=False)
    differences = pd.read_csv(output / "domain_differences.csv")
    changed = set(differences.loc[(differences.fold == "test") & differences.field.eq(
        "gardenCompatibility.values.hp_ref_capacity_W"), "profile_id"].astype(int))
    sensitivity = []
    for name, frame in distributions.items():
        subset = frame[frame.kpi_kind.eq("f") & ~frame.profile_id.isin(changed)]
        summary = native_scores(subset)[0]
        sensitivity.append(summary.assign(calibration=name, buildings=subset.profile_id.nunique()))
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    sensitivity.to_csv(reports / "unchanged_hp_sizing_direct_f.csv", index=False)
    identification = pd.read_csv(output / "kpi_identifiability_summary.csv")
    if len(identification) != 8:
        raise ValueError("Incomplete source/target/fold/control identifiability audit")
    tariff_rank = identification[identification.dataset.eq("tariff") & identification.controls.eq("full")]
    full_rank = bool(tariff_rank.min_directional_rank.eq(2).all())
    warning = ("The tariff regression directions are identifiable in this audit." if full_rank else
               "WARNING: separate up/down regression coefficients are not uniquely identified "
               "under the tariff schedule and the existing full controls. Their scores below are "
               "comparisons of the existing least-squares convention, not identified physical responses.")

    reliability = pd.read_csv(reports / "reliability.csv")
    figure = make_subplots(rows=2, cols=2, subplot_titles=[LABELS[m] for m in METHODS])
    colors = {"raw": "#246fa8", "source_rns": "#a34526", "target_rns": "#277a48"}
    for i, method in enumerate(METHODS):
        row, col = i // 2 + 1, i % 2 + 1
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#888", dash="dot"),
                                   showlegend=False), row=row, col=col)
        for branch, color in colors.items():
            frame = reliability[reliability.method.eq(method) & reliability.kpi_kind.eq("f")
                                & reliability.calibration.eq(branch)]
            curve = frame.groupby("quantile").empirical_coverage.mean()
            figure.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines+markers", name=branch,
                                       legendgroup=branch, showlegend=i == 0, line=dict(color=color)), row=row, col=col)
    figure.update_layout(title="Direct-F reliability: target RNS is adaptation", template="plotly_white", height=750)
    figure.update_xaxes(title="Nominal quantile", range=[0, 1])
    figure.update_yaxes(title="Observed fraction below quantile", range=[0, 1])
    figure.write_html(reports / "direct_f_reliability.html", include_plotlyjs=True)

    direct = macro[macro.kpi_kind.eq("f")]
    score_table = direct.pivot(index="method", columns="calibration", values="crps").reindex(METHODS)
    score_table = score_table[["raw", "source_rns", "target_rns"]].reset_index()
    paired = pd.read_csv(reports / "paired_crps.csv")
    paired = paired[paired.kpi_kind.eq("f") & paired.calibration.isin(colors)]
    coverage = direct[direct.calibration.isin(colors)][
        ["method", "calibration", "bias", "coverage_90", "width_90", "mean_pit"]]
    annual = pd.read_csv(reports / "annual_signal_metrics.csv")
    temporal = []
    for method in METHODS[:-1]:
        summary = read_json(output / f"scoring/test/{method}/summary.json")
        rows = summary["temporal_trace_calibration"]
        if {row["signal"] for row in rows} != {"Tin", "Qroom", "Pel_SH"}:
            raise ValueError(f"Incomplete native trace calibration for {method}")
        temporal.extend(dict(row, model=method) for row in rows)
    temporal = pd.DataFrame(temporal)
    temporal.to_csv(reports / "native_trace_calibration.csv", index=False)
    inputs = pd.read_csv(output / "deterministic_input_audit.csv")
    inputs = inputs.groupby("field").agg(
        changed_buildings=("fraction_different", lambda x: int((x > 0).sum())),
        max_abs_diff=("max_abs_diff", "max")).reset_index()
    dhw = differences[differences.field.eq("inputs.building.dhwSetpoint")]
    plots = [(method, path) for method in METHODS[:-1]
             for path in sorted((output / f"scoring/test/{method}").glob("profile_*.html"))]
    links = "".join(f'<li>{LABELS[method]}: <a href="../scoring/test/{method}/{escape(path.name)}">'
                    f'{escape(path.stem)}</a></li>' for method, path in plots)
    (reports / "review.html").write_text(f'''<!doctype html><meta charset="utf-8">
    <title>Tariff transfer: results and validity checks</title><style>
    body{{font:15px system-ui;line-height:1.5;margin:32px;color:#222;max-width:1400px}}
    table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}
    .warning{{border-left:4px solid #b34b2b;padding-left:16px}}</style>
    <h1>Tariff transfer: results and validity checks</h1>
    <p>Frozen structured 0.5-hour, vanilla LSTM, Markov LSTM and metadata LightGBM.
    Same {completion['test_buildings']} test buildings, {completion['calibration_buildings']} calibration buildings,
    {completion['particles']} annual particles. No retraining or path modification.</p>
    <p class="warning"><strong>{escape(warning)}</strong></p>
    <p>The tariff events are all +/-1 C. With an intercept, the two event columns already have one
    linear dependency; the pre-event setpoint control explains the remaining direction. More events
    do not resolve this. Direct-F remains an identifiable descriptive event statistic, not a claim
    of a policy-independent causal effect. All requested regression scores are retained in the
    <a href="index.html">full numerical report</a>.</p>
    <h2>Direct-F CRPS: lower is better</h2>{_html_table(score_table)}
    <p>Source RNS uses only the original identification-policy calibration fold. Target RNS fits
    the tariff calibration fold and is <strong>adaptation, not zero-shot transfer</strong>.
    RNS-ECC has exactly the same marginal CRPS, PIT and coverage as RNS.</p>
    <h2>Direct-F calibration and sharpness</h2>{_html_table(coverage)}
    <p><a href="direct_f_reliability.html">Reliability curves</a>. Nominal central coverage: 90%.
    Wider intervals can improve coverage without improving CRPS.</p>
    <h2>Paired direct-F CRPS differences versus structured</h2>{_html_table(paired)}
    <p>95% intervals bootstrap whole buildings (2,000 draws). Negative differences favor the candidate.</p>
    <h2>Direct-F joint scores: eight coordinates</h2>{_html_table(joint)}
    <p>Normalization is fixed by source calibration truths. ECC retains raw ranks without generating
    physical traces. LightGBM's raw ensemble shares Gaussian quantile indices and has no learned joint copula.</p>
    <h2>Native annual trace accuracy</h2>{_html_table(annual)}<ul>{links}</ul>
    <p>The plotted confidence bands are native emulator uncertainty, not KPI-calibrated bands.</p>
    <h2>Native trace calibration</h2>{_html_table(temporal[["model", "signal", "scope", "mean_crps",
        "empirical_central_90_coverage", "mean_central_90_width"]])}
    <p>Trace statistics pool timestamps within the stated scope; they are not independent observations
    for confidence intervals. KPI recalibration does not repair these native trace distributions.</p>
    <h2>Transfer is not a pure policy intervention</h2>
    <p>LightGBM has no schedule input: its distributions are unchanged for {unchanged_predictions} of
    {len(prediction_changes)} test buildings. Changes in the remaining buildings reflect metadata,
    not an explicit response to the tariff schedule.</p>
    <p>DHW nominal setpoint metadata differ in {dhw.profile_id.nunique()} held-out buildings;
    HP and buffer sizing differ in {len(changed)} test buildings: {escape(str(sorted(changed)))}.
    Some newer plant-design fields are absent in the older metadata; absence alone does not prove
    a physical difference. Full differences: ../domain_differences.csv.</p>{_html_table(inputs)}
    <h2>Sensitivity: exclude changed HP sizing</h2>{_html_table(sensitivity)}
    <p>This removes sizing changes, not the DHW configuration mismatch shared by all buildings.</p>
    <h2>Regression identifiability audit</h2>{_html_table(identification)}
    <p>Directional rank is computed after projecting both event columns onto all nuisance controls.
    Source rank 2 identifies both coefficients; rank 0 identifies neither. Per-building details:
    ../kpi_identifiability.csv. The benchmark estimator itself was not modified.</p>
    <p>For a clean policy-transfer claim, regenerate tariff data with the source physical/DHW configuration
    held fixed. Use direct-F and trace/energy metrics here; separately revisit the regression estimand
    for a binary policy. One training seed and a previously inspected holdout limit generalization claims.</p>''')
    write_json(output / "review_completion.json", dict(
        implementation_sha256=digest(Path(__file__)), excluded_test_sizing_ids=sorted(changed),
        full_control_directional_identifiability=full_rank, annual_plots=len(plots),
        hashes={str(path.resolve()): digest(path) for path in (
            reports / "review.html", reports / "direct_f_reliability.html", reports / "direct_f_joint_metrics.csv",
            reports / "unchanged_hp_sizing_direct_f.csv", reports / "lightgbm_prediction_changes.csv",
            reports / "native_trace_calibration.csv",
            output / "kpi_identifiability.csv",
            output / "kpi_identifiability_summary.csv", output / "deterministic_input_audit.csv")},
    ))
    print(score_table.to_string(index=False))
    print(f"review={reports / 'review.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "tariff_policy_transfer")
    run(parser.parse_args().output)
