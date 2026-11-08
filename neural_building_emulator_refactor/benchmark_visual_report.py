"""Build a read-only visual review from the completed campaign's saved scores.

Run with the repository's .venv/bin/python. No model inference or fitting occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT: Path
ROOT: Path
POLICIES = ["setpoint_excitation_v2", "force_off_excitation_v2", "tariff_setpoint_v2", "tariff_force_off_v2"]
MODELS = ["baseline", "two_tanks", "continuous", "hurdle_markov"]
METHODS = MODELS + ["lightgbm"]
SIGNALS = {"Tin": "temperature", "Qroom": "qroom", "Pel_SH": "pel"}
LABELS = {"baseline": "Structured baseline", "two_tanks": "Structured + two tanks", "continuous": "Vanilla LSTM", "hurdle_markov": "LSTM + Markov HP", "lightgbm": "LightGBM"}
COLORS = {"baseline": "#3776b6", "two_tanks": "#d77c27", "continuous": "#2a9d79", "hurdle_markov": "#8057b6", "lightgbm": "#737c8d"}
PLABELS = {"setpoint_excitation_v2": "Excitation · setpoint", "force_off_excitation_v2": "Excitation · force-off", "tariff_setpoint_v2": "Tariff · setpoint", "tariff_force_off_v2": "Tariff · force-off"}
SOURCES: dict[str, dict] = {}
TABLES: dict[str, list[pd.DataFrame]] = {}
split: dict
TEST_IDS: set


def track(path):
    path = Path(path)
    stat = path.stat()
    entry = dict(path=str(path), resolved_path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size < 20_000_000:
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path)] = entry
    return path


def read(path, **kw):
    return pd.read_csv(track(path), **kw)


def append(name, frame, **tags):
    TABLES.setdefault(name, []).append(frame.assign(**tags))


def policy_root(policy):
    return ROOT if "excitation" in policy else ROOT / "transfer" / policy


def kinds(policy):
    return ["force_off_regression", "force_off_f"] if "force_off" in policy else ["regression_based", "f"]


def boot(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    # Resample buildings, never individual correlated timestamps or KPI cells.
    indices = np.random.default_rng(1357).integers(0, len(values), (2000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def rank_summary(frame, groups):
    rows = []
    for key, part in frame.groupby(groups, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        part = part.sort_values("quantile")
        row = dict(zip(groups, key))
        for nominal in [.5, .8, .9]:
            lo, hi = (1 - nominal) / 2, (1 + nominal) / 2
            row[f"coverage_{round(nominal * 100)}"] = float(np.interp(hi, part["quantile"], part.empirical_coverage) - np.interp(lo, part["quantile"], part.empirical_coverage))
        row["calibration_mae"] = float(part.abs_calibration_error.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def load_traces():
    links = []
    for policy in POLICIES:
        for method in MODELS:
            off = "force_off" in policy
            dest = policy_root(policy) / "variants" / method / ("force_off" if off else "scoring") / "test"
            tags = dict(policy=policy, method=method)
            scores = read(dest / "temporal_trace_probabilistic_scores.csv")
            reliability = read(dest / "temporal_trace_calibration.csv")
            hist = read(dest / "temporal_trace_rank_histogram.csv")
            assert set(scores.signal) == set(SIGNALS)
            assert scores.profile_count.eq(100).all() and scores.min_scenarios_per_profile.eq(100).all()
            assert reliability.groupby("signal").size().eq(19).all()
            assert np.allclose(hist.groupby("signal").probability.sum(), 1)
            summary = scores.merge(rank_summary(reliability, ["signal"]), on="signal", validate="one_to_one")
            if off:
                profile = read(dest / "force_off_profile_metrics.csv")
                assert set(profile.profile_id) == TEST_IDS
                append("force_off_profile_metrics", profile, **tags)
                profile = profile[profile.period.eq("all") & profile.signal.isin(SIGNALS)].copy()
            else:
                original = read(dest / "profile_metrics.csv")
                assert len(original) == 100 and set(original.profile_id) == TEST_IDS
                parts = []
                for signal, prefix in SIGNALS.items():
                    columns = [c for c in original if c.startswith(prefix + "_") and c not in [prefix + "_c_rmse", prefix + "_c_mae", prefix + "_c_bias"]]
                    columns = [c for c in columns if c.removeprefix(prefix + "_") in ["rmse", "mae", "bias", "nrmse", "profile_std_nrmse", "wasserstein_norm", "acf_mae", "increment_spectral_js"]]
                    parts.append(original[["profile_id", *columns]].rename(columns={c: c.removeprefix(prefix + "_") for c in columns}).assign(signal=signal))
                profile = pd.concat(parts, ignore_index=True)
                sj = json.loads(track(dest / "summary.json").read_text())
                append("hp_dynamics_summary", pd.DataFrame([sj["pel_dynamics"]]), **tags)
                append("qroom_summary", pd.DataFrame([sj["qroom_calibration"]]), **tags)
                for stem in ["pel_mode_autocorrelation", "pel_run_duration_distribution", "pel_transition_reliability", "pel_annual_energy_calibration", "pel_conditional_calibration_summary", "qroom_occurrence_reliability"]:
                    append(stem, read(dest / (stem + ".csv")), **tags)
            accuracy = []
            for signal, part in profile.groupby("signal"):
                row = dict(signal=signal, buildings=len(part))
                for metric in ["rmse", "mae", "bias", "acf_mae", "increment_spectral_js", "wasserstein_norm"]:
                    if metric in part:
                        row[metric] = float(part[metric].mean())
                        if metric == "rmse":
                            row["rmse_lo"], row["rmse_hi"] = boot(part[metric])
                accuracy.append(row)
            summary = summary.merge(pd.DataFrame(accuracy), on="signal", validate="one_to_one")
            append("trace_summary", summary, **tags)
            append("trace_by_building", profile, **tags)
            append("trace_reliability", reliability, **tags)
            append("trace_histogram", hist, **tags)
            if off:
                regime = read(dest / "temporal_trace_probabilistic_scores_by_regime.csv")
                rel = read(dest / "temporal_trace_calibration_by_regime.csv")
                regime = regime.merge(rank_summary(rel, ["regime", "signal"]), on=["regime", "signal"], validate="one_to_one")
                append("regime_summary", regime, **tags)
                append("regime_reliability", rel, **tags)
            plot_dir = dest / "probabilistic_traces" if off else dest
            for p in sorted(plot_dir.glob("*profile_*.html")):
                links.append(dict(**tags, file=os.path.relpath(p, OUT), title=p.name))
            print("loaded traces", policy, method, flush=True)
    return links


def load_kpis():
    cache = {}
    for policy in POLICIES:
        folder = policy_root(policy) / "reports"
        if folder not in cache:
            cache[folder] = {name: read(folder / (name + ".csv")) for name in ["distribution_metrics", "distribution_by_building", "distribution_reliability", "native_pit", "joint_distribution_metrics", "paired_crps"]}
        data = cache[folder]
        metrics = data["distribution_metrics"][data["distribution_metrics"].kpi_kind.isin(kinds(policy))].copy()
        buildings = data["distribution_by_building"][data["distribution_by_building"].kpi_kind.isin(kinds(policy))].copy()
        # Verify identical marginal scores for RNS and RNS-ECC rather than drawing two apparent improvements.
        check = metrics.set_index(["method", "kpi_kind", "horizon_hours", "direction", "calibration"])
        left = check.xs("rns", level="calibration").sort_index()
        right = check.xs("rns_ecc", level="calibration").sort_index()
        pd.testing.assert_frame_equal(left, right)
        assert set(metrics.method) == set(METHODS) and metrics["count"].eq(100).all()
        assert set(buildings.profile_id) == TEST_IDS
        expected = 8 if "setpoint" in policy else 4
        assert metrics.groupby(["method", "kpi_kind", "calibration"]).size().eq(expected).all()
        for _, part in buildings.groupby(["kpi_kind", "horizon_hours", "direction", "calibration"]):
            truth = part.pivot(index="profile_id", columns="method", values="truth")
            assert np.allclose(truth, truth.iloc[:, [0]], atol=1e-6, rtol=1e-6)
        append("kpi_cells", metrics, policy=policy)
        rel = data["distribution_reliability"]
        append("kpi_reliability", rel[rel.kpi_kind.isin(kinds(policy))], policy=policy)
        pit = data["native_pit"]
        append("kpi_native_pit", pit[pit.kpi_kind.isin(kinds(policy))], policy=policy)
        family = "force_off" if "force_off" in policy else "setpoint"
        append("joint_scores", data["joint_distribution_metrics"].query("policy_family == @family"), policy=policy)
        pair = data["paired_crps"]
        append("native_paired_crps", pair[pair.kpi_kind.isin(kinds(policy))], policy=policy)
        rows = []
        for (method, kind, calibration), part in metrics.groupby(["method", "kpi_kind", "calibration"]):
            row = dict(method=method, kpi_kind=kind, calibration=calibration, cells=len(part))
            row.update(part[["rmse", "mae", "bias", "r2", "spearman", "crps", "calibration_mae", "coverage_50", "coverage_80", "coverage_90", "coverage_95", "width_50", "width_80", "width_90", "width_95"]].mean().to_dict())
            selected = buildings[(buildings.method == method) & (buildings.kpi_kind == kind) & (buildings.calibration == calibration)]
            avg = selected.groupby("profile_id").crps.mean().sort_index()
            assert len(avg) == 100 and np.isclose(avg.mean(), row["crps"])
            row["crps_lo"], row["crps_hi"] = boot(avg)
            baseline = buildings[(buildings.method == "baseline") & (buildings.kpi_kind == kind) & (buildings.calibration == calibration)].groupby("profile_id").crps.mean().sort_index()
            delta = avg - baseline
            row["crps_delta_baseline"] = float(delta.mean())
            row["delta_lo"], row["delta_hi"] = boot(delta)
            rows.append(row)
        append("kpi_summary", pd.DataFrame(rows), policy=policy)
        print("loaded KPIs", policy, flush=True)


def load_event_curves():
    # Streaming reduction of saved event tables; no scenarios regenerated.
    cache_path = OUT / "event_curves.csv"
    cache_meta = OUT / "event_curves_sources.json"
    files = [(p, m, policy_root(p) / "variants" / m / "force_off/test/force_off_event_responses.csv") for p in POLICIES if "force_off" in p for m in MODELS]
    signatures = {str(f): [f.stat().st_size, f.stat().st_mtime_ns] for _, _, f in files}
    for _, _, path in files:
        track(path)
    if cache_path.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()) == signatures:
        return pd.read_csv(cache_path)
    rows = []
    keys = ["profile_id", "model", "signal", "lag_hours"]
    for policy, method, path in files:
        sums = []
        for part in pd.read_csv(path, chunksize=400_000):
            sums.append(part.groupby(keys, observed=True).value.agg(["sum", "count"]))
        grouped = pd.concat(sums).groupby(level=list(range(4))).sum()
        grouped["mean"] = grouped["sum"] / grouped["count"]
        profiles = grouped.reset_index()
        for (source, signal, lag), part in profiles.groupby(["model", "signal", "lag_hours"]):
            # Every building gets the same weight, regardless of its number of events.
            rows.append(dict(policy=policy, method=method, source=source, signal=signal, lag_hours=float(lag), mean=float(part["mean"].mean()), buildings=len(part), events=int(part["count"].sum())))
        print("reduced force-off event curves", policy, method, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    cache_meta.write_text(json.dumps(signatures, indent=2))
    return result


def records(df):
    return json.loads(df.to_json(orient="records"))


def static_report(data):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
    traces = pd.DataFrame(data["trace_summary"])
    kpis = pd.DataFrame(data["kpi_summary"])
    with PdfPages(OUT / "benchmark_summary.pdf") as pdf:
        for metric, title, suffix in [("rmse", "Trace accuracy · mean building RMSE", "native_trace_accuracy"), ("mean_crps", "Native trace CRPS · lower is better", "native_trace_crps"), ("coverage_90", "Native trace calibration · central rank 90% coverage", "native_trace_coverage")]:
            fig, axes = plt.subplots(4, 3, figsize=(12.5, 11), constrained_layout=True)
            for i, policy in enumerate(POLICIES):
                for j, signal in enumerate(SIGNALS):
                    ax = axes[i, j]
                    sub = traces[(traces.policy == policy) & (traces.signal == signal)].set_index("method").loc[MODELS]
                    x = np.arange(len(MODELS)); values = sub[metric].to_numpy()
                    if metric == "coverage_90": values *= 100
                    ax.bar(x, values, color=[COLORS[m] for m in MODELS], width=.7)
                    for a, b in zip(x, values): ax.annotate(f"{b:.1f}" if metric == "coverage_90" else f"{b:.3g}", (a, b), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
                    ax.set_xticks(x, ["Base", "2 tanks", "LSTM", "Markov"], fontsize=8)
                    if metric == "coverage_90": ax.axhline(90, color="#222", ls="--", lw=1); ax.set_ylim(0, 108)
                    else: ax.set_ylim(0, max(values) * 1.2)
                    if i == 0: ax.set_title(signal + (" (%)" if metric == "coverage_90" else " (°C)" if signal == "Tin" else " (W/m²)"), fontweight="bold")
                    if j == 0: ax.set_ylabel(PLABELS[policy], fontsize=9)
                    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
            fig.suptitle(title + "\n100 matched test buildings · 100 annual particles · selected from 15 epochs", fontsize=14)
            fig.supxlabel("RMSE: full year. CRPS / rank coverage: Tin full year; Qroom and Pel during heating availability. No trace recalibration.", fontsize=9)
            pdf.savefig(fig); fig.savefig(OUT / (suffix + ".png")); plt.close(fig)
        for family in ["f", "regression"]:
            fig, axes = plt.subplots(4, 3, figsize=(13.5, 11), constrained_layout=True)
            for i, policy in enumerate(POLICIES):
                kind = kinds(policy)[1 if family == "f" else 0]
                for j, (metric, label) in enumerate([("crps", "CRPS ↓"), ("rmse", "Mean cell RMSE ↓"), ("coverage_90", "90% interval coverage (%)")]):
                    ax = axes[i, j]
                    for n, method in enumerate(METHODS):
                        sub = kpis[(kpis.policy == policy) & (kpis.kpi_kind == kind) & (kpis.method == method)].set_index("calibration")
                        a, b = float(sub.loc["raw", metric]), float(sub.loc["rns_ecc", metric])
                        if metric == "coverage_90": a *= 100; b *= 100
                        ax.plot([n, n], [a, b], color=COLORS[method], lw=2)
                        ax.scatter(n, a, s=42, facecolors="white", edgecolors=COLORS[method], zorder=3)
                        ax.scatter(n, b, s=30, color=COLORS[method], zorder=4)
                    ax.set_xticks(range(5), ["Base", "2 tanks", "LSTM", "Markov", "LGBM"], fontsize=8)
                    if metric == "coverage_90": ax.axhline(90, color="#222", ls="--", lw=1); ax.set_ylim(0, 105)
                    else: ax.set_ylim(bottom=0)
                    if i == 0: ax.set_title(label, fontweight="bold")
                    if j == 0: ax.set_ylabel(PLABELS[policy])
                    ax.grid(axis="y", alpha=.18)
            fig.suptitle(("Direct-F" if family == "f" else "Regression-based flexibility") + " · raw ○ → RNS–ECC ●", fontsize=15)
            fig.supxlabel("Equal weight per horizon / direction. Source-calibration fit only. Setpoint: Wh/(m² K); force-off: Wh/(m² intervention).\nLightGBM is a metadata-only KPI baseline; it has no temporal policy inputs. RNS and RNS–ECC have identical marginals.", fontsize=9)
            pdf.savefig(fig); fig.savefig(OUT / ("kpi_" + family + "_overview.png")); plt.close(fig)


def main():
    global ROOT, OUT, split, TEST_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed four-model benchmark with 100 test buildings.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to RUN_DIR/visual_summary.")
    args = parser.parse_args()
    ROOT = args.run_dir.resolve()
    OUT = args.output_dir.resolve() if args.output_dir else ROOT / "visual_summary"
    split = json.loads((ROOT / "split_manifest.json").read_text())
    TEST_IDS = set(split["test_ids"])
    SOURCES.clear()
    TABLES.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    links = load_traces()
    load_kpis()
    frames = {name: pd.concat(values, ignore_index=True) for name, values in TABLES.items()}
    frames["event_curves"] = load_event_curves()
    for policy, part in frames["event_curves"].query("source == 'simulation'").groupby("policy"):
        truth = part.pivot(index=["signal", "lag_hours"], columns="method", values="mean")
        assert np.allclose(truth, truth.iloc[:, [0]], atol=1e-7, rtol=1e-7), policy
    for name, frame in frames.items():
        frame.to_csv(OUT / (name + ".csv"), index=False)
    # Do not embed tens of thousands of raw KPI ranks. A per-policy/model/kind histogram suffices.
    pit = frames["kpi_native_pit"]
    histrows = []
    for (policy, method, kind), part in pit.groupby(["policy", "method", "kpi_kind"]):
        counts, edges = np.histogram(part.pit, bins=np.linspace(0, 1, 11))
        histrows.extend(dict(policy=policy, method=method, kpi_kind=kind, center=float((edges[i]+edges[i+1])/2), probability=float(c/len(part))) for i, c in enumerate(counts))
    data = {name: records(frame) for name, frame in frames.items() if name not in ["kpi_native_pit", "force_off_profile_metrics"]}
    data["kpi_pit_histogram"] = histrows
    data["links"] = links
    data["labels"] = LABELS
    data["policyLabels"] = PLABELS
    data["colors"] = COLORS
    data["policies"] = POLICIES
    data["models"] = MODELS
    data["methods"] = METHODS
    data["generated"] = datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds")
    data["status"] = json.loads((ROOT / "status.json").read_text())
    data["epochs"] = {m: json.loads((ROOT / "variants" / m / "training_status.json").read_text()) for m in MODELS}
    data["folds"] = {k: len(split[k + "_ids"]) for k in ["train", "validation", "calibration", "test"]}
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "summary_data.json").write_text(payload)
    template = (Path(__file__).resolve().parent / "viewer_assets" / "benchmark_dashboard.html").read_text()
    (OUT / "index.html").write_text(template.replace("__REPORT_DATA__", payload))
    from plotly.offline import get_plotlyjs
    (OUT / "plotly.min.js").write_text(get_plotlyjs())
    static_report(data)
    provenance = dict(generated=data["generated"], sources=list(SOURCES.values()), checks=dict(matched_test_buildings=100, trace_particles=100, policies=4, neural_models=4, kpi_models=5, rns_ecc_marginals_equal=True, source_truth_matches_all_models=True), methods=dict(accuracy="Mean per-building RMSE/MAE/bias of ensemble-mean annual trajectories", trace_probability="Existing empirical CRPS; tie-aware randomized ensemble-rank calibration; Tin all timestamps, Qroom/Pel heating availability", kpi="Equal weight per horizon/direction cell; point accuracy uses ensemble median", intervals="2000 building bootstrap draws; correlated timestamps and horizon cells are not independent units", temporal="Setpoint ACF/spectral/Wasserstein are for ensemble-mean traces; HP mode/run distributions use sampled paths. Force-off event curves average events within each building then buildings equally.", calibration="RNS and ECC use excitation calibration buildings only; neither modifies time traces; identical marginal scores by construction."))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("REPORT_READY", OUT / "index.html", flush=True)


if __name__ == "__main__":
    main()
