"""Collect saved annual Plotly traces into an offline, lazily loaded explorer.

No inference is performed: the exported means and 2.5/97.5 percentiles are reused.
Missing source plots fail explicitly rather than substituting another checkpoint.

Example::

    python -m neural_building_emulator_refactor.annual_trace_explorer \
        --run-dir output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15 \
        --dataset-dir data/garden-ticino-five-cities-1000-v9-final
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


MODELS = {
    "baseline": "Structured baseline",
    "two_tanks": "Structured · two tanks",
    "continuous": "Vanilla LSTM",
    "hurdle_markov": "LSTM · Markov HP",
}
POLICIES = (
    "setpoint_excitation_v2", "force_off_excitation_v2",
    "tariff_setpoint_v2", "tariff_force_off_v2",
)
CHANNELS = {
    "Tin": "Indoor temperature",
    "Qroom": "Delivered room heat",
    "Pel_SH": "HP electric power",
}


def read_plot(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    match = re.search(r'Plotly\.newPlot\(\s*"[^"\n]+"\s*,\s*', source)
    if not match:
        raise ValueError(f"No serialized Plotly figure in {path}")
    data, _ = json.JSONDecoder().raw_decode(source, match.end())
    return data, hashlib.sha256(raw).hexdigest()


def array(value) -> np.ndarray:
    if isinstance(value, dict) and "bdata" in value:
        value = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    result = np.asarray(value, dtype="<f4")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Expected a finite one-dimensional saved trace")
    return result


def packed(value: np.ndarray) -> str:
    return base64.b64encode(value.astype("<f4", copy=False).tobytes()).decode("ascii")


def intervals(mask: np.ndarray, start: int, step: int) -> list[list[int]]:
    """Half-open command intervals, including a final active reporting interval."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[start + int(a) * step, start + int(b) * step]
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def plot_directory(run: Path, policy: str, model: str) -> Path:
    base = run / "transfer" / policy if policy.startswith("tariff") else run
    suffix = "scoring/test" if "setpoint" in policy else "force_off/test/probabilistic_traces"
    return base / "variants" / model / suffix


def plot_paths(run: Path, policy: str, model: str) -> dict[str, Path]:
    pattern = (r"profile_\d+_(\d+)\.html" if "setpoint" in policy
               else r"force_off_profile_\d+_(\d+)_structured_gated\.html")
    paths = {}
    for path in sorted(plot_directory(run, policy, model).glob("*.html")):
        match = re.fullmatch(pattern, path.name)
        if match:
            if match[1] in paths:
                raise ValueError(f"Duplicate source for {policy}/{model}/{match[1]}")
            paths[match[1]] = path
    return paths


def write_payload(path: Path, key: str, value: dict) -> None:
    # Script loading also works from file://, without fetch/CORS or a web server.
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False)
    path.write_text(f"window.registerAnnualTrace({json.dumps(key)},{payload});\n")


def build(run: Path, dataset: Path, output: Path, egids: list[str] | None) -> None:
    inventories = {(p, m): plot_paths(run, p, m) for p in POLICIES for m in MODELS}
    available = set.intersection(*(set(x) for x in inventories.values()))
    selected = sorted(set(egids) if egids else available, key=int)
    if not selected:
        raise ValueError("No buildings have saved plots under all policies and models")
    missing = [(p, m, e) for (p, m), paths in inventories.items() for e in selected if e not in paths]
    if missing:
        raise FileNotFoundError(f"Generate these selected-checkpoint annual plots first: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    assets = Path(__file__).with_name("viewer_assets")
    shutil.copyfile(assets / "annual_trace_explorer.html", output / "index.html")
    shutil.copyfile(assets / "annual_trace_explorer.js", output / "viewer.js")
    (output / "plotly.min.js").write_text(get_plotlyjs())

    manifest = {
        "run": run.name, "models": MODELS, "egids": selected,
        "policies": list(POLICIES), "interval": [0.025, 0.975],
        "default_egid": "11107352" if "11107352" in selected else selected[0],
        "entries": {}, "common": {}, "artifacts": {},
    }
    for model in MODELS:
        artifact = json.loads((run / "variants" / model / "artifact.json").read_text())
        manifest["artifacts"][model] = artifact
    audits = []
    for policy in POLICIES:
        for egid in selected:
            reference = None
            command_path = dataset / "data" / f"control={policy}" / f"egid={egid}" / "timeseries.parquet"
            commands = pd.read_parquet(command_path, columns=["datetime", "zone", "hp_force_off_requested"])
            commands = commands.loc[commands.zone.eq("building_total")].sort_values("datetime")
            command_index = pd.DatetimeIndex(commands.datetime)
            if not command_index.is_unique:
                raise ValueError(f"Duplicate command timestamps: {command_path}")
            for model in MODELS:
                source = inventories[(policy, model)][egid]
                traces, digest = read_plot(source)
                named = {t["name"]: t for t in traces if t.get("name")}
                x = pd.DatetimeIndex(named["Simulated Indoor temperature"]["x"])
                times = x.to_numpy(dtype="datetime64[ms]").astype(np.int64)
                step = int(times[1] - times[0])
                if len(x) != 35039 or step != 900000 or not np.all(np.diff(times) == step):
                    raise ValueError(f"Unexpected annual sampling in {source}")
                mean, lower, upper, truth = {}, {}, {}, {}
                for channel, name in CHANNELS.items():
                    truth[channel] = array(named[f"Simulated {name}"]["y"])
                    mean[channel] = array(named[f"Emulated {name}"]["y"])
                    upper_trace = named[f"95% {name}"]
                    index = next(i for i, t in enumerate(traces) if t is upper_trace)
                    lower_trace = traces[index - 1]
                    if lower_trace.get("name") or upper_trace.get("fill") != "tonexty":
                        raise ValueError(f"Unknown interval encoding in {source}")
                    lower[channel], upper[channel] = array(lower_trace["y"]), array(upper_trace["y"])
                    for trace in (named[f"Simulated {name}"], named[f"Emulated {name}"], lower_trace, upper_trace):
                        if trace["x"] != traces[0]["x"] or len(array(trace["y"])) != len(x):
                            raise ValueError(f"Mismatched channel timestamps in {source}")
                    if np.any(lower[channel] > upper[channel]):
                        raise ValueError(f"Inverted interval in {source}")
                setpoint = array(named["Heating setpoint"]["y"])
                command = commands.set_index("datetime").reindex(x).hp_force_off_requested.to_numpy(dtype=float)
                if not np.isfinite(command).all() or not np.isin(command, [0, 1]).all():
                    raise ValueError(f"Missing/invalid command alignment in {source}")
                if "HP force-off request" in named:
                    np.testing.assert_array_equal(array(named["HP force-off request"]["y"]), command)
                elif np.any(command):
                    raise ValueError(f"Unrepresented force-off command in {source}")
                common = {
                    "start_ms": int(times[0]), "step_ms": step, "count": len(x),
                    "truth": {c: packed(v) for c, v in truth.items()}, "setpoint": packed(setpoint),
                    "force_off_intervals": intervals(command > 0.5, int(times[0]), step),
                    "force_off_hours": float(command.sum() * step / 3600000),
                }
                common_key = f"{policy}/{egid}"
                if reference is None:
                    reference = common
                    filename = f"data/{policy}_{egid}_observed.js"
                    write_payload(output / filename, common_key, common)
                    manifest["common"][common_key] = filename
                elif reference != common:
                    raise ValueError(f"Observed traces/commands differ across models: {policy}/{egid}")
                key = f"{policy}/{model}/{egid}"
                filename = f"data/{policy}_{model}_{egid}.js"
                forecast = {"mean": {c: packed(v) for c, v in mean.items()},
                            "lower": {c: packed(v) for c, v in lower.items()},
                            "upper": {c: packed(v) for c, v in upper.items()}}
                write_payload(output / filename, key, forecast)
                manifest["entries"][key] = {
                    "file": filename, "source": quote(os.path.relpath(source, output)),
                    "source_sha256": digest,
                }
                audits.append({"key": key, "samples": len(x), "source": str(source), "source_sha256": digest,
                               "command_events": len(common["force_off_intervals"]),
                               "command_hours": common["force_off_hours"]})
            print(f"collected {policy} / {egid}: all four models", flush=True)
    public_manifest = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False)
    (output / "manifest.js").write_text(f"window.ANNUAL_TRACE_MANIFEST={public_manifest};\n")
    (output / "build_audit.json").write_text(json.dumps({
        "run": str(run), "dataset": str(dataset), "trace_count": len(audits),
        "egids": selected, "inference_performed": False, "quantiles": [0.025, 0.975],
        "sampling": "original 15-minute samples; no resampling or smoothing",
        "shading": "requested force-off; half-open [start, end) intervals; power panel only",
        "checks": ["finite arrays", "ordered quantiles", "annual regular timestamps",
                   "identical observed traces across models", "commands match dataset timestamps"],
        "traces": audits,
    }, indent=2) + "\n")
    print(f"Ready: {output / 'index.html'} ({len(audits)} annual views)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--egids", nargs="+")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    build(run, args.dataset_dir.resolve(), args.output_dir.resolve() if args.output_dir else run / "annual_trace_explorer", args.egids)


if __name__ == "__main__":
    main()
