"""Loss-only screen on the frozen effective-storage baseline; no new plant states."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from .calibrated_model_benchmark import _eligible_hp_ids, _policy_profile_ids
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .force_off_policy_evaluation import evaluate_force_off_policy
from .plant_accounting_experiment import BASE
from .routed_latent_targeted_loss_ablation import _load_context, _score_variant, _train_variant
from .trainer import train


OUTPUT = Path("output/neural_building_emulator_refactor/generic_response_current_best_200")
VARIANT = "response_loss"


def response_recipe(base):
    recipe = copy.deepcopy(base)
    settings = recipe["probabilistic_structured"]
    settings.update(
        prob_event_energy_weight=1.0,
        prob_event_energy_include_force_off=True,
        prob_event_energy_horizons_hours=[0.5, 1.0, 2.0, 3.0],
        prob_event_energy_events_per_profile=4,
        prob_block_energy_weight=1.0,
        prob_block_energy_horizons_hours=[6.0, 24.0, 72.0, 240.0],
        prob_block_energy_horizons_per_batch=1,
    )
    return recipe


def fingerprints(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in (".json", ".npz", ".eqx")}


def require_same(path, content):
    if path.exists() and json.loads(path.read_text()) != content:
        raise ValueError(f"Provenance changed: {path}; use a fresh output directory")
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def run(args):
    base, split, baseline = _load_context(args.base_root)
    metadata = json.loads((baseline / "metadata_closed_loop_hp_contracting_probabilistic.json").read_text())
    base["probabilistic_structured"] = metadata["train_config"]
    recipe = response_recipe(base)
    args.epochs = int(base["optimizer"]["epochs"])
    args.num_scenarios = int(base["evaluation"]["num_scenarios"])
    args.num_profile_plots = 3
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
        recipe["sequence_length"] = 96
        recipe["stride"] = 35039
        recipe["optimizer"].update(batch_size=2, train_eval_max_windows=2)
        recipe["probabilistic_structured"].update(
            state_dim=4, hidden_dim=8, depth=1, prob_particles=2, prob_eval_particles=2,
            prob_plot_particles=2, prob_flex_kpi_crps_weight=0.,
            prob_flex_kpi_crps_windows_per_profile=1, prob_targeted_flex_energy_weight=0.,
        )
    destination = args.output_dir / "variants" / VARIANT
    destination.mkdir(parents=True, exist_ok=True)
    require_same(destination / "experiment_request.json", {
        "recipe": recipe, "split": json.loads(json.dumps(asdict(split))),
        "baseline_artifact": str(baseline), "baseline_hashes": fingerprints(baseline),
        "native_only": True, "smoke": args.smoke,
        "scope": "Loss-only: joint command responses plus block-average scores. "
                 "No architecture/storage changes, no new observed plant labels. "
                 "240-hour windows are not full-year training; evaluation is continuous annual.",
    })
    sources = [Path("neural_building_emulator/train.py"), Path(__file__),
               Path("neural_building_emulator_refactor/train.py"),
               Path("neural_building_emulator/models/probabilistic_contracting_closed_loop_hp.py")]
    require_same(destination / "implementation_hashes.json",
                 {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if args.smoke:
        retained = {*split.train_ids[:2], *split.validation_ids[:1]}
        configs = _experiment_configs(
            recipe, destination / "models", max_profiles=len(retained), epochs=1,
            fixed_test_profile_ids=split.validation_ids[:1],
            excluded_profile_ids=tuple(sorted(set(_eligible_hp_ids(Path(recipe["dataset_path"]))) - retained)),
        )
        print(f"Smoke artifact: {train(configs['probabilistic_structured'])}", flush=True)
        return
    state = args.output_dir / "compute_state.json"

    def progress(stage):
        state.write_text(json.dumps({"status": stage, "updated_unix": time.time(),
                                     "variant": VARIANT, "native_only": True}, indent=2))
        print(f"stage={stage}", flush=True)

    if "train" in args.stages:
        progress("training")
        artifact = _train_variant(args, recipe, split, VARIANT)
    else:
        artifact = Path(json.loads((destination / "artifact.json").read_text())["artifact_dir"])
    require_same(destination / "scoring_checkpoint.json", fingerprints(artifact))
    if "score" in args.stages:
        progress("setpoint_scoring")
        _score_variant(args, recipe, split, VARIANT, artifact)
    if "force_off" in args.stages:
        progress("force_off_scoring")
        output = destination / "force_off_policy" / "test"
        request = {
            "artifact": str(artifact.resolve()), "hashes": fingerprints(artifact),
            "profile_ids": list(split.test_ids), "particles": args.num_scenarios,
            "seed": int(recipe["seed"]) + 2000, "evaluation": recipe["evaluation"],
        }
        output.mkdir(parents=True, exist_ok=True)
        require_same(output / "experiment_request.json", request)
        if not (args.resume and (output / "force_off_policy_summary.json").exists()):
            evaluate_force_off_policy(
                artifact, dataset_path=Path("data/control=force_off"), output_dir=output,
                profile_ids=split.test_ids, num_scenarios=args.num_scenarios,
                num_eval_particles=int(recipe["probabilistic_structured"]["prob_eval_particles"]),
                hp_scenario_mode="bernoulli",
                ventilation_rollout_mode=recipe["evaluation"]["ventilation_rollout_mode"],
                kpi_horizons_hours=recipe["evaluation"]["horizons_hours"],
                kpi_min_events=recipe["evaluation"]["min_events"], num_profile_plots=3,
                profile_plot_ids=_policy_profile_ids(split.test_ids, policy="force_off", count=3),
                evaluate_ungated=False, seed=request["seed"],
            )
    if "diagnose" in args.stages:
        progress("event_diagnostics")
        from .plant_rollout_diagnostics import run as diagnose

        diagnose(argparse.Namespace(
            artifact=artifact, output=args.output_dir / "diagnostics" / VARIANT,
            dataset=Path("data/control=force_off"),
            events=Path("output/neural_building_emulator_refactor/buffer_energy_diagnostics_200/events.csv"),
            particles=args.num_scenarios, seed=int(recipe["seed"]) + 2000,
            profile_ids=list(split.test_ids), plot_ids=[11152319, 11230131, 191952443]))
    progress("requested_stages_finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stages", nargs="+", choices=("train", "score", "force_off", "diagnose"),
                        default=["train", "score", "force_off", "diagnose"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
