"""Controlled raw-ensemble plant-accounting experiments on the frozen 200 split."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .calibrated_model_benchmark import _eligible_hp_ids, _policy_profile_ids
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .force_off_policy_evaluation import evaluate_force_off_policy
from .routed_latent_targeted_loss_ablation import _load_context, _score_variant, _train_variant
from .trainer import train


BASE = Path("output/neural_building_emulator_refactor/buffer_capacity_sweep_current_best_200/cap_0p5h")
OUTPUT = Path("output/neural_building_emulator_refactor/plant_accounting_current_best_200")


def recipe_for_variant(base, variant):
    recipe = copy.deepcopy(base)
    settings = recipe["probabilistic_structured"]
    settings.update(hp_whole_building_accounting=True,
                    prob_plant_source_loss_weight=1.0, prob_plant_delivery_loss_weight=1.0,
                    prob_plant_block_hours=[4., 24.])
    if variant in ("physical", "physical_dhw"):
        settings.update(hp_buffer_capacity_mode="physical_volume", hp_energy_cap_wh_m2=0.,
                        hp_buffer_delta_t_usable_init_k=25., hp_buffer_delta_t_min_k=1.,
                        hp_buffer_delta_t_max_k=50.)
    elif variant != "accounting":
        raise ValueError(f"Unknown plant experiment {variant}")
    if variant == "physical_dhw":
        settings.update(prob_dhw_recovery_state=True, prob_dhw_mode_loss_weight=1.0)
    if variant == "physical":
        # Match the DHW arm's known inputs so its state/loss is the only extra mechanism.
        settings["hp_plant_metadata"] = True
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--variant", choices=("accounting", "physical", "physical_dhw"), default="accounting")
    parser.add_argument("--stages", nargs="+", choices=("train", "score", "force_off", "diagnose"),
                        default=["train", "score", "force_off", "diagnose"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, split, baseline_artifact = _load_context(args.base_root)
    metadata = json.loads((baseline_artifact / "metadata_closed_loop_hp_contracting_probabilistic.json").read_text())
    base["probabilistic_structured"] = metadata["train_config"]
    recipe = recipe_for_variant(base, args.variant)
    args.epochs = int(base["optimizer"]["epochs"])
    args.num_scenarios = int(base["evaluation"]["num_scenarios"])
    args.num_profile_plots = 3
    destination = args.output_dir / ("smoke" if args.smoke else "variants") / args.variant
    destination.mkdir(parents=True, exist_ok=True)
    request = {"recipe": recipe, "split": json.loads(json.dumps(asdict(split))), "baseline_artifact": str(baseline_artifact),
               "smoke": args.smoke, "native_only": True}
    path = destination / "experiment_request.json"
    if path.exists() and json.loads(path.read_text()) != request:
        raise ValueError("Experiment request changed; use a fresh output directory")
    path.write_text(json.dumps(request, indent=2, sort_keys=True))
    sources = [Path("neural_building_emulator") / name for name in (
        "train.py", "data.py", "columns.py", "scaling.py", "dataset_adapter.py", "plant_supervision.py",
        "models/probabilistic_contracting_closed_loop_hp.py", "models/dhw_recovery.py", "model_io.py")]
    sources.append(Path(__file__))
    if not (destination / "implementation_hashes.json").exists():
        (destination / "implementation_hashes.json").write_text(json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
    if args.smoke:
        recipe["sequence_length"] = 96
        recipe["stride"] = 35039
        recipe["optimizer"].update(batch_size=2, train_eval_max_windows=2)
        recipe["probabilistic_structured"].update(
            state_dim=4, hidden_dim=8, depth=1, prob_particles=2, prob_eval_particles=2,
            prob_plot_particles=2, prob_flex_kpi_crps_weight=0., prob_flex_kpi_crps_windows_per_profile=1,
            prob_targeted_flex_energy_weight=0.,
        )
        # Distinct buildings for the integration check; never used as an ablation result.
        retained = {*split.train_ids[:2], *split.validation_ids[:1]}
        configs = _experiment_configs(recipe, destination / "models", max_profiles=len(retained), epochs=1,
            fixed_test_profile_ids=split.validation_ids[:1],
            excluded_profile_ids=tuple(sorted(set(_eligible_hp_ids(Path(recipe["dataset_path"]))) - retained)))
        print(f"Smoke artifact: {train(configs['probabilistic_structured'])}", flush=True)
        return
    if "train" in args.stages:
        artifact = _train_variant(args, recipe, split, args.variant)
    else:
        record = json.loads((destination / "artifact.json").read_text())
        artifact = Path(record["artifact_dir"])
    if set(args.stages) & {"score", "force_off", "diagnose"}:
        fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(artifact.iterdir()) if p.suffix in (".eqx", ".npz", ".json")}
        fingerprint_path = destination / "scoring_checkpoint.json"
        if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()) != fingerprints:
            raise ValueError("Scoring checkpoint changed; do not reuse cached scores")
        fingerprint_path.write_text(json.dumps(fingerprints, indent=2, sort_keys=True))
    if "score" in args.stages:
        _score_variant(args, recipe, split, args.variant, artifact)
    if "force_off" in args.stages:
        forceoff = destination / "force_off_policy" / "test"
        if not (args.resume and (forceoff / "force_off_policy_summary.json").exists()):
            evaluate_force_off_policy(artifact, dataset_path=Path("data/control=force_off"),
                output_dir=forceoff, profile_ids=split.test_ids, num_scenarios=args.num_scenarios,
                num_eval_particles=int(recipe["probabilistic_structured"]["prob_eval_particles"]),
                hp_scenario_mode="bernoulli", ventilation_rollout_mode=recipe["evaluation"]["ventilation_rollout_mode"],
                kpi_horizons_hours=recipe["evaluation"]["horizons_hours"],
                kpi_min_events=recipe["evaluation"]["min_events"], num_profile_plots=args.num_profile_plots,
                profile_plot_ids=_policy_profile_ids(split.test_ids, policy="force_off", count=args.num_profile_plots),
                evaluate_ungated=False, seed=int(recipe["seed"]) + 2000)
    if "diagnose" in args.stages:
        from .plant_rollout_diagnostics import run as run_diagnostics

        run_diagnostics(argparse.Namespace(
            artifact=artifact, output=args.output_dir / "diagnostics" / args.variant,
            dataset=Path("data/control=force_off"),
            events=Path("output/neural_building_emulator_refactor/buffer_energy_diagnostics_200/events.csv"),
            particles=args.num_scenarios, seed=int(recipe["seed"]) + 2000,
            profile_ids=list(split.test_ids), plot_ids=[11152319, 11230131, 191952443]))


if __name__ == "__main__":
    main()
