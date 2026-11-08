"""Reproducible Qroom-to-Tin stability and noise-conditioning benchmark.

Run ``python -m neural_building_emulator_refactor.thermal_noise_experiment``.
Stages run in separate processes; every epoch saves model AND Adam state.
The calibration fold remains unused. Annual tariff scoring never
selects checkpoints or tunes the models. All intervals are native ensembles.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path("output/neural_building_emulator_refactor/ticino_v9_thermal_four_models_1000_e30")
CAMPAIGN = Path("data/garden-ticino-five-cities-1000-v9-final")
COHORT = Path("output/neural_building_emulator_refactor/ticino_v9_full_benchmark_1000_fixed_e15/split_manifest.json")
POLICIES = ("setpoint_excitation_v2", "force_off_excitation_v2", "tariff_setpoint_v2", "tariff_force_off_v2")
VARIANTS = ("diagonal_exogenous", "diagonal_unconstrained_exogenous", "diagonal_state", "lstm_state")
LABELS = dict(diagonal_exogenous="Diagonal · exogenous noise",
              diagonal_unconstrained_exogenous="Diagonal without contraction · exogenous noise",
              diagonal_state="Diagonal · state-dependent noise", lstm_state="LSTM · state-dependent noise")
COLORS = dict(zip(VARIANTS, ("#2563eb", "#9333ea", "#d97706", "#0f8b70")))
FOLDS = ("train_ids", "validation_ids", "calibration_ids", "test_ids")
MAIN_METRICS = ("rmse", "mae", "bias", "crps", "coverage_90", "width_90")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def freeze(path, payload):
    payload = json.loads(json.dumps(payload))
    if Path(path).exists() and read(path) != payload:
        raise ValueError(f"Frozen experiment configuration changed: {path}")
    write(path, payload)


def status(root, **values):
    old = read(root / "status.json") if (root / "status.json").exists() else {}
    write(root / "status.json", {**old, **values, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")})


def input_columns():
    from neural_building_emulator.columns import closed_loop_input_columns
    full = list(closed_loop_input_columns(include_hp_force_off=True))
    keep = [i for i, name in enumerate(full) if name not in (
        full[0], "hp_force_off_requested", "input_dhw_mixed_water_l_per_h_m2_heated_area")]
    return full, keep, ["observed_qroom_w_m2"] + [full[i] for i in keep]


def request(args):
    return dict(campaign=str(args.campaign.resolve()), cohort=str(args.cohort.resolve()),
        cohort_sha256=digest(args.cohort), variants=list(getattr(args,"variants",VARIANTS)), seed=args.seed,
        epochs=args.epochs, sequence_length=args.sequence_length, stride=args.stride,
        batch_size=args.batch_size, train_particles=args.train_particles,
        eval_particles=args.eval_particles, annual_particles=args.annual_particles,
        learning_rate=.0003, gradient_clip_norm=1., bptt_truncate_steps=32,
        latent_dim=4, diagonal_state_dim=8, lstm_state_dim=32, encoded_dim=8, width=64,
        tau_min_hours=.1, tau_max_hours=504., dt_hours=.25,
        matched_pole_initialization=True,
        unconstrained_poles="Positive C1 tangent continuation above each initial pole parameter; no upper bound, clipping or stability penalty. Initial values and derivatives match the constrained model.",
        minimum_epochs=getattr(args,"minimum_epochs",15), early_stopping_patience=getattr(args,"patience",5),
        early_stopping_min_delta=1e-4,
        objective="Fair RMS trajectory energy score + variogram order 0.5 at lags 1,2,4,8 (weight 1).",
        noise_regularization=0., checkpoint_selection=f"Minimum excitation validation objective; shared early stopping, maximum {args.epochs} epochs; never test/tariff.",
        numerical_failure_policy="Record numerical training/annual failures explicitly; never drop failed particles or rank means over surviving buildings. Continue other model arms.",
        native_only=True, calibration_fold="Reserved and unused", prescribed_ventilation=True,
        input_columns=input_columns()[2], thermal_readout="Affine; no clipping or temperature feedback in diagonal arms.",
        noise_scope="Hidden state masked to zero for exogenous arm; identical parameter shapes and initial values in diagonal pair.",
        scope="Conditional thermal prediction with observed Qroom AND ventilation, not end-to-end policy transfer or flexibility prediction. Single-seed screen.",
        time_alignment="Interval-end Tin[t] from prior Tin[t-1] and observed interval-average Qroom[t]; first timestamp initialises only.")


def cache_path(root, policy, building):
    return root / "cache" / policy / f"{building}.npz"


def load_profile(campaign, policy, building):
    from neural_building_emulator.columns import CLOSED_LOOP_METADATA_COLUMNS, closed_loop_required_columns
    from neural_building_emulator.data import read_dataset_frame, to_closed_loop_profiles
    metadata = tuple(CLOSED_LOOP_METADATA_COLUMNS)
    columns = closed_loop_required_columns(metadata, include_internal_gains=True,
        include_dhw_request=True, include_hp_force_off=True)
    frame = read_dataset_frame(campaign / "data" / ("control=" + policy),
                               columns=columns, profile_ids=(building,))
    profile, = to_closed_loop_profiles(frame, include_space_heating_availability=True,
        include_internal_gains=True, include_dhw_request=True, include_hp_force_off=True,
        hp_power_area_normalization="building_heated_area", metadata_columns=metadata)
    full, keep, names = input_columns()
    if profile.inputs.shape[1] != len(full):
        raise ValueError("Input schema mismatch")
    values = dict(metadata=profile.metadata, inputs=np.column_stack((profile.targets[:, 1], profile.inputs[:, keep])),
        target=profile.targets[:, 0], initial=profile.initial_temperature[:, 0],
        datetime=profile.datetime.astype("datetime64[ns]").astype(np.int64),
        setpoint=profile.inputs[:, 0], force_off=profile.inputs[:, full.index("hp_force_off_requested")],
        available=profile.inputs[:, full.index("space_heating_available")])
    if len(values["target"]) != 35039:
        raise ValueError(f"Incomplete annual profile {policy}/{building}")
    if any(not np.isfinite(x).all() for x in values.values()):
        raise ValueError(f"Nonfinite data {policy}/{building}")
    if not np.all(np.diff(values["datetime"]) == 15*60*10**9):
        raise ValueError("Nonuniform time grid")
    np.testing.assert_allclose(values["initial"][1:], values["target"][:-1])
    return values


def prepare(args):
    root = args.output
    req = request(args)
    freeze(root / "comparison_request.json", req)
    split = read(args.cohort)
    counts=[len(split[k]) for k in FOLDS]
    if counts not in ([140,20,20,20], [701,100,99,100]):
        raise ValueError("Expected preserved 200- or 1000-building campaign folds")
    ids=[x for f in FOLDS for x in split[f]]
    if len(set(ids)) != len(ids) or set(ids)!=set(split["selected_ids"]):
        raise ValueError("Overlapping folds")
    campaign_split=read(args.campaign/"split_manifest.json")
    if any(not set(split[f]).issubset(campaign_split[f]) for f in FOLDS):
        raise ValueError("Building fold assignment differs from original campaign")
    freeze(root / "split_manifest.json", split)
    # Freeze source before processing, including data adapter and model helpers.
    source_hashes = {}
    for package in ("neural_building_emulator", "neural_building_emulator_refactor"):
        for p in Path(package).rglob("*.py"):
            source_hashes[str(p)] = digest(p)
            dest = root / "source_snapshot" / p
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
    freeze(root / "implementation_hashes.json", source_hashes)
    markers, building_metadata = {}, {}
    for policy in POLICIES:
        for building in split["selected_ids"]:
            folder = args.campaign / "data" / ("control="+policy) / f"egid={building}"
            complete = read(folder / "complete.json")
            if digest(folder / "simulation_metadata.json") != complete["simulationMetadataSha256"]:
                raise ValueError(f"Metadata hash mismatch: {folder}")
            physical = read(folder / "simulation_metadata.json")["gardenCompatibility"]["values"]
            if building in building_metadata and physical != building_metadata[building]:
                raise ValueError(f"Building metadata differs across policies: {building}")
            building_metadata[building] = physical
            markers[str(folder / "complete.json")] = digest(folder / "complete.json")
    freeze(root / "data_hashes.json", markers)
    records = []
    for policy in POLICIES:
        folds = ("train_ids", "validation_ids", "test_ids") if policy in POLICIES[:2] else ("test_ids",)
        records.extend((policy, fold, b) for fold in folds for b in split[fold])
    for i, (policy, fold, building) in enumerate(records):
        path = cache_path(root, policy, building)
        if not path.exists():
            values = load_profile(args.campaign, policy, building)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
        if (i+1) % 10 == 0:
            print(f"prepare profiles={i+1}/{len(records)}", flush=True)
            status(root, state="running", stage="prepare", progress=f"{i+1}/{len(records)} profiles")
            if (i+1)%50==0:report(args)
    # Only training buildings and excitation policies determine normalisation.
    metadata, total, square, count, target_sum, target_square = [], None, None, 0, 0., 0.
    for policy in POLICIES[:2]:
        for b in split["train_ids"]:
            with np.load(cache_path(root, policy, b)) as d:
                x, y = d["inputs"].astype(np.float64), d["target"].astype(np.float64)
                if policy == POLICIES[0]: metadata.append(d["metadata"])
            total = x.sum(0) if total is None else total+x.sum(0)
            square = (x*x).sum(0) if square is None else square+(x*x).sum(0)
            count += len(x); target_sum += y.sum(); target_square += (y*y).sum()
    m = np.asarray(metadata, np.float64)
    mean = total/count
    m_std=m.std(0);x_std=np.sqrt(np.maximum(square/count-mean**2, 0))
    scales = dict(metadata_mean=m.mean(0).tolist(), metadata_std=np.where(m_std>1e-5,m_std,1.).tolist(),
        input_mean=mean.tolist(), input_std=np.where(x_std>1e-5,x_std,1.).tolist(),
        target_mean=float(target_sum/count), target_std=float(np.sqrt(target_square/count-(target_sum/count)**2)))
    freeze(root / "scalers.json", scales)
    from neural_building_emulator.columns import CLOSED_LOOP_METADATA_COLUMNS
    freeze(root / "data_schema.json", dict(metadata_columns=CLOSED_LOOP_METADATA_COLUMNS,
        input_columns=req["input_columns"], target="Tin [C]", cached_profiles=len(records),
        training_normalisation_rows=count, calibration_profiles_loaded=0))
    write(root / "prepare_complete.json", dict(request_sha256=digest(root/"comparison_request.json"),
        cache_hashes={str(cache_path(root,p,b)):digest(cache_path(root,p,b)) for p,_,b in records}))
    report(args)


def verify(root):
    saved = read(root / "implementation_hashes.json")
    # Other concurrently edited experiment scripts are irrelevant to this run.
    relevant = [p for p in saved if p.startswith("neural_building_emulator/")]
    relevant += [__file__, "neural_building_emulator_refactor/models/thermal_noise.py"]
    for path in relevant:
        p = Path(path)
        key = str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p)
        if digest(p) != saved[key]:
            raise ValueError(f"Scientific source changed: {key}")


def normalised_profile(root, policy, building, scales):
    with np.load(cache_path(root, policy, building)) as f:
        d = {k:f[k] for k in f.files}
    m = ((d["metadata"]-scales["metadata_mean"])/scales["metadata_std"]).astype(np.float32)
    x = ((d["inputs"]-scales["input_mean"])/scales["input_std"]).astype(np.float32)
    y = ((d["target"]-scales["target_mean"])/scales["target_std"]).astype(np.float32)
    initial = ((d["initial"]-scales["target_mean"])/scales["target_std"]).astype(np.float32)
    return m, x, y, initial, d


def windows(root, req, fold):
    split, scales = read(root/"split_manifest.json"), read(root/"scalers.json")
    m_all, x_all, y_all, initial_all = [], [], [], []
    for policy in POLICIES[:2]:
        for b in split[fold]:
            m, x, y, initial, _ = normalised_profile(root, policy, b, scales)
            starts = list(range(0, len(x)-req["sequence_length"]+1, req["stride"]))
            if fold == "validation_ids":
                starts = [starts[i] for i in np.unique(np.linspace(0,len(starts)-1,4).astype(int))]
            for start in starts:
                end = start+req["sequence_length"]
                m_all.append(m); x_all.append(x[start:end].copy()); y_all.append(y[start:end].copy()); initial_all.append(initial[start])
    return tuple(np.asarray(a, dtype=np.float32) for a in (m_all,x_all,y_all,initial_all))


def make_model(req, scales, variant):
    import jax
    from .models.thermal_noise import ThermalNoiseModel
    return ThermalNoiseModel(len(scales["metadata_mean"]),len(scales["input_mean"]),
        architecture="lstm" if variant=="lstm_state" else "diagonal",
        noise_conditioning="exogenous" if variant.endswith("exogenous") else "state",
        transition_constraint="none" if variant in ("diagonal_unconstrained_exogenous","lstm_state") else "contractive",
        matched_pole_initialization=req.get("matched_pole_initialization",False),
        state_dim=req["lstm_state_dim"] if variant=="lstm_state" else req["diagonal_state_dim"],
        encoded_dim=req["encoded_dim"], latent_dim=req["latent_dim"], width=req["width"],
        truncate_steps=req["bptt_truncate_steps"], tau_min=req["tau_min_hours"], tau_max=req["tau_max_hours"],
        key=jax.random.PRNGKey(req["seed"]))


def save_eqx(path, tree):
    import equinox as eqx
    tmp = Path(str(path)+".tmp")
    eqx.tree_serialise_leaves(tmp, tree)
    tmp.replace(path)


def should_stop(history, req):
    """Validation-only patience, reconstructed exactly when resuming."""
    patience=req.get("early_stopping_patience",0)
    if not history or not patience or history[-1]["epoch"]<req.get("minimum_epochs",0):
        return False
    anchor=float("inf");last_improvement=0
    for row in history:
        if row["validation_loss"] < anchor-req.get("early_stopping_min_delta",0.):
            anchor=row["validation_loss"];last_improvement=row["epoch"]
    return history[-1]["epoch"]-last_improvement>=patience


def sampled_poles(model, metadata, key, particles=16):
    import jax
    import jax.numpy as jnp
    xi=jax.random.normal(key,(len(metadata),particles,model.latent_dim))
    return jax.vmap(lambda m,zs:jax.vmap(lambda z:model.coefficients(m,z)[0])(zs))(jnp.asarray(metadata),xi)


def train(args):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from .models.thermal_noise import distribution_loss
    root, variant = args.output, args.variant
    verify(root)
    req, scales = read(root/"comparison_request.json"), read(root/"scalers.json")
    folder = root / "variants" / variant
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"train_complete.json").exists(): return
    model = make_model(req,scales,variant)
    optim = optax.chain(optax.clip_by_global_norm(req["gradient_clip_norm"]), optax.adam(req["learning_rate"]))
    opt_state = optim.init(eqx.filter(model,eqx.is_inexact_array))
    history = read(folder/"history.json") if (folder/"history.json").exists() else []
    last_epoch, best, best_epoch = 0, float("inf"), 0
    if (folder/"resume.json").exists():
        resume = read(folder/"resume.json")
        model,opt_state = eqx.tree_deserialise_leaves(folder/resume["checkpoint"], (model,opt_state))
        last_epoch, best, best_epoch = resume["epoch"], resume["best_loss"], resume["best_epoch"]
        history = [r for r in history if r["epoch"]<=last_epoch]
    parameter_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model,eqx.is_inexact_array)))
    freeze(folder/"model_spec.json", dict(variant=variant, parameter_count=parameter_count,
        architecture=model.architecture, noise_conditioning=model.noise_conditioning,
        transition_constraint=model.transition_constraint,
        matched_pole_initialization=model.matched_pole_initialization,
        state_dim=model.state_dim, guaranteed_common_noise_contraction=variant=="diagonal_exogenous",
        gamma_upper=float(np.exp(-req["dt_hours"]/req["tau_max_hours"]))
            if model.architecture=="diagonal" and model.transition_constraint=="contractive" else None))
    print(f"Loading windows for {variant}; parameters={parameter_count}",flush=True)
    training,validation = windows(root,req,"train_ids"),windows(root,req,"validation_ids")
    probe_idx=np.random.default_rng(1321).choice(len(training[0]),min(160,len(training[0])),replace=False)
    probe=tuple(x[probe_idx] for x in training)
    print(f"windows train={len(training[0])} validation={len(validation[0])} sequence={req['sequence_length']}",flush=True)
    def loss_fn(model,batch,key,particles):
        m,x,y,initial=batch
        keys=jax.random.split(key,len(m))
        pred=jax.vmap(lambda mm,xx,yy,kk:model.sample(mm,xx,yy,kk,particles))(m,x,initial,keys)
        return distribution_loss(pred,y)
    @eqx.filter_jit
    def step(model,opt_state,batch,key):
        (loss,components),grads=eqx.filter_value_and_grad(loss_fn,has_aux=True)(model,batch,key,req["train_particles"])
        norm=optax.global_norm(grads)
        updates,next_opt=optim.update(grads,opt_state,eqx.filter(model,eqx.is_inexact_array))
        next_model=eqx.apply_updates(model,updates)
        return next_model,next_opt,loss,components,norm
    evaluate=eqx.filter_jit(lambda model,batch,key:loss_fn(model,batch,key,req["eval_particles"]))
    pole_evaluator=eqx.filter_jit(sampled_poles)
    pole_metadata=np.unique(validation[0],axis=0)
    def evaluate_windows(data,seed):
        rows=[]
        for i in range(0,len(data[0]),req["batch_size"]):
            batch=tuple(jnp.asarray(x[i:i+req["batch_size"]]) for x in data)
            value,components=evaluate(model,batch,jax.random.fold_in(jax.random.PRNGKey(seed),i))
            rows.append((len(batch[0]),float(value)))
        return sum(n*v for n,v in rows)/sum(n for n,_ in rows)
    started=time.time()
    for epoch in range(last_epoch+1,req["epochs"]+1):
        if should_stop(history,req):break
        epoch_start=time.time(); indices=np.random.default_rng(req["seed"]+epoch).permutation(len(training[0]))
        total=0.;count=0;max_norm=0.;clipped=0;skipped=0
        for batch_index,start in enumerate(range(0,len(indices),req["batch_size"]),1):
            idx=indices[start:start+req["batch_size"]]
            batch=tuple(jnp.asarray(x[idx]) for x in training)
            key=jax.random.fold_in(jax.random.PRNGKey(req["seed"]+epoch),batch_index)
            candidate,next_opt,loss,components,norm=step(model,opt_state,batch,key)
            loss,norm=float(loss),float(norm)
            if not np.isfinite(loss) or not np.isfinite(norm):
                skipped+=1
                if skipped>=3: raise FloatingPointError(f"Repeated nonfinite gradients in {variant} epoch {epoch}")
                continue
            model,opt_state=candidate,next_opt
            total+=loss*len(idx);count+=len(idx);max_norm=max(max_norm,norm);clipped+=norm>req["gradient_clip_norm"]
            if batch_index==1 or batch_index%10==0:
                print(f"batch_progress variant={variant} epoch={epoch} batch={batch_index} loss={loss:.6g} elapsed_min={(time.time()-epoch_start)/60:.2f}",flush=True)
                status(root,state="running",stage="train",variant=variant,epoch=epoch,
                       progress=f"batch {batch_index}/{int(np.ceil(len(indices)/req['batch_size']))}")
                report(args)
        print(f"validation_start variant={variant} epoch={epoch}",flush=True)
        status(root,state="running",stage="validation",variant=variant,epoch=epoch,progress="fixed validation and training windows")
        validation_loss=evaluate_windows(validation,9001)
        training_loss=evaluate_windows(probe,9002)
        if not np.isfinite(validation_loss): raise FloatingPointError("Nonfinite validation objective")
        if validation_loss<best:
            best,best_epoch=validation_loss,epoch
            save_eqx(folder/"selected.eqx",model)
        row=dict(epoch=epoch,training_loss=training_loss,online_training_loss=total/max(count,1),
            validation_loss=validation_loss,epoch_minutes=(time.time()-epoch_start)/60,
            max_gradient_norm=max_norm,clipped_batches=int(clipped),skipped_updates=skipped,
            selected_epoch=best_epoch)
        if model.architecture=="diagonal":
            rho=np.asarray(pole_evaluator(model,pole_metadata,jax.random.PRNGKey(178)))
            row.update(pole_max=float(rho.max()),pole_p99=float(np.quantile(rho,.99)),
                       noncontractive_particle_fraction=float(np.mean(np.max(rho,axis=-1)>=1.)))
        history.append(row)
        # Alternate complete checkpoints; the resume pointer is updated last.
        checkpoint=f"resume_{epoch%2}.eqx"
        save_eqx(folder/checkpoint,(model,opt_state))
        write(folder/"history.json",history)
        write(folder/"resume.json",dict(epoch=epoch,best_epoch=best_epoch,best_loss=best,checkpoint=checkpoint,
            rng="Deterministic seed+epoch and batch fold-in; permutation seed+epoch"))
        print(json.dumps(row),flush=True)
        report(args)
    write(folder/"train_complete.json",dict(completed_epochs=history[-1]["epoch"],selected_epoch=best_epoch,
        selected_validation_loss=best,selected_sha256=digest(folder/"selected.eqx"),
        stopping_reason="validation_patience" if should_stop(history,req) else "epoch_limit",
        invocation_minutes=(time.time()-started)/60))
    report(args)


def crps_array(samples,truth):
    """Empirical-ensemble CRPS, particle axis first; native evaluation score."""
    sorted_values=np.sort(samples,axis=0)
    n=len(samples);weights=(2*np.arange(1,n+1)-n-1).reshape((n,)+(1,)*(samples.ndim-1))
    return np.mean(np.abs(samples-np.asarray(truth)),axis=0)-np.sum(weights*sorted_values,axis=0)/(n*n)


def score_samples(samples,truth):
    # Preserve very large but finite unconstrained predictions in the scores;
    # float32 squared errors would overflow long before float64 does.
    samples=np.asarray(samples,dtype=np.float64)
    truth=np.asarray(truth,dtype=np.float64)
    mean=samples.mean(0); residual=mean-truth
    result=dict(rmse=float(np.sqrt(np.mean(residual**2))),mae=float(np.mean(np.abs(residual))),
                bias=float(np.mean(residual)),crps=float(np.mean(crps_array(samples,truth))),
                prediction_min=float(samples.min()),prediction_max=float(samples.max()))
    for level in (50,80,90,95):
        a=(1-level/100)/2
        lo,hi=np.quantile(samples,[a,1-a],axis=0)
        result[f"coverage_{level}"]=float(np.mean((truth>=lo)&(truth<=hi)))
        result[f"width_{level}"]=float(np.mean(hi-lo))
    pit=(np.sum(samples<truth,axis=0)+.5*np.sum(samples==truth,axis=0))/len(samples)
    result["pit_mean"]=float(pit.mean())
    for lag in (1,4,96):
        if len(truth)<=lag:continue
        result[f"increment_crps_{lag}"]=float(np.mean(crps_array(samples[:,lag:]-samples[:,:-lag],truth[lag:]-truth[:-lag])))
    return result,pit


def annual_figure(destination,raw,samples,variant,policy,building):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    dates=raw["datetime"].astype("datetime64[ns]")
    lo,hi=np.quantile(samples,[.05,.95],axis=0)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.09,
                      subplot_titles=("Indoor temperature · native 90% interval","Prescribed delivered room heat"))
    for y,name,color in [(lo,"90% lower","rgba(220,40,40,0)"),(hi,"90% interval","rgba(220,40,40,0)")]:
        fig.add_trace(go.Scatter(x=dates,y=y,name=name,mode="lines",line=dict(color=color),
            fill="tonexty" if name=="90% interval" else None,fillcolor="rgba(220,40,40,.14)",
            showlegend=name=="90% interval"),row=1,col=1)
    for y,name,color,dash in [(raw["target"],"EnergyPlus Tin","#2467ab",None),
        (samples.mean(0),"Ensemble mean Tin","#cf3030",None),
        (raw["setpoint"],"Setpoint (reference only)","#666","dot")]:
        fig.add_trace(go.Scatter(x=dates,y=y,name=name,mode="lines",line=dict(color=color,width=1.5,dash=dash)),row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=raw["inputs"][:,0],name="Observed Qroom input",line=dict(color="#2467ab")),row=2,col=1)
    fig.update_yaxes(title_text="°C",row=1,col=1);fig.update_yaxes(title_text="W/m²",row=2,col=1)
    fig.update_layout(title=f"{LABELS[variant]} · {policy} · {building}<br><sup>Observed Qroom and ventilation; continuous annual rollout; no temperature resets</sup>",
        height=850,template="plotly_white",hovermode="x unified",legend=dict(orientation="h",y=1.09),margin=dict(t=150))
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.write_html(destination,include_plotlyjs="../../../../plotly.min.js")


def evaluate(args):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import pandas as pd
    root,variant=args.output,args.variant
    verify(root)
    req,scales,split=read(root/"comparison_request.json"),read(root/"scalers.json"),read(root/"split_manifest.json")
    folder=root/"variants"/variant
    complete=read(folder/"train_complete.json")
    if digest(folder/"selected.eqx")!=complete["selected_sha256"]:raise ValueError("Selected checkpoint changed")
    model=eqx.tree_deserialise_leaves(folder/"selected.eqx",make_model(req,scales,variant))
    chunk_size=960
    @eqx.filter_jit
    def chunk(model,m,x,xi,eps,carry):
        return jax.vmap(lambda z,e,c:model.rollout(m,x,jnp.asarray(0.),z,e,carry=c))(xi,eps,carry)
    @eqx.filter_jit
    def get_poles(model,m,xi):
        return jax.vmap(lambda z:model.coefficients(m,z)[0])(xi)
    failed_partitions=[]
    for policy_index,policy in enumerate(POLICIES):
        destination=folder/"evaluation"/policy
        destination.mkdir(parents=True,exist_ok=True)
        rows=[];histograms=[];annual_rows=[]
        for building_index,building in enumerate(split["test_ids"]):
            score_file=destination/f"scores_{building}.json"
            sample_file=destination/f"samples_{building}.npz"
            if score_file.exists():
                saved=read(score_file)
                if sample_file.exists() or saved["metrics"].get("simulation_failed",0):
                    rows.append(saved["metrics"]);histograms.append(saved["pit"]);annual_rows.extend(saved["annual"])
                    continue
            m,x,y,initial,raw=normalised_profile(root,policy,building,scales)
            # Same building gets the same annual latent draws across policies;
            # paired structured arms also share the exact innovations.
            key=jax.random.fold_in(jax.random.PRNGKey(45013),building)
            latent_key,process_key=jax.random.split(key)
            xi=jax.random.normal(latent_key,(req["annual_particles"],model.latent_dim))
            pole_diagnostic={}
            if model.architecture=="diagonal":
                poles=np.asarray(get_poles(model,jnp.asarray(m),xi))
                pole_diagnostic=dict(poles=poles.tolist(),max_pole=float(poles.max()),
                    noncontractive_particle_fraction=float(np.mean(poles.max(axis=1)>=1.)))
            carry=jax.vmap(lambda z:model.initial_carry(jnp.asarray(m),z,jnp.asarray(initial[0])))(xi)
            pieces=[];max_state=0.;state_finite=True
            for block,start in enumerate(range(0,len(x),chunk_size)):
                n=min(chunk_size,len(x)-start)
                # The final carry must correspond to the actual year end.
                # Running padded timesteps could falsely label an otherwise
                # finite year as unstable in the unconstrained arm.
                xx=x[start:start+n]
                eps=jax.random.normal(jax.random.fold_in(process_key,block),
                    (req["annual_particles"],chunk_size,model.state_dim))[:,:n]
                carry,out=chunk(model,jnp.asarray(m),jnp.asarray(xx),xi,eps,carry)
                pieces.append(np.asarray(out)[:,:n])
                for leaf in jax.tree.leaves(carry):
                    array=np.asarray(leaf)
                    state_finite=state_finite and bool(np.isfinite(array).all())
                    if np.isfinite(array).all():max_state=max(max_state,float(np.max(np.abs(array))))
            samples=np.concatenate(pieces,axis=1).astype(np.float64)*scales["target_std"]+scales["target_mean"]
            failed_particles=np.flatnonzero(~np.isfinite(samples).all(axis=1)).tolist()
            diagnostic=dict(**pole_diagnostic,hidden_states_finite=state_finite,
                max_abs_hidden_state_at_chunk_boundaries=max_state,failed_particles=failed_particles)
            write(destination/f"stability_{building}.json",diagnostic)
            if failed_particles or not state_finite:
                metrics=dict.fromkeys(MAIN_METRICS)
                metrics.update(variant=variant,policy=policy,building=building,simulation_failed=1,
                    failed_particle_count=len(failed_particles))
                write(score_file,dict(metrics=metrics,pit=[0]*10,annual=[],
                    failure="Nonfinite annual state or prediction; entire building ensemble invalid, no particle filtering.",
                    stability=diagnostic))
                rows.append(metrics);histograms.append([0]*10)
                pd.DataFrame(rows).to_csv(destination/"profile_metrics.csv",index=False)
                print(f"annual_numerical_failure variant={variant} policy={policy} building={building} failed_particles={len(failed_particles)}",flush=True)
                report(args)
                continue
            metrics,pit=score_samples(samples,raw["target"])
            metrics.update(variant=variant,policy=policy,building=building,simulation_failed=0,failed_particle_count=0)
            heating=raw["available"]>.5
            if heating.any():
                hm,_=score_samples(samples[:,heating],raw["target"][heating])
                for k in ("rmse","bias","crps","coverage_90","width_90"):metrics["heating_"+k]=hm[k]
            # Realised annual summaries, not flexibility KPIs.
            annual=[]
            values=[("mean_tin_c",samples.mean(1),float(raw["target"].mean())),
                ("below_setpoint_degree_hours",(.25*np.maximum(raw["setpoint"]-samples,0)*heating).sum(1),
                 float((.25*np.maximum(raw["setpoint"]-raw["target"],0)*heating).sum()))]
            for name,ensemble,truth in values:
                lo,hi=np.quantile(ensemble,[.05,.95])
                annual.append(dict(variant=variant,policy=policy,building=building,kpi=name,
                    truth=truth,median=float(np.median(ensemble)),crps=float(crps_array(ensemble,truth)),
                    coverage_90=float(lo<=truth<=hi),width_90=float(hi-lo)))
            pit_hist=np.histogram(pit,bins=np.linspace(0,1,11))[0].tolist()
            storage_dtype=np.float32 if np.max(np.abs(samples))<np.finfo(np.float32).max else np.float64
            np.savez_compressed(sample_file,samples=samples.astype(storage_dtype),datetime=raw["datetime"],truth=raw["target"])
            if building_index<2:
                annual_figure(destination/f"profile_{building}.html",raw,samples,variant,policy,building)
            write(score_file,dict(metrics=metrics,pit=pit_hist,annual=annual))
            rows.append(metrics);histograms.append(pit_hist);annual_rows.extend(annual)
            print(f"annual_score variant={variant} policy={policy} building={building} {building_index+1}/{len(split['test_ids'])} rmse={metrics['rmse']:.4f} crps={metrics['crps']:.4f} coverage90={metrics['coverage_90']:.3f}",flush=True)
            status(root,state="running",stage="evaluate",variant=variant,progress=f"{policy}: {building_index+1}/{len(split['test_ids'])}")
            pd.DataFrame(rows).to_csv(destination/"profile_metrics.csv",index=False)
            report(args)
        pd.DataFrame(rows).to_csv(destination/"profile_metrics.csv",index=False)
        pd.DataFrame(annual_rows,columns=["variant","policy","building","kpi","truth","median","crps","coverage_90","width_90"]).to_csv(destination/"annual_temperature_kpis.csv",index=False)
        write(destination/"pit_histogram.json",np.asarray(histograms).sum(0).tolist())
        failed=[r["building"] for r in rows if r.get("simulation_failed",0)]
        if failed:failed_partitions.append(dict(policy=policy,buildings=failed))
        write(destination/"complete.json",dict(buildings=split["test_ids"],scenarios=req["annual_particles"],
            checkpoint_sha256=complete["selected_sha256"],native_only=True,failed_buildings=failed))
    write(folder/"evaluation_complete.json",dict(policies=list(POLICIES),numerical_failures=failed_partitions))
    report(args)


def aggregate_trace_scores(scores):
    """Never report optimistic aggregate scores after dropping failed buildings."""
    scores=scores.copy()
    if "simulation_failed" not in scores:scores["simulation_failed"]=0
    grouped=scores.groupby(["variant","policy"],sort=False)
    summary=grouped.mean(numeric_only=True).drop(columns="building")
    summary["buildings"]=grouped.size()
    summary["failed_buildings"]=grouped.simulation_failed.sum()
    metrics=[c for c in summary.columns if c not in ("buildings","failed_buildings")]
    summary.loc[summary.failed_buildings>0,metrics]=np.nan
    return summary


def report(args):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    req=read(root/"comparison_request.json") if (root/"comparison_request.json").exists() else {}
    variants=req.get("variants",list(getattr(args,"variants",VARIANTS)))
    split=read(root/"split_manifest.json") if (root/"split_manifest.json").exists() else read(args.cohort)
    total_buildings=len(split["selected_ids"])
    fold_counts=" / ".join(f"{len(split[k])} {label}" for k,label in zip(FOLDS,("training","validation","reserved calibration","test")))
    final_reports=all((root/"variants"/v/"evaluation_complete.json").exists() or
                      (root/"variants"/v/"training_failed.json").exists() for v in variants)
    if not (root/"plotly.min.js").exists(): (root/"plotly.min.js").write_text(get_plotlyjs())
    current=read(root/"status.json") if (root/"status.json").exists() else {}
    figures=[]
    fig=go.Figure()
    for v in variants:
        color=COLORS[v]
        p=root/"variants"/v/"history.json"
        if p.exists():
            rows=read(p)
            for key,dash,label in [("training_loss","solid","training probe"),("validation_loss","dash","validation")]:
                fig.add_trace(go.Scatter(x=[r["epoch"] for r in rows],y=[r[key] for r in rows],
                    name=f"{LABELS[v]} · {label}",line=dict(color=color,dash=dash),mode="lines+markers"))
    fig.update_layout(title="Training and validation objective · fixed evaluation windows",template="plotly_white",
        xaxis_title="Epoch",yaxis_title="Fair trajectory energy score + variogram",height=600,
        legend=dict(orientation="h",y=-.2,font=dict(size=11)),margin=dict(b=190))
    figures.append(fig.to_html(full_html=False,include_plotlyjs=False))
    frames=[]
    for v in variants:
        for p in POLICIES:
            path=root/"variants"/v/"evaluation"/p/"profile_metrics.csv"
            if path.exists():frames.append(pd.read_csv(path))
    table="<p>Annual scores will appear after training.</p>"
    if frames:
        all_scores=pd.concat(frames,ignore_index=True)
        all_scores.to_csv(root/"profile_metrics.csv",index=False)
        summary=aggregate_trace_scores(all_scores)
        summary.to_csv(root/"summary.csv")
        table=summary[["rmse","crps","coverage_90","width_90","bias","buildings","failed_buildings"]].round(4).to_html()
        fig=make_subplots(rows=2,cols=2,subplot_titles=("RMSE [°C] ↓","CRPS [°C] ↓","90% coverage · target 0.90","90% interval width [°C]"))
        for index,metric in enumerate(("rmse","crps","coverage_90","width_90")):
            for v in variants:
                color=COLORS[v]
                part=summary.reset_index();part=part[part.variant.eq(v)]
                fig.add_trace(go.Bar(x=part.policy,y=part[metric],name=LABELS[v],marker_color=color,
                    showlegend=index==0,legendgroup=v),row=index//2+1,col=index%2+1)
        fig.add_hline(y=.9,line_dash="dash",row=2,col=1)
        fig.update_layout(template="plotly_white",height=850,barmode="group",legend=dict(orientation="h",y=1.1),margin=dict(b=150))
        figures.append(fig.to_html(full_html=False,include_plotlyjs=False))
        paired=[]
        for p in POLICIES if final_reports else ():
            part=all_scores[all_scores.policy.eq(p)]
            for candidate,reference in [(variants[j],variants[i]) for i in range(len(variants)) for j in range(i+1,len(variants))]:
                a=part[part.variant.eq(candidate)].set_index("building")
                b=part[part.variant.eq(reference)].set_index("building")
                if any("simulation_failed" in z and z.simulation_failed.fillna(0).any() for z in (a,b)):continue
                common=a.index.intersection(b.index)
                if len(common)<2:continue
                for metric in ("rmse","crps","coverage_90","width_90"):
                    delta=(a.loc[common,metric]-b.loc[common,metric]).to_numpy()
                    rng=np.random.default_rng(1313)
                    boot=delta[rng.integers(0,len(delta),(4000,len(delta)))].mean(1)
                    paired.append(dict(policy=p,candidate=candidate,reference=reference,metric=metric,
                        n=len(delta),difference=float(delta.mean()),low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975))))
        if paired:pd.DataFrame(paired).to_csv(root/"paired_bootstrap.csv",index=False)
    annual_frames=[]
    for v in variants:
        for path in (root/"variants"/v/"evaluation").glob("*/annual_temperature_kpis.csv"):
            annual_frames.append(pd.read_csv(path))
    if annual_frames:
        annual=pd.concat(annual_frames,ignore_index=True)
        if len(annual):
            annual["absolute_error"]=(annual["median"]-annual["truth"]).abs()
            annual.to_csv(root/"annual_kpi_by_building.csv",index=False)
            a_summary=annual.groupby(["variant","policy","kpi"]).mean(numeric_only=True).drop(columns="building")
            a_summary["buildings"]=annual.groupby(["variant","policy","kpi"]).size()
            for idx in a_summary.index:
                if frames and summary.loc[idx[:2],"failed_buildings"]>0:
                    a_summary.loc[idx,:]=np.nan
            a_summary.to_csv(root/"annual_kpi_summary.csv")
            table+="<h2>Annual temperature KPIs</h2>"+a_summary.round(4).to_html()
    failures=[]
    for v in variants:
        path=root/"variants"/v/"training_failed.json"
        if path.exists():failures.append(f"{LABELS[v]}: {read(path)['reason']}")
    failure_text="<p>Numerical training failures: "+html.escape("; ".join(failures))+"</p>" if failures else ""
    links=[]
    for v in variants:
        for path in sorted((root/"variants"/v/"evaluation").glob("*/profile_*.html")):
            links.append(f'<li><a href="{path.relative_to(root)}">{html.escape(LABELS[v]+" · "+path.parent.name+" · "+path.stem)}</a></li>')
    refresh='<meta http-equiv="refresh" content="45">' if current.get("state") not in ("complete","complete_with_failures") else ""
    body=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{refresh}
    <title>Thermal structure and noise · {total_buildings} buildings</title><script src="plotly.min.js"></script>
    <style>body{{font:16px system-ui;margin:24px auto;max-width:1250px;padding:0 16px;color:#24344a}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd}}.scroll{{overflow:auto}}code{{white-space:pre-wrap}}li{{margin:8px 0}}</style></head><body>
    <h1>Thermal structure and noise · {total_buildings} buildings</h1><p>{html.escape(str(current))}</p>
    <p>{fold_counts} buildings. Shared observed Qroom, ventilation, weather, gains and metadata. Native ensembles; no ex-post calibration. Tariff results are conditional thermal transfer, not full-emulator policy transfer.</p>
    <p>The noise ablation changes only access to hidden state in the diffusion network. The contraction ablation keeps exogenous noise and positive diagonal structure, while allowing poles to exceed one. New runs match initial pole values and first derivatives. LSTM: unrestricted recurrent dynamics. TBPTT 32 steps (8 h); training windows 960 steps (10 days). Annual evaluation carries states and each annual latent through every chunk without resets.</p>
    <p>Validation stopping: minimum {req.get('minimum_epochs',req.get('epochs',15))} epochs, maximum {req.get('epochs',15)}, patience {req.get('early_stopping_patience',0)}. All models use the same rule. Nonfinite simulations are recorded without discarding particles; an affected policy's aggregate scores remain unavailable.</p>
    {failure_text}{''.join(figures)}<div class="scroll">{table}</div><h2>Annual temperature traces</h2><ul>{''.join(links)}</ul>
    <p>Training curves use fixed training-probe and validation windows at each epoch's completed parameters. Checkpoints are selected by excitation validation only. CRPS uses the empirical native ensemble; coverage is central-interval coverage. Temperature increments are scored at 15 min, 1 h and 24 h. Paired confidence intervals resample buildings, not timestamps; this single-seed screen does not quantify training-seed uncertainty.</p>
    <p><a href="comparison_request.json">Protocol</a> · <a href="summary.csv">Mean scores</a> · <a href="profile_metrics.csv">Per-building scores</a> · <a href="annual_kpi_summary.csv">Annual KPIs</a> · <a href="paired_bootstrap.csv">Paired bootstrap (after completion)</a></p></body></html>'''
    tmp=root/"index.html.tmp";tmp.write_text(body);tmp.replace(root/"index.html")


def run(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    jobs=[("prepare",None)] + [(stage,v) for v in args.variants for stage in ("train","evaluate")]
    write(root/"job_plan.json",[dict(stage=s,variant=v) for s,v in jobs])
    for stage,variant in jobs:
        if variant and (root/"variants"/variant/"training_failed.json").exists():
            print(f"Skipping terminal numerical training failure: {variant}",flush=True)
            continue
        cmd=[sys.executable,"-u","-m",__package__+".thermal_noise_experiment","--stage",stage,
             "--output",str(root),"--campaign",str(args.campaign),"--cohort",str(args.cohort),
             "--epochs",str(args.epochs),"--sequence-length",str(args.sequence_length),"--stride",str(args.stride),
             "--batch-size",str(args.batch_size),"--train-particles",str(args.train_particles),
             "--eval-particles",str(args.eval_particles),"--annual-particles",str(args.annual_particles),"--seed",str(args.seed),
             "--minimum-epochs",str(args.minimum_epochs),"--patience",str(args.patience),"--variants",*args.variants]
        if variant:cmd += ["--variant",variant]
        log=root/"logs"/f"{stage}_{variant or 'data'}.log";log.parent.mkdir(parents=True,exist_ok=True)
        status(root,state="running",stage=stage,variant=variant,progress="starting")
        report(args)
        with log.open("a") as handle:
            child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
            status(root,child_pid=child.pid)
            code=child.wait()
        if code==42 and stage=="train" and (root/"variants"/variant/"training_failed.json").exists():
            print(f"Numerical training failure recorded for {variant}; continuing remaining arms.",flush=True)
            report(args)
            continue
        if code:
            status(root,state="failed",exit_code=code,log=str(log),child_pid=None)
            report(args)
            raise RuntimeError(f"{stage}/{variant} failed; see {log}")
    failures=[]
    for v in args.variants:
        folder=root/"variants"/v
        if (folder/"training_failed.json").exists():failures.append(dict(variant=v,stage="train"))
        elif read(folder/"evaluation_complete.json").get("numerical_failures"):
            failures.append(dict(variant=v,stage="evaluate"))
    status(root,state="complete_with_failures" if failures else "complete",stage="complete",
        progress=f"{len(args.variants)} model arms finished; numerical failures in {len(failures)} arms",
        numerical_failures=failures,child_pid=None)
    report(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("run","prepare","train","evaluate","report"),default="run")
    p.add_argument("--output",type=Path,default=ROOT);p.add_argument("--campaign",type=Path,default=CAMPAIGN)
    p.add_argument("--cohort",type=Path,default=COHORT);p.add_argument("--variant",choices=VARIANTS)
    p.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
    p.add_argument("--epochs",type=int,default=30);p.add_argument("--seed",type=int,default=13)
    p.add_argument("--minimum-epochs",type=int,default=15);p.add_argument("--patience",type=int,default=5)
    p.add_argument("--sequence-length",type=int,default=960);p.add_argument("--stride",type=int,default=2024)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--train-particles",type=int,default=24)
    p.add_argument("--eval-particles",type=int,default=16);p.add_argument("--annual-particles",type=int,default=100)
    args=p.parse_args()
    if args.stage in ("train","evaluate") and args.variant is None:p.error("--variant required")
    if not 1<=args.minimum_epochs<=args.epochs:p.error("Require 1 <= minimum epochs <= maximum epochs")
    if args.patience<0:p.error("Patience must be nonnegative; zero disables early stopping")
    if min(args.train_particles,args.eval_particles,args.annual_particles)<2:p.error("At least two particles required")
    if min(args.sequence_length,args.stride,args.batch_size)<1:p.error("Window and batch sizes must be positive")
    if len(set(args.variants))!=len(args.variants):p.error("Duplicate variants")
    if args.variant and args.variant not in args.variants:p.error("--variant must be in --variants")
    try:
        globals()[args.stage](args)
    except FloatingPointError as exc:
        if args.stage!="train":raise
        write(args.output/"variants"/args.variant/"training_failed.json",dict(reason=str(exc),
            variant=args.variant,status=read(args.output/"status.json"),
            treatment="Numerical failure; no substitution of a different model or filtered scores."))
        report(args)
        print(f"numerical_training_failure: {exc}",flush=True)
        raise SystemExit(42) from exc


if __name__=="__main__":main()
