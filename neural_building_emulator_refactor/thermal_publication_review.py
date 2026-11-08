"""Read-only analysis of frozen thermal benchmark outputs; no retraining."""
import argparse
from pathlib import Path
import itertools
import json
import os

import numpy as np
import pandas as pd

HERE: Path
ROOT: Path
MODELS = ['diagonal_exogenous', 'diagonal_state', 'lstm_state']
POLICIES = ['setpoint_excitation_v2', 'force_off_excitation_v2', 'tariff_setpoint_v2', 'tariff_force_off_v2']
NAMES = ['Diagonal / exogenous noise', 'Diagonal / state-dependent noise', 'LSTM / state-dependent noise']
COLORS = ['#2563eb', '#d97706', '#0f8b70']


def boot(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    # Buildings, rather than timestamps or policy/building rows, are resampled.
    idx = np.random.default_rng(1313).integers(0, len(values), (4000, len(values)))
    means = values[idx].mean(1)
    return float(values.mean()), *np.quantile(means, [.025, .975]).tolist()


def intervals(frame, groups, metrics):
    rows = []
    for key, part in frame.groupby(groups):
        for metric in metrics:
            mean, lo, hi = boot(part[metric])
            rows.append(dict(zip(groups, key), metric=metric, mean=mean, low95=lo, high95=hi, n=len(part)))
    return pd.DataFrame(rows)


def paired(frame, groups, metrics):
    rows = []
    for key, part in frame.groupby(groups):
        if not isinstance(key, tuple): key = (key,)
        for reference, candidate in itertools.combinations(MODELS, 2):
            a = part[part.variant.eq(candidate)].set_index('building').sort_index()
            b = part[part.variant.eq(reference)].set_index('building').sort_index()
            assert len(a) == len(b) == 100 and a.index.equals(b.index)
            for metric in metrics:
                mean, lo, hi = boot(a[metric] - b[metric])
                rows.append(dict(zip(groups, key), candidate=candidate, reference=reference,
                    metric=metric, difference=mean, low95=lo, high95=hi, n=len(a),
                    relative_change_percent=100 * mean / b[metric].mean() if b[metric].mean() else None))
    return pd.DataFrame(rows)


def numerical_stability():
    rows = []
    for path in (ROOT/'variants').glob('*/evaluation/setpoint_excitation_v2/stability_*.json'):
        obj = json.loads(path.read_text())
        poles = np.asarray(obj.get('poles', []))
        rows.append(dict(variant=path.parents[2].name, building=int(path.stem.split('_')[1]),
            noncontractive_particles=int((poles.max(1) >= 1).sum()) if poles.size else None,
            max_pole=obj.get('max_pole'), failed_particles=len(obj['failed_particles']),
            hidden_states_finite=obj['hidden_states_finite']))
    table = pd.DataFrame(rows)
    table.to_csv(HERE/'stability_by_building.csv', index=False)
    return table


def deficit_diagnostic():
    destination = HERE/'deficit_decomposition.csv'
    if destination.exists(): return pd.read_csv(destination)
    rows = []
    for variant in MODELS:
        for policy in POLICIES:
            for path in sorted((ROOT/'variants'/variant/'evaluation'/policy).glob('samples_*.npz')):
                building = int(path.stem.split('_')[1])
                with np.load(ROOT/'cache'/policy/f'{building}.npz') as cache:
                    mask = cache['available'] > .5
                    setpoint = cache['setpoint'][mask].astype(float)
                with np.load(path) as obj:
                    samples = obj['samples'][:, mask].astype(float)
                    truth = obj['truth'][mask].astype(float)
                mean = samples.mean(0)
                k_truth = .25 * np.maximum(setpoint-truth, 0).sum()
                k_mean = .25 * np.maximum(setpoint-mean, 0).sum()
                k_samples = .25 * np.maximum(setpoint-samples, 0).sum(1)
                # Convex thresholding separates deterministic mean-path deficit
                # from the additional expected deficit caused by ensemble spread.
                extra = k_samples.mean()-k_mean
                assert extra >= -1e-6
                rows.append(dict(variant=variant, policy=policy, building=building,
                    truth_degree_hours=k_truth, mean_path_degree_hours=k_mean,
                    expected_path_degree_hours=k_samples.mean(), median_path_degree_hours=np.median(k_samples),
                    mean_path_error=k_mean-k_truth, spread_increment=extra,
                    heating_bias=(mean-truth).mean(),
                    truth_within_0p1_of_setpoint=float((np.abs(truth-setpoint) <= .1).mean())))
            print('deficit diagnostic', variant, policy, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(destination, index=False)
    return result


def make_figure(trace, annual, trace_ci):
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    s = trace.groupby(['variant','policy']).mean(numeric_only=True)
    a = annual.groupby(['variant','policy','kpi']).mean(numeric_only=True)
    ci = trace_ci.set_index(['variant','policy','metric'])
    x = np.arange(4)
    labels = ['Excitation\nsetpoint', 'Excitation\nforce-off', 'Tariff\nsetpoint', 'Tariff\nforce-off']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, (model, name, color) in enumerate(zip(MODELS, NAMES, COLORS)):
        xx = x+(i-1)*.23
        for ax, metric in zip(axes[0], ['rmse','crps','coverage_90']):
            mul = 100 if metric=='coverage_90' else 1
            vals = mul*np.array([s.loc[(model,p),metric] for p in POLICIES])
            lo = mul*np.array([ci.loc[(model,p,metric),'low95'] for p in POLICIES])
            hi = mul*np.array([ci.loc[(model,p,metric),'high95'] for p in POLICIES])
            ax.errorbar(xx, vals, yerr=[vals-lo,hi-vals], fmt='o', color=color, capsize=3, label=name)
        for ax,kpi,metric in [(axes[1,0],'mean_tin_c','crps'),
                (axes[1,1],'below_setpoint_degree_hours','crps'),
                (axes[1,2],'below_setpoint_degree_hours','coverage_90')]:
            mul = 100 if metric=='coverage_90' else 1
            vals = [mul*a.loc[(model,p,kpi),metric] for p in POLICIES]
            ax.bar(xx,vals,color=color,width=.21)
            if metric=='coverage_90':
                for xx_,vv in zip(xx,vals):ax.text(xx_,vv+1,f'{vv:.0f}',ha='center',fontsize=8,color=color)
    titles = ['A  Trace accuracy', 'B  Trace probabilistic score', 'C  Trace interval calibration',
              'D  Annual mean temperature', 'E  Annual setpoint deficit', 'F  Annual setpoint-deficit coverage']
    ylabels = ['RMSE [°C] ↓','CRPS [°C] ↓','90% interval coverage [%]',
               'CRPS [°C] ↓','CRPS [°C h] ↓','90% interval coverage [% of buildings]']
    for ax,title,label in zip(axes.flat,titles,ylabels):
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(label)
        ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in (axes[0,2],axes[1,2]):ax.axhline(90,color='#444',linestyle='--',linewidth=1)
    axes[0,2].set_ylim(73,100);axes[1,2].set_ylim(0,104)
    fig.suptitle('Thermal-only benchmark: accuracy, calibration and annual accumulation',fontsize=16,y=.98)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.06,.065,'Unconstrained diagonal: 5/100 buildings have nonfinite annual particles under every policy; aggregate scores unavailable.\n'
        '701 training / 100 validation / 99 reserved calibration / 100 test buildings. One seed; 100 annual particles per building.\n'
        'Observed Qroom and ventilation are prescribed. No post-calibration. Error bars: 95% building-bootstrap intervals (unadjusted).\n'
        'Setpoint deficit uses zero tolerance during heating availability; it is not a regulatory comfort KPI.',fontsize=9)
    fig.subplots_adjust(left=.065,right=.985,top=.86,bottom=.21,hspace=.45,wspace=.32)
    fig.savefig(HERE/'thermal_results_review.png',dpi=180,bbox_inches='tight')
    fig.savefig(HERE/'thermal_results_review.pdf',bbox_inches='tight')


def main():
    global ROOT, HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Completed four-arm thermal benchmark with 100 test buildings.')
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to RUN_DIR/publication_review.')
    args = parser.parse_args()
    ROOT = args.run_dir.resolve()
    HERE = args.output_dir.resolve() if args.output_dir else ROOT/'publication_review'
    HERE.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(ROOT/'profile_metrics.csv')
    trace = trace[trace.variant.isin(MODELS)].copy()
    assert len(trace)==1200 and not trace.simulation_failed.any()
    annual = pd.read_csv(ROOT/'annual_kpi_by_building.csv')
    annual = annual[annual.variant.isin(MODELS)].copy()
    assert len(annual)==2400
    metrics = ['rmse','crps','bias','coverage_50','coverage_80','coverage_90','coverage_95','width_90',
        'heating_rmse','heating_crps','heating_coverage_90','heating_width_90',
        'increment_crps_1','increment_crps_4','increment_crps_96']
    ci = intervals(trace,['variant','policy'],metrics)
    ci.to_csv(HERE/'trace_score_intervals.csv',index=False)
    effects = paired(trace,['policy'],metrics)
    effects.to_csv(HERE/'paired_trace_effects.csv',index=False)
    annual_ci = intervals(annual,['variant','policy','kpi'],['crps','coverage_90','width_90','absolute_error'])
    annual_ci.to_csv(HERE/'annual_score_intervals.csv',index=False)
    paired(annual,['policy','kpi'],['crps','coverage_90','width_90','absolute_error']).to_csv(HERE/'paired_annual_effects.csv',index=False)
    # Paired policy-shift contrasts respect the repeated use of each building.
    shifts = []
    for v in MODELS:
        for training_policy,transfer_policy in zip(POLICIES[:2],POLICIES[2:]):
            left=trace[trace.variant.eq(v)&trace.policy.eq(training_policy)].set_index('building').sort_index()
            right=trace[trace.variant.eq(v)&trace.policy.eq(transfer_policy)].set_index('building').sort_index()
            for metric in ['rmse','crps','coverage_90']:
                mean,lo,hi=boot(right[metric]-left[metric])
                shifts.append(dict(variant=v,reference=training_policy,transfer=transfer_policy,metric=metric,
                    difference=mean,low95=lo,high95=hi))
    pd.DataFrame(shifts).to_csv(HERE/'paired_policy_shift.csv',index=False)
    # Coverage differences from nominal, not raw coverage differences, measure
    # whether more coverage actually improved calibration.
    reliability=[]
    for (v,p),part in trace.groupby(['variant','policy']):
        for level in [50,80,90,95]:
            vals=part[f'coverage_{level}']
            reliability.append(dict(variant=v,policy=p,nominal=level/100,coverage=vals.mean(),
                pooled_absolute_gap=abs(vals.mean()-level/100),
                mean_building_absolute_gap=np.abs(vals-level/100).mean(),
                fraction_buildings_below_80pct_coverage=float((vals<.8).mean()) if level==90 else None))
    pd.DataFrame(reliability).to_csv(HERE/'reliability.csv',index=False)
    # Normalized PIT bin frequencies are descriptive, not independent-timestep tests.
    pits=[]
    for v in MODELS:
        for p in POLICIES:
            counts=np.asarray(json.loads((ROOT/'variants'/v/'evaluation'/p/'pit_histogram.json').read_text()))
            pits.extend(dict(variant=v,policy=p,bin=i,frequency=float(n/counts.sum())) for i,n in enumerate(counts))
    pd.DataFrame(pits).to_csv(HERE/'pit_bins.csv',index=False)
    numerical_stability()
    make_figure(trace,annual,ci)
    print('Score comparisons and publication figure complete',flush=True)
    diag=deficit_diagnostic()
    diag.groupby(['variant','policy']).mean(numeric_only=True).drop(columns='building').to_csv(HERE/'deficit_decomposition_summary.csv')
    (HERE/'protocol.json').write_text(json.dumps(dict(
        source='Frozen selected-checkpoint outputs; no model, score, or checkpoint changes',
        bootstrap_replicates=4000,bootstrap_seed=1313,bootstrap_unit='building',
        interval_caveat='Unadjusted exploratory 95% intervals, conditional on this training seed; policies share buildings',
        unstable_arm='Report failure incidence and pole diagnostics; no survivor-only aggregate ranking',
        temperature_deficit='0.25 sum(max(setpoint-Tin,0)) over space-heating availability',
        scope='Conditional thermal prediction with observed Qroom and ventilation; no full-emulator flexibility claims'),indent=2)+'\n')
    print('Review finished',flush=True)


if __name__=='__main__': main()
