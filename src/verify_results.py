"""Recompute saved outcomes and check source isolation and model selection."""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT,config,sha256,write_json,per_row_metrics

def verify_target_extension(data):
    c=json.loads((ROOT/'target_validity_config.json').read_text())
    dst=ROOT/'results'/'target_validity'
    s=json.loads((dst/'summary.json').read_text())
    path=ROOT/'data/target_validity/latent_coefficients.npz'
    assert s['config']==c
    assert s['config_sha256']==sha256(ROOT/'target_validity_config.json')
    assert sha256(path)==s['known_source']['data_sha256']
    assert sha256(ROOT/'src/target_validity.py')==s['code_sha256']
    latent=dict(np.load(path));u=latent['u'];b=latent['b'];low=latent['low'];a=latent['overlap']
    np.testing.assert_allclose(latent['basis']@latent['basis'].T/512,np.eye(3),atol=1e-12)
    amp=c['simulation_task_amplitude'];sigma=c['simulation_overlap_artifact_sd']
    n=np.column_stack([b,amp*u,np.zeros(len(u))]);artifact=np.column_stack([np.zeros(len(u)),a,low])
    observed=n+artifact
    m=amp*np.tanh(amp*(amp*u+a)/sigma**2)
    for row in s['known_source']['rows']:
        alpha=row['alpha'];t=n.copy();t[:,1]*=1-alpha
        ah=artifact.copy();ah[:,1]+=alpha*amp*u
        np.testing.assert_allclose(t+ah,observed,atol=1e-12)
        f=np.column_stack([b,(1-alpha)*m,np.zeros(len(u))])
        assert abs(np.mean(np.sum((f-t)**2,1))-row['benchmark_mse'])<1e-12
        assert abs(np.mean(np.sum((f-n)**2,1))-row['neural_mse'])<1e-12
        assert abs(np.mean(np.where(f[:,1]>0,1.,-1.)==u)-row['empirical_task_accuracy'])<1e-12
        benchmark_snr=10*np.log10(np.mean(np.sum(t*t,1))/np.mean(np.sum(ah*ah,1)))
        physiological_snr=10*np.log10(np.mean(np.sum(n*n,1))/np.mean(np.sum(artifact*artifact,1)))
        assert abs(benchmark_snr-row['benchmark_snr_db'])<1e-12
        assert abs(physiological_snr-row['physiological_snr_db'])<1e-12
        assert abs(row['neural_mse']-row['theoretical_neural_mse'])<.01
    assert s['known_source']['rows'][-1]['benchmark_mse']==0
    assert s['known_source']['rows'][-1]['neural_mse']==amp**2
    for row in s['ranking_sensitivity']:
        typ=row['artifact'];target=np.load(ROOT/'data/mixtures'/f'{typ}_test.npz')['s'].astype(float)
        pred=dict(np.load(ROOT/'results/mixtures'/typ/'predictions.npz'))
        def family(name):
            return np.asarray([pred[f'{name}_seed{k}'] for k in [17,29,43]],float) if name in ['tiny_cnn','large_mlp'] else np.asarray([pred[name]],float)
        f,g=family(row['f']),family(row['g']);gap=np.mean((f-target)**2)-np.mean((g-target)**2)
        direction=f.mean(0)-g.mean(0);kappa=np.sqrt(np.mean(direction**2))
        radius=abs(gap)/(2*kappa)
        assert abs(radius-row['minimum_bias_rms'])<1e-12
        delta=-np.sign(gap)*1.01*radius*direction/kappa
        hypothetical=target-delta
        shifted=np.mean((f-hypothetical)**2)-np.mean((g-hypothetical)**2)
        assert abs(shifted-row['adversarial_check_difference'])<1e-12
        assert gap*shifted<0
    for row in csv.DictReader((dst/'risk_bounds.csv').open()):
        typ=row['artifact'];target=np.load(ROOT/'data/mixtures'/f'{typ}_test.npz')['s'].astype(float)
        pred=dict(np.load(ROOT/'results/mixtures'/typ/'predictions.npz'))
        name=row['method']
        members=np.asarray([pred[f'{name}_seed{k}'] for k in [17,29,43]],float) if name in ['tiny_cnn','large_mlp'] else np.asarray([pred[name]],float)
        mean=members.mean(0);spread=np.mean((members-mean)**2)
        mean_residual_rms=np.sqrt(np.mean((mean-target)**2))
        radius=float(row['bias_fraction_target_rms'])*np.sqrt(np.mean(target*target))
        lower=spread+max(0,mean_residual_rms-radius)**2
        upper=spread+(mean_residual_rms+radius)**2
        assert abs(np.mean((members-target)**2)-float(row['reference_mse']))<1e-12
        assert abs(lower-float(row['neural_mse_lower']))<1e-12
        assert abs(upper-float(row['neural_mse_upper']))<1e-12
    teacher=json.loads((ROOT/'results/teacher_intervention/summary.json').read_text())
    assert teacher['config']==c
    assert sha256(ROOT/'src/teacher_intervention.py')==teacher['code_sha256']
    assert sha256(ROOT/'data/bci_epochs.npz')==teacher['data_sha256']
    from teacher_intervention import teacher_projection
    for variant in c['real_teacher_variants']:
        transformed=teacher_projection(data['minimal'],variant)
        energy=np.mean((transformed-data['minimal'])**2)
        p=dict(np.load(ROOT/'results/teacher_intervention'/f'{variant}_predictions.npz'))
        for row in [r for r in teacher['summary'] if r['teacher']==variant]:
            scores=[np.mean(p[f'{row["decoder"]}_subject{sub}']==data['label'][data['subject']==sub]) for sub in range(1,10)]
            assert abs(np.mean(scores)-row['mean_accuracy'])<1e-12
            assert abs(energy-row['parent_against_teacher_mse_uv2'])<1e-4
    return dict(experiment='target validity extension',latent_truth_simulation_rescored=True,
        identical_observations=True,sharp_ranking_reversals_verified=True,risk_bounds_rescored=True,
        teacher_predictions_rescored=True,
        simulated_draws=len(u),real_participants=9,status='exploratory controlled interventions')

def main():
    checks=[];c=config()
    for typ in ['EOG','EMG','ECG']:
        dst=ROOT/'results'/'mixtures'/typ
        banks={p:dict(np.load(ROOT/'data'/'mixtures'/f'{typ}_{p}.npz')) for p in ['train','val','test']}
        run=json.loads((dst/'run.json').read_text())
        assert run['config']==c
        assert sha256(ROOT/run.get('executed_code_snapshot','src/mixture_study.py'))==run['code_sha256']
        for part in ['train','val','test']:
            assert sha256(ROOT/'data'/'mixtures'/f'{typ}_{part}.npz')==run['data_sha256'][part]
        for component in ['eeg_id','artifact_id']:
            for a,b in [('train','val'),('train','test'),('val','test')]:
                assert not set(banks[a][component])&set(banks[b][component])
        manifest=json.loads((ROOT/'data'/'mixtures'/'manifest.json').read_text())
        if typ=='ECG':
            records=np.asarray(manifest['sources']['ECG']['record_for_row'])
            for a,b in [('train','val'),('train','test'),('val','test')]:
                assert not set(records[banks[a]['artifact_id']])&set(records[banks[b]['artifact_id']])
        for part,d in banks.items():
            assert d['x'].shape==(c['mixture_counts'][part],512)
            assert np.isfinite(d['x']).all() and np.isfinite(d['s']).all()
            actual=10*np.log10(np.sum(d['s']**2,1)/np.sum((d['x']-d['s'])**2,1))
            assert np.max(abs(actual-d['snr_db']))<1e-4
        histories=list(dst.glob('*_history.json'))
        assert len(histories)==12
        for path in histories:
            info=json.loads(path.read_text())
            assert len(info['history'])==c['epochs']
            assert info['best_validation_mse']==min(h['validation_mse'] for h in info['history'])
            assert info['history'][info['best_epoch']-1]['validation_mse']==info['best_validation_mse']
            assert sha256(path.with_name(path.name.replace('_history.json','.pt')))==info['checkpoint_sha256']
        selections=json.loads((dst/'selection.json').read_text())
        for s in selections:
            tag=f'{s["kind"]}_seed{s["seed"]}_lr{s["lr"]}'
            assert sha256(dst/f'{tag}.pt')==s['checkpoint_sha256']
            assert len(s['history'])==c['epochs']
            assert s['best_validation_mse']==min(h['validation_mse'] for h in s['history'])
            alternatives=[json.loads((dst/f'{s["kind"]}_seed{s["seed"]}_lr{lr}_history.json').read_text()) for lr in c['learning_rates']]
            assert s['best_validation_mse']==min(a['best_validation_mse'] for a in alternatives)
        predictions=dict(np.load(dst/'predictions.npz'))
        summary={r['method']:r for r in csv.DictReader((dst/'summary.csv').open())}
        for method,pred in predictions.items():
            assert pred.shape==banks['test']['s'].shape and np.isfinite(pred).all()
            metr=per_row_metrics(pred,banks['test']['s'])
            for k,v in metr.items():
                assert abs(float(summary[method][k])-np.nanmean(v))<1e-10
        for family in ['tiny_cnn','large_mlp']:
            members=[per_row_metrics(predictions[f'{family}_seed{seed}'],banks['test']['s']) for seed in c['training_seeds']]
            for k in members[0]:
                score=np.nanmean(np.mean([m[k] for m in members],axis=0))
                assert abs(float(summary[family+'_mean_seeds'][k])-score)<1e-10
        checks.append(dict(experiment=typ,source_disjoint=True,power_snr_verified=True,
                           validation_only_selection=True,predictions_rescored=True,completed_fits=len(list(dst.glob('*_history.json')))))
    data=dict(np.load(ROOT/'data'/'bci_epochs.npz'))
    assert len(data['label'])==2592
    summary=json.loads((ROOT/'results'/'real_task'/'summary.json').read_text())
    all_metrics=[]
    for fold in summary['folds']:
        sub=fold['subject'];assert sub not in fold['training_subjects']
        p=dict(np.load(ROOT/'results'/'real_task'/f'predictions_{sub}.npz'))
        assert np.array_equal(p['label'],data['label'][data['subject']==sub])
        f=json.loads((ROOT/'results'/'real_task'/f'fold_{sub}.json').read_text())
        for row in f['metrics']:
            mask={'all':np.ones(len(p['label']),bool),'retained':p['retained'],
                  'artifact_marked':p['artifact'],'unmarked':~p['artifact']}[row['stratum']]
            accuracy=np.mean(p[row['method']][mask]==p['label'][mask])
            assert abs(accuracy-row['accuracy'])<1e-12
            all_metrics.append(row)
    for row in summary['summary']:
        selected=[m for m in all_metrics if m['method']==row['method'] and m['stratum']==row['stratum']]
        assert len(selected)==9
        assert abs(np.mean([m['accuracy'] for m in selected])-row['mean_accuracy'])<1e-12
        assert abs(np.mean([m['coverage'] for m in selected])-row['mean_coverage'])<1e-12
    checks.append(dict(experiment='BCI real task',participant_held_out=True,identical_trials=True,
                       predictions_rescored=True,participants=9,trials=2592,
                       ica_converged=all(not f['ica']['warnings'] for f in summary['folds'])))
    matched=json.loads((ROOT/'results'/'benchmark_task_transfer'/'summary.json').read_text())
    for row in matched['summary']:
        if row['method'].endswith('mean_seeds'):
            family=row['method'].replace('_mean_seeds','')
            methods=[f'{family}_seed{seed}' for seed in c['training_seeds']]
        else:
            methods=[row['method']]
        values=[]
        for method in methods:
            p=dict(np.load(ROOT/'results'/'benchmark_task_transfer'/f'{method}_predictions.npz'))
            values.extend([np.mean(p[('csp' if row['decoder']=='CSP' else 'bandpower')+f'_subject{sub}']==data['label'][data['subject']==sub]) for sub in c['bci_subjects']])
        assert abs(np.mean(values)-row['mean_accuracy'])<1e-12
    checks.append(dict(experiment='matched benchmark task transfer',predictions_rescored=True,status='post hoc'))
    checks.append(verify_target_extension(data))
    write_json(ROOT/'results'/'verification.json',dict(status='passed',checks=checks,
        config_sha256=sha256(ROOT/'config.json'),verification_code_sha256=sha256(__file__)))
    print('Verified all completed fits, disjoint source pools, power SNR, validation selection and stored task predictions.')

if __name__=='__main__':
    main()
