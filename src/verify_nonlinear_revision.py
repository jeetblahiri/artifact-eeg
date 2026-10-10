"""Rescore paired CNN evidence without importing its assessment functions."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from common import ROOT, config, sha256, write_json
from benchmark_models import BenchmarkCNN,DEVICE

OUT=ROOT/'results/nonlinear_revision'
CACHE=ROOT/'data/nonlinear_revision'
OLD=ROOT/'data/erp_validation'
CFG=json.loads((ROOT/'nonlinear_revision_config.json').read_text())
PART=json.loads((ROOT/'causal_revision_config.json').read_text())
TASKS=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks']
TIMES=np.arange(512)/256-.5


def manual_endpoint(x,y,task):
    names=json.loads((OLD/f'{task}_001.json').read_text())['channels']
    channels=[names.index(c) for c in TASKS[task]['roi']]
    window=(TIMES>=TASKS[task]['window_s'][0])&(TIMES<TASKS[task]['window_s'][1])
    base=(TIMES>=-.2)&(TIMES<0)
    waveform=(x[y==1].astype(float).mean(0)-x[y==0].astype(float).mean(0))[channels].mean(0)
    return float(waveform[window].mean()-waveform[base].mean()),waveform


def manual_feature(x):
    return np.concatenate([x[:,:,(TIMES>=a)&(TIMES<a+.1)].mean(-1) for a in np.arange(0,.8,.1)],axis=1)


def manual_ba(features,y,decoder):
    d=np.load(decoder)
    z=features.copy();np.subtract(z,d['scale_mean'].astype(z.dtype),out=z);np.divide(z,d['scale'].astype(z.dtype),out=z)
    score=z@d['coef'].T+d['intercept']
    p=d['classes'][(score[:,0]>0).astype(int)]
    return float(np.mean([np.mean(p[y==k]==k) for k in [0,1]]))


def input_metrics(p,t):
    p=np.asarray(p,float).reshape(-1,512);t=np.asarray(t,float).reshape(-1,512)
    n=np.sum(t*t,axis=1);r=np.sqrt(np.sum((p-t)**2,axis=1)/n)
    pc=p-p.mean(1,keepdims=True);tc=t-t.mean(1,keepdims=True)
    correlations=np.sum(pc*tc,1)/np.maximum(np.sqrt(np.sum(pc*pc,1)*np.sum(tc*tc,1)),1e-12)
    return dict(normalized_mse=float(np.sum((p-t)**2)/np.sum(t*t)),
                pooled_relative_rms=float(np.sqrt(np.sum((p-t)**2)/np.sum(t*t))),
                median_window_relative_rms=float(np.median(r)),mean_correlation=float(correlations.mean()),
                through_origin_gain=float(np.sum(p*t)/np.sum(t*t)),
                output_to_input_rms=float(np.sqrt(np.sum(p*p)/np.sum(t*t))),fraction_below_10pct_relative_rms=float(np.mean(r<=.1)))


def independent_state_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().numpy()
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()


def regenerate_development_designs(design):
    """Recover selected mixture arrays from raw cached sources and RNG draws."""
    causal=json.loads((ROOT/'causal_revision_config.json').read_text())['target_intervention']
    manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text())
    raw=np.load(ROOT/Path(config()['source_dir'])/'EOG_all_epochs.npy').astype(float)
    raw-=raw.mean(1,keepdims=True);raw/=raw.std(1,keepdims=True)
    maximum=0.
    for task_index,task in enumerate(CFG['tasks']):
        metadata=json.loads((ROOT/f'results/causal_revision/{task}_target_intervention.json').read_text())
        rng=np.random.default_rng(causal['seed'])
        for split,field,n,key_prefix,selected in [
                ('train','provenance',causal['fit_mixtures'],'',CFG['fit_rows_per_task']),
                ('val','validation_provenance',causal['validation_mixtures'],'validation_',CFG['validation_rows_per_task'])]:
            evidence=metadata[field];pools={t:[] for t in CFG['targets']}
            for row in evidence['erp_trials']:
                for t in pools:
                    source=(OLD if t=='parent' else ROOT/'data/causal_revision')/f'{task}_{row["subject"]:03d}_{t}.npy'
                    pools[t].append(np.load(source)[row['trial_indices']].reshape(-1,512))
            pools={t:np.concatenate(v).astype(float) for t,v in pools.items()}
            original=[manifest['sources']['EOG']['kept_original_rows'][i] for i in manifest['pools']['EOG'][split]]
            si=rng.integers(len(pools['parent']),size=n);ai=rng.choice(original,n);db=rng.choice(causal['power_snr_db'],n)
            np.testing.assert_array_equal(ai,evidence['eog_original_indices']);np.testing.assert_array_equal(db,evidence['power_snr_db'])
            p=pools['parent'][si[:selected]];noise=raw[ai[:selected]]
            added=p.std(1,keepdims=True)*10.**(-db[:selected,None]/20.)*noise
            centered=p-p.mean(1,keepdims=True)
            achieved=10*np.log10(np.mean(centered*centered,1)/np.mean(added*added,1))
            np.testing.assert_allclose(achieved,db[:selected],atol=1e-10)
            mixed=p+added
            sd=mixed.std(1,keepdims=True);rows=slice(task_index*selected,(task_index+1)*selected)
            expected={'vx' if split=='val' else 'x':mixed/sd}
            expected.update({key_prefix+t:v[si[:selected]]/sd for t,v in pools.items()})
            for key,values in expected.items():
                difference=float(np.max(abs(values.astype(np.float32)-design[key][rows])))
                maximum=max(maximum,difference)
                np.testing.assert_allclose(values.astype(np.float32),design[key][rows],atol=2e-6,rtol=2e-6)
    return maximum


def main():
    design=np.load(OUT/'matched_design.npz')
    provenance=json.loads((OUT/'design_provenance.json').read_text())
    for key in design.files:
        expected=[]
        for task in CFG['tasks']:
            src=ROOT/f'results/causal_revision/{task}_target_design.npz'
            assert sha256(src)==provenance['source_design_sha256'][task]
            n=CFG['validation_rows_per_task'] if key=='vx' or key.startswith('validation_') else CFG['fit_rows_per_task']
            expected.append(np.load(src)[key][:n].astype(np.float32))
        np.testing.assert_array_equal(design[key],np.concatenate(expected))
    for task in CFG['tasks']:
        old=json.loads((ROOT/f'results/causal_revision/{task}_target_intervention.json').read_text())
        tr,va=old['provenance'],old['validation_provenance']
        assert set(tr['eog_original_indices']).isdisjoint(va['eog_original_indices'])
        for a,b in zip(tr['erp_trials'],va['erp_trials']):
            assert a['subject']==b['subject'] and a['subject'] in PART['development']
            assert set(a['trial_indices']).isdisjoint(b['trial_indices'])
    assert provenance['training_people']==PART['development']
    assert set(PART['development']).isdisjoint(PART['evaluation'])
    source_regeneration_error=regenerate_development_designs(design)
    fits=json.loads((OUT/'matched_training.json').read_text())['models']
    assert len(fits)==len(CFG['architectures'])*len(CFG['seeds'])*len(CFG['targets'])==18
    for architecture in CFG['architectures']:
        for seed in CFG['seeds']:
            group=[r for r in fits if r['architecture']==architecture and r['seed']==seed]
            assert {r['target'] for r in group}==set(CFG['targets'])
            for field in ['initial_state_sha256','minibatch_order_sha256','input_design_sha256','selected_epoch']:
                assert len({r[field] for r in group})==1,field
            torch.manual_seed(seed);initial_model=BenchmarkCNN(architecture)
            assert independent_state_hash(initial_model.state_dict())==group[0]['initial_state_sha256']
            del initial_model
            h=hashlib.sha256();rng=np.random.default_rng(seed)
            for _ in range(CFG['epochs']):h.update(rng.permutation(len(design['x'])).astype('<i8').tobytes())
            assert group[0]['minibatch_order_sha256']==h.hexdigest()
            for r in group:
                tag=f'{architecture}_{seed}_{r["target"]}'
                assert sha256(OUT/f'{tag}.pth')==r['checkpoint_sha256']
                assert r['selected_epoch']==CFG['epochs']==50
                assert [v['epoch'] for v in r['history']]==list(range(1,51))
                assert r['config_sha256']==sha256(ROOT/'nonlinear_revision_config.json')
                assert r['input_design_sha256']==sha256(OUT/'matched_design.npz')
                assert r['architecture_code_sha256']==sha256(ROOT/'src/benchmark_models.py')
                assert r['code_sha256']==sha256(ROOT/'src/matched_cnn.py')
                pred=np.load(OUT/f'{tag}_validation_predictions.npy')
                for t in CFG['targets']:
                    np.testing.assert_allclose(np.mean((pred-design['validation_'+t])**2),r['cross_target_validation_mse'][t],atol=1e-10)
    sanity=json.loads((OUT/'cnn_sanity.json').read_text())
    assert sanity['code_sha256']==sha256(ROOT/'src/cnn_sanity.py')
    assert sanity['config_sha256']==sha256(ROOT/'nonlinear_revision_config.json')
    assert sanity['source_bank_sha256']==sha256(ROOT/Path(config()['source_dir'])/'EEG_all_epochs.npy')
    manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text())
    indices=np.array(manifest['sources']['EEG']['kept_original_rows'])[manifest['pools']['EEG']['test']]
    assert list(indices)==sanity['heldout_original_rows']
    source=np.load(ROOT/Path(config()['source_dir'])/'EEG_all_epochs.npy').astype(float)
    source-=source.mean(-1,keepdims=True);source/=np.sqrt(np.mean(source*source,axis=-1,keepdims=True))
    bank=np.load(OUT/'uncontaminated_benchmark_inputs.npy')
    np.testing.assert_allclose(bank,source[indices],atol=2e-6)
    for row in sanity['benchmark']:
        pred=np.load(OUT/f'{row["architecture"]}_uncontaminated_predictions.npy')
        for k,v in input_metrics(pred.mean(0),bank).items():np.testing.assert_allclose(v,row[k],rtol=1e-9,atol=1e-10)
    for row in sanity['individual_seeds']:
        predictions=np.load(OUT/f'{row["architecture"]}_uncontaminated_predictions.npy')
        pred=predictions[CFG['seeds'].index(row['seed'])]
        for key,value in input_metrics(pred,bank).items():np.testing.assert_allclose(value,row[key],atol=1e-10)
    max_roundoff=0.
    for row in sanity['erp_rows']:
        t,s,a,m=row['task'],row['subject'],row['architecture'],row['mode'];name=f'{t}_{s:03d}'
        parent=np.load(OLD/f'{name}_parent.npy');y=np.load(OLD/f'{name}_identity.npz')['label']
        if m=='delivered':p=ROOT/f'data/causal_revision/{name}_{a}.npy'
        elif m=='central_backgrounds':p=ROOT/f'data/causal_revision/{name}_{a}.npy'
        else:p=CACHE/f'{name}_{a}_{m}.npy'
        z=np.load(p)
        if 'background' in m:
            ids=row['background_trial_ids'];parent=parent[ids];y=y[ids]
            if m=='central_backgrounds':z=z[ids]
            max_roundoff=max(max_roundoff,row['context_roundoff_uv'])
        np.testing.assert_allclose(manual_endpoint(z,y,t)[0],row['contrast_uv'],atol=1e-10)
        for k,v in input_metrics(z,parent).items():np.testing.assert_allclose(v,row[k],atol=1e-10)
    data=json.loads((OUT/'matched_assessment.json').read_text());lookup={}
    assert data['code_sha256']==sha256(ROOT/'src/matched_cnn_assessment.py')
    assert data['config_sha256']==sha256(ROOT/'nonlinear_revision_config.json')
    for row in data['individual_seeds']:
        a,t,s,target=row['architecture'],row['task'],row['subject'],row['target']
        n=f'{t}_{s:03d}';tag=f'matched_{a}_{target}'
        cache=np.load(CACHE/f'{n}_{tag}_parent_input_seed_endpoints.npz')
        index=list(cache['seed']).index(row['seed']);f=cache['features'][index];w=cache['roi_contrasts'][index];y=cache['label']
        window=(TIMES>=TASKS[t]['window_s'][0])&(TIMES<TASKS[t]['window_s'][1]);base=(TIMES>=-.2)&(TIMES<0)
        np.testing.assert_allclose(w[window].mean()-w[base].mean(),row['amplitude_uv'],atol=1e-10)
        assert manual_ba(f,y,ROOT/f'results/revision/{t}_parent_decoder.npz')==row['frozen_ba']
    max_endpoint_error=0.;max_identity_error=0.;count=0
    for row in data['rows']:
        a,t,s,target,c=row['architecture'],row['task'],row['subject'],row['target'],row['condition']
        name=f'{t}_{s:03d}';tag=f'matched_{a}_{target}'
        dst=CACHE/f'{name}_{tag}_{c}.npy';z=np.load(dst);y=np.load(OLD/f'{name}_identity.npz')['label'];f=manual_feature(z)
        amplitude,w=manual_endpoint(z,y,t);max_endpoint_error=max(max_endpoint_error,abs(amplitude-row['amplitude_uv']))
        np.testing.assert_allclose(amplitude,row['amplitude_uv'],atol=1e-10)
        assert manual_ba(f,y,ROOT/f'results/revision/{t}_parent_decoder.npz')==row['frozen_ba']
        assert manual_ba(f,y,OUT/f'{t}_{tag}_adapted_decoder.npz')==row['adapted_ba']
        if c=='output_reference':
            original=np.load(CACHE/f'{name}_{tag}_parent_input.npy')
            expected=original-original.mean(1,keepdims=True);base=(TIMES>=-.2)&(TIMES<0)
            expected-=expected[...,base].mean(-1,keepdims=True)
            np.testing.assert_array_equal(z,expected)
        else:
            meta=json.loads(dst.with_suffix('.json').read_text())
            assert meta['output_sha256']==sha256(dst)
            src=OLD/f'{name}_parent.npy' if c=='parent_input' else ROOT/f'data/causal_revision/{name}_intervention_contaminated_inputs.npy'
            assert meta['input_sha256']==sha256(src)
        lookup[a,target,t,s,c]=row;count+=1
    for row in data['endpoint_transmission']:
        a,target,t,s,c=row['architecture'],row['target'],row['task'],row['subject'],row['condition']
        n=f'{t}_{s:03d}';y=np.load(OLD/f'{n}_identity.npz')['label'];parent=np.load(OLD/f'{n}_parent.npy')
        teacher=np.load(ROOT/f'data/causal_revision/{n}_{target}.npy')
        z0=np.load(CACHE/f'{n}_matched_{a}_parent_{c}.npy');z1=np.load(CACHE/f'{n}_matched_{a}_{target}_{c}.npy')
        direct=manual_endpoint(teacher,y,t)[0]-manual_endpoint(parent,y,t)[0]
        learned=manual_endpoint(z1,y,t)[0]-manual_endpoint(z0,y,t)[0]
        residual=manual_endpoint(z1.astype(float)-teacher,y,t)[0]-manual_endpoint(z0.astype(float)-parent,y,t)[0]
        for k,v in [('direct_reference_shift_uv',direct),('learned_output_shift_uv',learned),('residual_shift_uv',residual)]:
            np.testing.assert_allclose(v,row[k],atol=1e-10)
        max_identity_error=max(max_identity_error,abs(learned-direct-residual))
    assert count==480 and len(data['individual_seeds'])==720
    for row in data['summary']:
        a,target,t,c=row['architecture'],row['target'],row['task'],row['condition']
        rr=[lookup[a,target,t,s,c] for s in PART['evaluation']]
        assert len(rr)==row['n_people']==10
        for key,field in [('mean_amplitude_uv','amplitude_uv'),('mean_frozen_ba','frozen_ba'),('mean_adapted_ba','adapted_ba')]:
            np.testing.assert_allclose(np.mean([r[field] for r in rr]),row[key],atol=1e-12)
    for r in data['ensemble_cross_target_losses']:
        a,t=r['architecture'],r['training_target'];p=np.mean([np.load(OUT/f'{a}_{s}_{t}_validation_predictions.npy').astype(float) for s in CFG['seeds']],0)
        np.testing.assert_allclose(np.mean((p-design['validation_'+r['scoring_target']])**2),r['ensemble_mse'],atol=1e-12)
    for r in data['pairs']:
        a,t,target,c=r['architecture'],r['task'],r['target'],r['condition']
        values=np.array([lookup[a,target,t,s,c]['amplitude_uv']-lookup[a,'parent',t,s,c]['amplitude_uv'] for s in PART['evaluation']])
        np.testing.assert_allclose(values.mean(),r['amplitude_difference_uv'],atol=1e-12)
        # Same declared seed, sampled independently of the report builder.
        seed=json.loads((ROOT/'tim_revision_config.json').read_text())['seed']
        draw_count=json.loads((ROOT/'tim_revision_config.json').read_text())['bootstrap_draws']
        rng=np.random.default_rng(seed);means=rng.choice(values,(draw_count,len(values)),replace=True).mean(1)
        np.testing.assert_allclose(np.quantile(means,[.025,.975]),r['amplitude_ci95'],atol=1e-12)
    # Independently execute every new checkpoint on the same four complete
    # evaluation trials, with manual SD restoration/baseline/feature scoring.
    reruns=0;max_feature_error=0.
    person=PART['evaluation'][0];task='P3';name=f'{task}_{person:03d}'
    labels=np.load(OLD/f'{name}_identity.npz')['label'];idx=np.r_[np.flatnonzero(labels==0)[:2],np.flatnonzero(labels==1)[:2]]
    parent=np.load(OLD/f'{name}_parent.npy')[idx];a=parent.reshape(-1,512);scale=a.std(1,keepdims=True)
    baseline_mask=(TIMES>=-.2)&(TIMES<0)
    for architecture in CFG['architectures']:
        for target in CFG['targets']:
            tag=f'matched_{architecture}_{target}';saved=np.load(CACHE/f'{name}_{tag}_parent_input_seed_endpoints.npz')
            for seed in CFG['seeds']:
                model=BenchmarkCNN(architecture);model.load_state_dict(torch.load(OUT/f'{architecture}_{seed}_{target}.pth',map_location='cpu',weights_only=True),strict=True)
                model=model.to(DEVICE).eval()
                with torch.no_grad():p=model(torch.from_numpy((a/scale).astype(np.float32))[:,None].to(DEVICE)).cpu().numpy()
                z=(p*scale).reshape(parent.shape);z-=z[...,baseline_mask].mean(-1,keepdims=True);z=z.astype(np.float32)
                f=manual_feature(z);reference=saved['features'][CFG['seeds'].index(seed)][idx]
                err=float(np.max(abs(f-reference)));max_feature_error=max(max_feature_error,err)
                # Batch-size-dependent floating accumulation on MPS is allowed;
                # this is numerical verification, not a voltage tolerance.
                np.testing.assert_allclose(f,reference,atol=3e-4,rtol=3e-5)
                del model;reruns+=1
    assert max_identity_error<1e-10
    assert not any(r['split']=='calibration' for r in data['rows'])
    write_json(OUT/'verification.json',dict(status='passed',fixed_final_epoch_fits=18,paired_initializations_and_batch_order=True,
               development_only_design_verified=True,source_indices_and_no_artifact_inputs_checked=True,
               raw_development_source_design_regenerated=True,maximum_design_regeneration_discrepancy=source_regeneration_error,
               centered_voltage_power_snr_verified=True,erp_window_means_retained=True,
               sanity_records_checked=len(sanity['erp_rows']),ensemble_records_rescored=count,individual_seed_records=len(data['individual_seeds']),
               transmission_identities_checked=len(data['endpoint_transmission']),maximum_identity_discrepancy_uv=max_identity_error,
               maximum_endpoint_rescoring_discrepancy_uv=max_endpoint_error,maximum_context_cache_roundoff_uv=max_roundoff,
               checkpoints_independently_rerun=reruns,rerun_device=DEVICE,complete_trials_per_checkpoint=4,maximum_rerun_feature_discrepancy_uv=max_feature_error,
               cross_target_matrices_and_paired_intervals_verified=True,configuration_sha256=sha256(ROOT/'nonlinear_revision_config.json'),
               code_sha256=sha256(__file__),scope='Saved-output and fit-pairing verification; no cortical ground truth or prospective test claimed.'))
    print('Nonlinear revision independently verified:',count,'ensemble records;',len(data['endpoint_transmission']),'endpoint identities')


if __name__=='__main__':main()
