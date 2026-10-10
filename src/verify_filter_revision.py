"""Independent four-arm scoring, ratio checks and raw-source pairing audit."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha256,write_json,config
from benchmark_models import BenchmarkCNN,DEVICE
from verify_nonlinear_revision import manual_endpoint,manual_feature,manual_ba,independent_state_hash

OUT=ROOT/'results/filter_revision';OLD=ROOT/'data/erp_validation';CACHE=ROOT/'data/filter_revision'
CFG=json.loads((ROOT/'filter_revision_config.json').read_text());PART=json.loads((ROOT/'causal_revision_config.json').read_text())

def bootstrap(a,b=None):
    a=np.asarray(a,float);index=np.random.default_rng(CFG['bootstrap_seed']).integers(len(a),size=(10000,len(a)))
    values=a[index].mean(1) if b is None else a[index].mean(1)/np.asarray(b)[index].mean(1)
    return np.quantile(values,[.025,.975])

def raw_design_check():
    design=np.load(OUT/'matched_design.npz');original=np.load(ROOT/'results/nonlinear_revision/matched_design.npz')
    for k in original.files:np.testing.assert_array_equal(design[k],original[k])
    tc=PART['target_intervention'];manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text())
    eog=np.load(ROOT/Path(config()['source_dir'])/'EOG_all_epochs.npy').astype(float)
    eog-=eog.mean(1,keepdims=True);eog/=eog.std(1,keepdims=True);maximum=0.
    for ti,task in enumerate(CFG['tasks']):
        old=np.load(ROOT/f'results/causal_revision/{task}_target_design.npz');added=np.load(OUT/f'{task}_filter_design.npz')
        meta=json.loads((OUT/f'{task}_filter_design.json').read_text());rng=np.random.default_rng(tc['seed'])
        for split,field,n,key,selected in [('train','provenance',20000,'recipe_filter',10000),('val','validation_provenance',4000,'validation_recipe_filter',2000)]:
            pools={t:[] for t in ['parent','recipe_filter']}
            for r in meta[field]['erp_trials']:
                assert r['subject'] in PART['development']
                for t in pools:
                    path=(OLD if t=='parent' else ROOT/'data/causal_revision')/f'{task}_{r["subject"]:03d}_{t}.npy'
                    pools[t].append(np.load(path)[r['trial_indices']].reshape(-1,512))
            pools={t:np.concatenate(v).astype(float) for t,v in pools.items()}
            ids=[manifest['sources']['EOG']['kept_original_rows'][i] for i in manifest['pools']['EOG'][split]]
            si=rng.integers(len(pools['parent']),size=n);ai=rng.choice(ids,n);db=rng.choice(tc['power_snr_db'],n)
            np.testing.assert_array_equal(ai,meta[field]['eog_original_indices']);np.testing.assert_array_equal(db,meta[field]['power_snr_db'])
            p=pools['parent'][si];mixed=p+p.std(1,keepdims=True)*10.**(-db[:,None]/20.)*eog[ai];sd=mixed.std(1,keepdims=True)
            np.testing.assert_array_equal(mixed/sd,old['x' if split=='train' else 'vx'])
            target=pools['recipe_filter'][si]/sd;np.testing.assert_array_equal(target,added[key])
            np.testing.assert_array_equal(target[:selected].astype(np.float32),design[key][ti*selected:(ti+1)*selected])
        # Independently solve the added affine normal equations on all original rows.
        x=old['x'];y=added['recipe_filter'];xc=x-x.mean(0);g=xc.T@xc;eta=.001*np.trace(g)/512
        w=np.linalg.solve(g+eta*np.eye(512),xc.T@(y-y.mean(0)));b=y.mean(0)-x.mean(0)@w
        fitted=np.load(OUT/f'{task}_filter_affine.npz');np.testing.assert_allclose(w,fitted['matrix'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(b,fitted['intercept'],atol=1e-12,rtol=1e-12)
        maximum=max(maximum,float(np.max(abs(w-fitted['matrix']))))
    return maximum

def output_path(row):
    name=f'{row["task"]}_{row["subject"]:03d}';a=row['architecture'];t=row['target'];condition=row['condition']
    if a=='affine':
        if t=='recipe_filter' or condition!='parent_input':return CACHE/f'{name}_affine_{t}_{condition}.npy'
        return ROOT/f'data/causal_revision/{name}_target_{t}.npy'
    directory=CACHE if t=='recipe_filter' else ROOT/'data/nonlinear_revision'
    return directory/f'{name}_matched_{a}_{t}_{condition}.npy'


def raw_recipe_check():
    # Independently rebuild a complete filter stream for one evaluation person
    # per paradigm, then check the frozen component actions on those epochs.
    import mne
    from scipy.signal import resample_poly,butter,sosfiltfilt,iirnotch,filtfilt
    for task in CFG['tasks']:
        name=f'{task}_007';source=ROOT/f'data/erp_core/sub-007/eeg/sub-007_task-{task}_eeg.set'
        original=mne.io.read_raw_eeglab(source,preload=True,verbose='ERROR')
        raw=resample_poly(original.get_data()[:30],1,4,axis=-1);raw-=raw.mean(0,keepdims=True)
        bounds=json.loads((OLD/f'{name}.json').read_text())['bounds_256']
        high=butter(4,[1,100],btype='bandpass',fs=256,output='sos');low=butter(4,30,btype='lowpass',fs=256,output='sos');nb,na=iirnotch(60,30,fs=256)
        filtered=np.empty_like(raw)
        for first,last in zip(bounds[:-1],bounds[1:]):
            filtered[:,first:last]=filtfilt(nb,na,sosfiltfilt(high,raw[:,first:last]))
        # RawArray shares the float64 high-pass buffer; MNE's explicitly
        # repeated average reference acts on it before the analysis lowpass.
        filtered-=filtered.mean(0,keepdims=True)
        for first,last in zip(bounds[:-1],bounds[1:]):
            filtered[:,first:last]=sosfiltfilt(low,filtered[:,first:last])*1e6
        sample=np.load(OLD/f'{name}_identity.npz')['sample'];indices=sample[:,None]+np.arange(512)-128
        epochs=filtered[:,indices].transpose(1,0,2);times=np.arange(512)/256-.5;base=(times>=-.2)&(times<0)
        epochs-=epochs[:,:,base].mean(-1,keepdims=True)
        saved=np.load(ROOT/f'data/filter_revision/passband100/{name}_recipe_filter.npy')
        np.testing.assert_array_equal(epochs.astype(np.float32),saved)
        projection=np.load(OUT/f'passband100/{name}_recipe_projection.npz')
        metadata=json.loads((OUT/f'passband100/{name}_recipe.json').read_text())
        assert metadata['source_sha256']==sha256(source) and metadata['source_fdt_sha256']==sha256(source.with_suffix('.fdt'))
        p=projection['probabilities'];assert metadata['rejected']['brain_argmax']==np.flatnonzero(p.argmax(1)!=0).tolist()
        assert metadata['rejected']['artifact80']==np.flatnonzero(p[:,1:6].max(1)>=.8).tolist()
        for policy in ['brain_argmax','artifact80']:
            projected=np.einsum('ck,nkt->nct',projection[policy],epochs);projected-=projected[:,:,base].mean(-1,keepdims=True)
            saved=np.load(ROOT/f'data/filter_revision/passband100/{name}_recipe_{policy}.npy')
            np.testing.assert_array_equal(projected.astype(np.float32),saved)

def main():
    assert raw_design_check()<1e-12
    prior_cfg=json.loads((ROOT/'nonlinear_revision_config.json').read_text())
    for key in ['architectures','seeds','tasks','fit_rows_per_task','validation_rows_per_task','epochs','batch_size','inference_batch_size','optimizer','learning_rate','rho','epsilon','weight_decay']:
        assert CFG[key]==prior_cfg[key],key
    old_source=(ROOT/'src/matched_cnn.py').read_text();new_source=(ROOT/'src/filter_cnn.py').read_text()
    for source in [old_source,new_source]:assert 'for epoch in range(1,CFG' in source
    original_loop=old_source.split('for epoch in range(1,CFG',1)[1].split('    state={',1)[0]
    added_loop=new_source.split('for epoch in range(1,CFG',1)[1].split('    state={',1)[0]
    assert original_loop==added_loop
    fits=[]
    for architecture,seed in itertools.product(CFG['architectures'],CFG['seeds']):
        r=json.loads((OUT/f'{architecture}_{seed}_recipe_filter.json').read_text());old=json.loads((ROOT/f'results/nonlinear_revision/{architecture}_{seed}_parent.json').read_text())
        torch.manual_seed(seed);m=BenchmarkCNN(architecture)
        assert r['initial_state_sha256']==old['initial_state_sha256']==independent_state_hash(m.state_dict())
        h=hashlib.sha256();rng=np.random.default_rng(seed)
        for epoch in range(50):h.update(rng.permutation(20000).astype('<i8').tobytes())
        assert r['minibatch_order_sha256']==old['minibatch_order_sha256']==h.hexdigest()
        assert r['selected_epoch']==50 and len(r['history'])==50
        assert r['config_sha256']==sha256(ROOT/'filter_revision_config.json')
        assert r['architecture_code_sha256']==old['architecture_code_sha256']==sha256(ROOT/'src/benchmark_models.py')
        assert r['device']==old['device']
        assert r['code_sha256']==sha256(ROOT/'src/filter_cnn.py')
        assert r['checkpoint_sha256']==sha256(OUT/f'{architecture}_{seed}_recipe_filter.pth')
        assert r['input_design_sha256']==sha256(OUT/'matched_design.npz')
        fits.append(r)
    report=json.loads((OUT/'four_arm_assessment.json').read_text());by={};maximum=0.
    for r in report['rows']:
        x=np.load(output_path(r));task=r['task'];y=np.load(OLD/f'{task}_{r["subject"]:03d}_identity.npz')['label']
        amplitude=manual_endpoint(x,y,task)[0];ba=manual_ba(manual_feature(x),y,ROOT/f'results/revision/{task}_parent_decoder.npz')
        np.testing.assert_allclose([amplitude,ba],[r['amplitude_uv'],r['frozen_ba']],atol=1e-12,rtol=1e-12)
        maximum=max(maximum,abs(amplitude-r['amplitude_uv']))
        by[r['architecture'],r['target'],task,r['subject'],r['condition']]=amplitude
        if r['architecture']=='affine' and r['target']=='recipe_filter' and r['condition']!='output_reference':
            name=f'{task}_{r["subject"]:03d}'
            source=OLD/f'{name}_parent.npy' if r['condition']=='parent_input' else ROOT/f'data/causal_revision/{name}_intervention_contaminated_inputs.npy'
            original=np.load(source);a=original.reshape(-1,512).astype(float);sd=a.std(1,keepdims=True);model=np.load(OUT/f'{task}_filter_affine.npz')
            predicted=((a/sd@model['matrix']+model['intercept'])*sd).reshape(original.shape)
            times=np.arange(512)/256-.5;predicted-=predicted[:,:,(times>=-.2)&(times<0)].mean(-1,keepdims=True)
            np.testing.assert_array_equal(predicted.astype(np.float32),x)
    teachers={}
    for r in report['teacher_endpoint_rows']:
        task=r['task'];name=f'{task}_{r["subject"]:03d}';t=r['target'];source=(OLD if t=='parent' else ROOT/'data/causal_revision')/f'{name}_{t}.npy'
        y=np.load(OLD/f'{name}_identity.npz')['label'];amp=manual_endpoint(np.load(source),y,task)[0]
        np.testing.assert_allclose(amp,r['amplitude_uv'],atol=1e-12);teachers[task,r['subject'],t]=amp
    for r in report['comparisons']:
        a,t,c,p,q=r['architecture'],r['task'],r['condition'],r['from_target'],r['to_target']
        output=np.array([by[a,q,t,s,c]-by[a,p,t,s,c] for s in PART['evaluation']]);direct=np.array([teachers[t,s,q]-teachers[t,s,p] for s in PART['evaluation']])
        np.testing.assert_allclose([output.mean(),direct.mean(),(output-direct).mean()],[r['output_shift_uv'],r['direct_reference_shift_uv'],r['residual_shift_uv']],atol=1e-12)
        np.testing.assert_allclose(bootstrap(output),r['output_shift_ci95'],atol=1e-12)
        np.testing.assert_allclose(output.mean()/direct.mean(),r['transfer']['ratio_of_means'],atol=1e-12)
        np.testing.assert_allclose(bootstrap(output,direct),r['transfer']['ci95'],atol=1e-12)
    for r in report['filter_shares']:
        a=r['architecture'];q=r['to_target'];f=np.array([by[a,'recipe_filter','P3',s,'parent_input']-by[a,'parent','P3',s,'parent_input'] for s in PART['evaluation']]);d=np.array([by[a,q,'P3',s,'parent_input']-by[a,'parent','P3',s,'parent_input'] for s in PART['evaluation']])
        np.testing.assert_allclose(f.mean()/d.mean(),r['filtering_share_of_learned_shift']['ratio_of_means'],atol=1e-12)
        np.testing.assert_allclose(bootstrap(f,d),r['filtering_share_of_learned_shift']['ci95'],atol=1e-12)
    seed_by={}
    for r in report['individual_seeds']:
        task=r['task'];name=f'{task}_{r["subject"]:03d}';target=r['target'];arch=r['architecture']
        folder=CACHE if target=='recipe_filter' else ROOT/'data/nonlinear_revision'
        data=np.load(folder/f'{name}_matched_{arch}_{target}_parent_input_seed_endpoints.npz')
        at=list(data['seed']).index(r['seed']);wave=data['roi_contrasts'][at]
        times=np.arange(512)/256-.5
        task_cfg=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks'][task]
        window=(times>=task_cfg['window_s'][0])&(times<task_cfg['window_s'][1]);base=(times>=-.2)&(times<0)
        amplitude=float(wave[window].mean()-wave[base].mean())
        ba=manual_ba(data['features'][at],data['label'],ROOT/f'results/revision/{task}_parent_decoder.npz')
        np.testing.assert_allclose([amplitude,ba],[r['amplitude_uv'],r['frozen_ba']],atol=1e-12)
        seed_by[arch,task,r['seed'],target,r['subject']]=amplitude
    for r in report['seed_pairs']:
        a,t,z,p,q=r['architecture'],r['task'],r['seed'],r['from_target'],r['to_target']
        differences=np.array([seed_by[a,t,z,q,s]-seed_by[a,t,z,p,s] for s in PART['evaluation']])
        np.testing.assert_allclose([differences.mean(),*bootstrap(differences)],[r['mean_shift_uv'],*r['ci95']],atol=1e-12)
    design=np.load(OUT/'matched_design.npz')
    for r in report['ensemble_cross_target_losses']:
        a,t,q=r['architecture'],r['training_target'],r['scoring_target'];folder=OUT if t=='recipe_filter' else ROOT/'results/nonlinear_revision'
        predictions=np.mean([np.load(folder/f'{a}_{z}_{t}_validation_predictions.npy').astype(float) for z in CFG['seeds']],axis=0)
        np.testing.assert_allclose(np.mean((predictions-design['validation_'+q])**2),r['ensemble_mse'],atol=1e-12)
    discrepancy=max(abs(r['identity_discrepancy_uv']) for r in report['participant_decompositions']);assert discrepancy<1e-12
    # Fresh independent checkpoint execution on complete physical trials of subject7.
    rerun_error=0.
    for r in fits:
        m=BenchmarkCNN(r['architecture']);m.load_state_dict(torch.load(OUT/f'{r["architecture"]}_{r["seed"]}_recipe_filter.pth',map_location='cpu',weights_only=True));m.to(DEVICE).eval()
        source=np.load(OLD/'P3_007_parent.npy');y=np.load(OLD/'P3_007_identity.npz')['label'];indices=np.r_[np.flatnonzero(y==0)[:2],np.flatnonzero(y==1)[:2]];source=source[indices]
        flat=source.reshape(-1,512);sd=flat.std(1,keepdims=True)
        with torch.no_grad():z=(m(torch.from_numpy((flat/sd).astype(np.float32))[:,None].to(DEVICE)).cpu().numpy()*sd).reshape(source.shape)
        times=np.arange(512)/256-.5;z-=z[:,:,(times>=-.2)&(times<0)].mean(-1,keepdims=True)
        saved=np.load(CACHE/f'P3_007_matched_{r["architecture"]}_recipe_filter_parent_input_seed_endpoints.npz')
        at=list(saved['seed']).index(r['seed']);f=manual_feature(z).astype(np.float32);expected=saved['features'][at,indices]
        np.testing.assert_allclose(f,expected,atol=3e-4,rtol=3e-5);rerun_error=max(rerun_error,float(np.max(abs(f-expected))))
    sensitivity=json.loads((OUT/'passband_sensitivity.json').read_text())
    assert len(sensitivity['rows'])==240 and len(sensitivity['recipe_metadata'])==80
    for r in sensitivity['recipe_metadata']:
        assert r['configuration_sha256']==sha256(ROOT/'filter_revision_config.json')
        assert r['code_sha256']==sha256(ROOT/'src/reference_passband100.py')
    for r in sensitivity['rows']:
        name=f'{r["task"]}_{r["subject"]:03d}';p=ROOT/f'data/filter_revision/passband100/{name}_{r["method"]}.npy'
        assert sha256(p)==r['output_100_sha256'];labels=np.load(OLD/f'{name}_identity.npz')['label']
        np.testing.assert_allclose(manual_endpoint(np.load(p),labels,r['task'])[0],r['amplitude_100_uv'],atol=1e-12)
    raw_recipe_check()
    write_json(OUT/'verification.json',dict(status='passed',new_fixed_epoch_fits=len(fits),total_paired_cnn_fits=24,
        ensemble_records_rescored=len(report['rows']),four_arm_comparisons_checked=len(report['comparisons']),
        individual_seed_records_rescored=len(report['individual_seeds']),seed_pairs_checked=len(report['seed_pairs']),cross_target_losses_checked=len(report['ensemble_cross_target_losses']),
        transmission_identities_checked=len(report['participant_decompositions']),transfer_and_filter_share_bootstrap_checked=True,
        exact_original_inputs_and_targets_verified=True,filter_labels_regenerated_from_original_draws=True,affine_normal_equations_and_outputs_verified=True,
        original_training_loop_and_hyperparameters_identical=True,
        maximum_endpoint_error_uv=maximum,maximum_identity_discrepancy_uv=discrepancy,checkpoints_independently_rerun=6,maximum_checkpoint_feature_error_uv=rerun_error,
        passband_recipe_records=80,passband_endpoint_records=240,configuration_sha256=sha256(ROOT/'filter_revision_config.json'),code_sha256=sha256(__file__),
        complete_100hz_filter_streams_regenerated=2,frozen_100hz_component_actions_regenerated=4,
        scope='Numerical and pairing verification; does not create independent cohort, neural source truth or causal validation of original released labels'))
    print('Four-arm revision verified independently',flush=True)

if __name__=='__main__':main()
