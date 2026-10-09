"""Independent endpoint, score, projection, target-fit and split verification."""
import json,math
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np

OUT=ROOT/'results/causal_revision';CACHE=ROOT/'data/causal_revision';OLD=ROOT/'data/erp_validation'

def main():
    cfg=json.loads((ROOT/'causal_revision_config.json').read_text());r=json.loads((OUT/'erp_assessment.json').read_text())
    assert r['config_sha256']==sha256(ROOT/'causal_revision_config.json')
    t=np.arange(512)/256-.5;bw=(t>=-.2)&(t<0);tasks=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks'];checked=0
    max_projection=0.;fit_error=0.;lookup={}
    for task,tc in tasks.items():
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in tc['roi']];sw=(t>=tc['window_s'][0])&(t<tc['window_s'][1])
        dec=np.load(ROOT/f'results/revision/{task}_parent_decoder.npz')
        for s in range(1,41):
            y=np.load(OLD/f'{task}_{s:03d}_identity.npz')['label'];parent=np.load(OLD/f'{task}_{s:03d}_parent.npy').astype(float)
            pp=parent[:,roi][:,:,sw].mean((1,2))-parent[:,roi][:,:,bw].mean((1,2));pam=pp[y==1].mean()-pp[y==0].mean()
            mat=np.load(OUT/f'{task}_{s:03d}_recipe_projection.npz');meta=json.loads((OUT/f'{task}_{s:03d}_recipe.json').read_text())
            np.testing.assert_allclose(mat['reconstruct'],np.eye(30),atol=1e-10)
            np.testing.assert_allclose(mat['probabilities'].sum(1),1,atol=1e-6)
            assert np.flatnonzero(np.argmax(mat['probabilities'],axis=1)!=0).tolist()==meta['rejected']['brain_argmax']
            assert np.flatnonzero(mat['probabilities'][:,1:6].max(1)>=.8).tolist()==meta['rejected']['artifact80']
            for m in r['methods']:
                record=next(a for a in r['rows'] if a['task']==task and a['subject']==s and a['method']==m);lookup[task,s,m]=record
                x0=np.load(CACHE/f'{task}_{s:03d}_{m}.npy');x=x0.astype(float);assert x.shape==parent.shape and np.isfinite(x).all()
                a=x[:,roi][:,:,sw].mean((1,2))-x[:,roi][:,:,bw].mean((1,2));amp=a[y==1].mean()-a[y==0].mean()
                assert abs(amp-record['amplitude_uv'])<1e-9 and abs(pam-record['parent_amplitude_uv'])<1e-9
                f=np.concatenate([x0[:,:,(t>=a)&(t<a+.1)].mean(-1,dtype=np.float32) for a in np.arange(0,.8,.1)],1)
                for d,key in [(dec,'frozen_ba'),(np.load(OUT/f'{task}_{m}_decoder.npz'),'adapted_ba')]:
                    z=f.copy();z-=d['scale_mean'].astype(z.dtype);z/=d['scale'].astype(z.dtype)
                    score=z@d['coef'].T+d['intercept'];pred=d['classes'][(score[:,0]>0).astype(int)]
                    ba=np.mean([np.mean(pred[y==c]==c) for c in [0,1]])
                    assert abs(ba-record[key])<1e-12
                if m.startswith('ica_') or m.startswith('recipe_') and m!='recipe_filter':
                    policy=m.split('_',1)[1] if m.startswith('ica_') else m[len('recipe_'):]
                    source=parent if m.startswith('ica_') else np.load(CACHE/f'{task}_{s:03d}_recipe_filter.npy').astype(float)
                    expected=np.einsum('ck,nkt->nct',mat[policy],source);expected-=expected[:,:,bw].mean(-1,keepdims=True)
                    max_projection=max(max_projection,float(np.max(abs(x-expected))));np.testing.assert_allclose(x,expected,atol=2e-4,rtol=2e-6)
                checked+=1
        target=np.load(OUT/f'{task}_target_design.npz');models=np.load(OUT/f'{task}_target_intervention.npz');xm=target['x'].mean(0);xc=target['x']-xm
        gram=xc.T@xc;ridge=json.loads((OUT/f'{task}_target_intervention.json').read_text())['ridge'];a=gram+ridge*np.eye(512)
        for m in cfg['target_intervention']['targets']:
            yy=target[m];ym=yy.mean(0);w=np.linalg.solve(a,xc.T@(yy-ym));b=ym-xm@w
            fit_error=max(fit_error,float(np.max(abs(w-models[m+'_matrix']))));np.testing.assert_allclose(w,models[m+'_matrix'],atol=1e-10);np.testing.assert_allclose(b,models[m+'_intercept'],atol=1e-10)
            if m!='parent':
                delta=yy-target['parent'];dw=np.linalg.solve(a,xc.T@(delta-delta.mean(0)))
                np.testing.assert_allclose(dw,models[m+'_matrix']-models['parent_matrix'],atol=1e-10)
        provenance=json.loads((OUT/f'{task}_target_intervention.json').read_text())
        assert all(d['subject'] in cfg['development'] for d in provenance['provenance']['erp_trials'])
        tr=provenance['provenance'];va=provenance['validation_provenance']
        assert set(tr['eog_original_indices']).isdisjoint(va['eog_original_indices'])
        for d,e in zip(tr['erp_trials'],va['erp_trials']):assert d['subject']==e['subject'] and set(d['trial_indices']).isdisjoint(e['trial_indices'])
    for row in r['summary']:
        rr=[lookup[row['task'],s,row['method']] for s in cfg['evaluation']]
        for k,source in [('mean_amplitude_uv','amplitude_uv'),('mean_frozen_ba','frozen_ba'),('mean_adapted_ba','adapted_ba')]:assert abs(row[k]-np.mean([z[source] for z in rr]))<1e-12
    for m,c in r['calibration'].items():
        v=[max(abs(lookup[t,s,m]['amplitude_uv']-lookup[t,s,m]['parent_amplitude_uv']) for t in tasks) for s in cfg['calibration']+cfg['evaluation']]
        b=[max(abs(lookup[t,s,m]['frozen_ba']-lookup[t,s,m]['parent_ba']) for t in tasks) for s in cfg['calibration']+cfg['evaluation']]
        assert abs(sorted(v[:20])[18]-c['voltage90_uv'])<1e-10 and abs(sorted(b[:20])[18]*100-c['ba90_points'])<1e-10
    control=json.loads((OUT/'target_contamination_control.json').read_text())
    control_checked=0
    for record in control['rows']:
        task,s,target=record['task'],record['subject'],record['target'];tc=tasks[task]
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in tc['roi']];sw=(t>=tc['window_s'][0])&(t<tc['window_s'][1])
        y=np.load(OLD/f'{task}_{s:03d}_identity.npz')['label'];x=np.load(CACHE/f'{task}_{s:03d}_contaminated_target_{target}.npy').astype(float)
        values=x[:,roi][:,:,sw].mean((1,2))-x[:,roi][:,:,bw].mean((1,2));amp=values[y==1].mean()-values[y==0].mean()
        assert abs(amp-record['amplitude_uv'])<1e-9;control_checked+=1
    output_control=json.loads((OUT/'output_reference_control.json').read_text());output_checked=0
    for record in output_control['rows']:
        task,s,m=record['task'],record['subject'],record['method'];tc=tasks[task]
        where=OLD if m in ['parent','femto'] else ROOT/'data/tim_revision' if m=='icunet' else CACHE
        x=np.load(where/f'{task}_{s:03d}_{m}.npy').astype(float)
        if record['mode']=='output_average':x-=x.mean(1,keepdims=True)
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in tc['roi']];sw=(t>=tc['window_s'][0])&(t<tc['window_s'][1]);y=np.load(OLD/f'{task}_{s:03d}_identity.npz')['label']
        values=x[:,roi][:,:,sw].mean((1,2))-x[:,roi][:,:,bw].mean((1,2));amp=values[y==1].mean()-values[y==0].mean()
        assert abs(amp-record['amplitude_uv'])<1e-9;output_checked+=1
    probes=json.loads((OUT/'probes.json').read_text());probe_records=len(probes['rows'])
    for aggregate in probes['summary']:
        a=np.array([z['gain'] for z in probes['rows'] if z['task']==aggregate['task'] and z['method']==aggregate['method']])
        if aggregate['method'].startswith('ica_'):
            a=np.array([z['gain'] for z in probes['rows'] if z['task']==aggregate['task'] and z['method']==aggregate['method'] and z['subject'] in cfg['evaluation']])
        assert len(a)==aggregate['n'] and abs(a.mean()-aggregate['mean_gain'])<1e-12
    # Independently rerun physical scaling and endpoint arithmetic for the
    # first evaluation person's eight backgrounds in each model/task pair.
    import torch,mne
    from benchmark_models import load,DEVICE
    manual_probes=0
    positions=mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos'];pos={k.lower():v for k,v in positions.items()}
    for task,tc in tasks.items():
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'][:30];roi=[names.index(n) for n in tc['roi']]
        p=np.array([pos[n.lower()] for n in names]);p/=np.linalg.norm(p,axis=1,keepdims=True)
        distance=np.arccos(np.clip(p@p[roi].T,-1,1));spatial=np.exp(-.5*(distance/.4)**2).mean(1);spatial-=spatial.mean();spatial/=spatial[roi].mean()
        temporal=np.exp(-.5*((t-tc['probe_center_s'])/tc['probe_sd_s'])**2);s=spatial[:,None]*temporal;s-=s[:,bw].mean(-1,keepdims=True)
        sw=(t>=tc['window_s'][0])&(t<tc['window_s'][1]);den=s[roi][:,sw].mean()-s[roi][:,bw].mean()
        person=cfg['evaluation'][0];y=np.load(OLD/f'{task}_{person:03d}_identity.npz')['label'];idx=np.r_[np.flatnonzero(y==0)[:4],np.flatnonzero(y==1)[:4]]
        parent=np.load(OLD/f'{task}_{person:03d}_parent.npy')[idx].astype(float)
        for m in ['simple_cnn','complex_cnn']:
            models=[load(m,z) for z in cfg['models']['seeds']];outputs=[]
            for a in [0.,1.]:
                z=(parent+a*s).reshape(-1,512);scale=z.std(1,keepdims=True);data=torch.from_numpy((z/scale).astype(np.float32))[:,None].to(DEVICE)
                with torch.no_grad():result=np.mean([model(data).cpu().numpy() for model in models],axis=0)
                outputs.append((result*scale).reshape(parent.shape))
            difference=outputs[1]-outputs[0];v=(difference[:,roi][:,:,sw].mean((1,2))-difference[:,roi][:,:,bw].mean((1,2)))/den
            expected=np.array([next(z['gain'] for z in probes['rows'] if z['task']==task and z['method']==m and z['subject']==person and z['amplitude_uv']==1. and z['background_trial']==int(i)) for i in idx])
            np.testing.assert_allclose(v,expected,atol=2e-5,rtol=1e-4);manual_probes+=len(v)
    write_json(OUT/'verification.json',dict(status='passed',records_checked=checked,manual_voltage_and_decoder_scores=True,recipe_probability_policies_and_identity_checked=True,
        max_projection_rounding_error_uv=max_projection,maximum_refit_matrix_error=fit_error,paired_target_identity_verified=True,target_training_development_only=True,
        target_mixture_component_pools_and_erp_trial_indices_disjoint=True,quantiles_recomputed=True,matched_contamination_endpoint_records_checked=control_checked,
        output_reference_endpoint_records_checked=output_checked,probe_records_aggregate_checked=probe_records,physical_model_probe_responses_independently_rerun=manual_probes,
        config_sha256=sha256(ROOT/'causal_revision_config.json'),verifier_sha256=sha256(Path(__file__))))
    print('Verified follow-up',checked,'records and paired target-fit identity',flush=True)

if __name__=='__main__':main()
