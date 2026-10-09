"""Rescore ERP results from saved voltages, identities, operators and decoders."""
import json,math
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def main():
    cfg=json.loads((ROOT/'tim_validation_config.json').read_text());out=ROOT/'results/erp_validation';cache=ROOT/'data/erp_validation'
    report=json.loads((out/'summary.json').read_text());assert report['config']==cfg
    assert report['config_sha256']==sha256(ROOT/'tim_validation_config.json')
    assert report['code_sha256']==sha256(ROOT/'src/erp_validation.py')
    parts=[set(cfg[k]) for k in ['development','calibration','evaluation']]
    assert set.union(*parts)==set(range(1,41)) and sum(map(len,parts))==40
    assert all(not(a&b) for i,a in enumerate(parts) for b in parts[i+1:])
    manifest=json.loads((ROOT/'data/erp_core/download_manifest.json').read_text())
    for r in manifest['files']:
        p=ROOT/'data/erp_core'/r['path'];assert p.stat().st_size==r['bytes'] and sha256(p)==r['sha256']
    t=np.arange(512)/256-.5;base=(t>=-.2)&(t<0);checked=0;max_guard_error=0;metriclookup={}
    for task,tc in cfg['tasks'].items():
        records=json.loads((out/f'{task}_metrics.json').read_text())['rows']
        op=json.loads((out/f'{task}_operations.json').read_text());assert op['development']==cfg['development']
        roi=json.loads((out/f'{task}_metrics.json').read_text())['roi'];signal=(t>=tc['window_s'][0])&(t<tc['window_s'][1])
        for s in range(1,41):
            stem=f'{task}_{s:03d}';identity=dict(np.load(cache/f'{stem}_identity.npz'));pred=dict(np.load(out/f'{stem}_predictions.npz'));y=identity['label']
            np.testing.assert_array_equal(pred['label'],y);np.testing.assert_array_equal(pred['sample'],identity['sample'])
            metadata=json.loads((cache/f'{stem}.json').read_text());assert metadata['config_sha256']==report['config_sha256']
            assert metadata['code_sha256']==report['code_sha256']
            assert len(y)==(160 if task=='N170' else 200)
            assert tuple(np.bincount(y))==((80,80) if task=='N170' else (160,40))
            p=np.load(cache/f'{stem}_parent.npy').astype(float)
            eye=np.load(cache/f'{stem}_eog.npy').astype(float)
            ed=np.load(out/f'{task}_eog_decoder.npz')
            ef=np.concatenate([eye[:,:,(t>=a)&(t<a+.1)].mean(-1) for a in np.arange(0,.8,.1)],1)
            es=((ef-ed['scale_mean'])/ed['scale'])@ed['coef'].T+ed['intercept']
            epred=ed['classes'][(es[:,0]>0).astype(int)]
            np.testing.assert_array_equal(epred,pred['eog_only'])
            evalue=eye[:,:,signal].mean(-1)-eye[:,:,base].mean(-1)
            econtrast=float(np.max(abs(evalue[y==1].mean(0)-evalue[y==0].mean(0))))
            for m in cfg['teachers']:
                x=np.load(cache/f'{stem}_{m}.npy').astype(float);assert x.shape==p.shape and np.isfinite(x).all()
                row=next(r for r in records if r['subject']==s and r['method']==m);metriclookup[(task,s,m)]=row
                assert abs(balanced_accuracy_score(y,epred)-row['eog_only_ba'])<1e-12
                assert abs(econtrast-row['eog_max_contrast_uv'])<2e-5
                # Compute per-trial measurement first, rather than grand waveform first.
                values=x[:,roi][:,:,signal].mean((1,2))-x[:,roi][:,:,base].mean((1,2))
                amp=values[y==1].mean()-values[y==0].mean()
                assert abs(amp-row['amplitude_uv'])<2e-5
                rms=float(np.sqrt(np.mean((x-p)**2)))
                # The runner subtracts delivered float32 arrays before conversion;
                # this independent float64 subtraction can differ by one rounding.
                assert abs(rms-row['changed_voltage_rms_uv'])<=4*np.finfo(np.float32).eps*max(1,rms)
                for decoder,key in [('parent','frozen_ba'),(m,'adapted_ba')]:
                    d=np.load(out/f'{task}_{decoder}_decoder.npz')
                    feature=np.concatenate([x[:,:,(t>=a)&(t<a+.1)].mean(-1) for a in np.arange(0,.8,.1)],1)
                    scores=((feature-d['scale_mean'])/d['scale'])@d['coef'].T+d['intercept']
                    recomputed=d['classes'][(scores[:,0]>0).astype(int)]
                    saved=pred[m if decoder=='parent' else m+'_adapted']
                    np.testing.assert_array_equal(recomputed,saved)
                    assert abs(balanced_accuracy_score(y,recomputed)-row[key])<1e-12
                if m.endswith('_guard'):
                    z=np.load(cache/f'{stem}_{m[:-6]}.npy').astype(float)
                    # Closest feasible correction is constant in each disjoint window.
                    correction=np.zeros_like(x)
                    correction[:,:,signal]=(p[:,:,signal].mean(-1)-z[:,:,signal].mean(-1))[:,:,None]
                    correction[:,:,base]=(p[:,:,base].mean(-1)-z[:,:,base].mean(-1))[:,:,None]
                    np.testing.assert_allclose(x,z+correction,atol=1e-4,rtol=2e-7)
                    errors=np.r_[((x[:,:,signal].mean(-1)-x[:,:,base].mean(-1))-(p[:,:,signal].mean(-1)-p[:,:,base].mean(-1))).ravel(),(x[:,:,base].mean(-1)-p[:,:,base].mean(-1)).ravel()]
                    max_guard_error=max(max_guard_error,float(np.max(abs(errors))));assert np.max(abs(errors))<1e-4
                checked+=1
        # Probe guard must retain the known digital measurement for every amplitude/background.
        probes=json.loads((out/f'{task}_probes.json').read_text())
        for r in probes['rows']:
            assert all(np.isfinite(r[k]) for k in ['mean_endpoint_gain','minimum_gain','maximum_gain'])
            if r['method'] in ['parent','regression_guard','femto_guard']:assert abs(r['mean_endpoint_gain']-1)<1e-10
    # Reconstruct calibration scores without importing the experiment's scoring helper.
    scores=[]
    for s in cfg['calibration']+cfg['evaluation']:
        values=[]
        for task in cfg['tasks']:
            p=metriclookup[(task,s,'parent')]
            for m in cfg['teachers'][1:]:
                r=metriclookup[(task,s,m)];values.extend([abs(r['amplitude_uv']-p['amplitude_uv']),abs(r['frozen_ba']-p['frozen_ba'])/.02])
        scores.append(max(values))
    q=sorted(scores[:10])[math.ceil(11*.9)-1]
    assert abs(q-report['conformal']['joint_q90'])<1e-12
    assert np.mean(np.asarray(scores[10:])<=q)==report['conformal']['evaluation_joint_coverage']
    for row in report['summary']:
        rr=[metriclookup[(row['task'],s,row['method'])] for s in cfg['evaluation']]
        for source,key in [('amplitude_uv','mean_amplitude_uv'),('frozen_ba','mean_frozen_ba'),('adapted_ba','mean_adapted_ba')]:assert abs(np.mean([r[source] for r in rr])-row[key])<1e-12
    write_json(out/'verification.json',dict(status='passed',source_files_sha256_verified=len(manifest['files']),participant_task_method_records_rescored=checked,
        complete_cohort_trials=14400,evaluation_trials=3600,split_isolation=True,decoder_predictions_recomputed=True,eog_control_predictions_and_contrasts_recomputed=True,
        projection_minimality_and_endpoints_verified=True,max_guard_measurement_error_uv=max_guard_error,conformal_scores_and_coverage_recomputed=True,
        scope='Numerical/provenance verification; not a cortical-fidelity or hardware-calibration certificate',verifier_code_sha256=sha256(Path(__file__))))
    print(f'Verified {checked} ERP records and {len(manifest["files"])} source files; guard maximum error {max_guard_error:.3g} uV')

if __name__=='__main__':main()
