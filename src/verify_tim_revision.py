"""Independently rescore the revised delivered voltages and statistical envelopes."""
import json,math
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def main():
    out=ROOT/'results/revision';cache=ROOT/'data/tim_revision';old=ROOT/'data/erp_validation'
    cfg=json.loads((ROOT/'tim_revision_config.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert summary['config']==cfg and summary['config_sha256']==sha256(ROOT/'tim_revision_config.json')
    sets=[set(cfg[k]) for k in ['development','calibration','evaluation']]
    assert list(map(len,sets))==[10,20,10] and sum(map(len,sets))==len(set.union(*sets))==40
    t=np.arange(512)/256-.5;base=(t>=-.2)&(t<0);checked=0;lookup={};max_guard=0
    matrices=np.load(out/'icunet_montage.npz');names=json.loads((old/'N170_001.json').read_text())['channels']
    from independent_denoiser import TEMPLATE
    pz=names.index('Pz');template_pz=TEMPLATE.index('Pz')
    assert matrices['forward'][template_pz,pz]==1 and matrices['backward'][pz,template_pz]==1
    for task,tc in cfg['tasks'].items():
        report=json.loads((out/f'{task}_metrics.json').read_text());roi=[names.index(c) for c in tc['roi']]
        signal=(t>=tc['window_s'][0])&(t<tc['window_s'][1])
        assert json.loads((out/f'{task}_operations.json').read_text())['development']==cfg['development']
        op=np.load(out/f'{task}_operations.npz')
        for s in range(1,41):
            y=np.load(old/f'{task}_{s:03d}_identity.npz')['label'];pred=np.load(out/f'{task}_{s:03d}_predictions.npz')
            parent=np.load(old/f'{task}_{s:03d}_parent.npy').astype(float)
            eye=np.load(old/f'{task}_{s:03d}_eog.npy').astype(float)
            for m in report['methods']:
                where=old if m in ['parent','hp0.5','hp1','hp2','femto'] else cache
                delivered=np.load(where/f'{task}_{s:03d}_{m}.npy')
                x=delivered.astype(float)
                assert x.shape==parent.shape and np.isfinite(x).all()
                r=next(r for r in report['rows'] if r['subject']==s and r['method']==m);lookup[task,s,m]=r
                values=x[:,roi][:,:,signal].mean((1,2))-x[:,roi][:,:,base].mean((1,2))
                amp=values[y==1].mean()-values[y==0].mean()
                assert abs(amp-r['amplitude_uv'])<2e-10
                # Reproduce the delivered float32 feature and scaler convention.
                # Voltage endpoints above are independently accumulated in float64.
                f=np.concatenate([delivered[:,:,(t>=a)&(t<a+.1)].mean(-1,dtype=np.float32) for a in np.arange(0,.8,.1)],axis=1)
                for decoder,key,saved in [('parent','frozen_ba',m),(m,'adapted_ba',m+'_adapted')]:
                    d=np.load(out/f'{task}_{decoder}_decoder.npz')
                    standardized=f.copy();standardized-=d['scale_mean'].astype(f.dtype);standardized/=d['scale'].astype(f.dtype)
                    score=standardized@d['coef'].T+d['intercept']
                    actual=d['classes'][(score[:,0]>0).astype(int)]
                    np.testing.assert_array_equal(actual,pred[saved],err_msg=f'{task}/{s}/{m}/{decoder}');assert abs(balanced_accuracy_score(y,actual)-r[key])<1e-12
                if m in ['regression','ica']:
                    if m=='regression':expected=parent-np.einsum('net,ec->nct',eye-op['eog_mean'][None,:,None],op['beta'])
                    else:expected=parent-np.einsum('ck,nkt->nct',op['ica_remove'],parent-op['ica_mean'][None,:,None])
                    expected-=expected[:,:,base].mean(-1,keepdims=True)
                    np.testing.assert_allclose(x,expected,atol=1e-4,rtol=3e-7)
                if m in ['femto_guard','femto_smooth','femto_bandlimited','icunet_smooth']:
                    window_change=(x-parent)[:,:,signal].mean(-1)
                    baseline_change=(x-parent)[:,:,base].mean(-1)
                    error=np.r_[(window_change-baseline_change).ravel(),baseline_change.ravel()]
                    max_guard=max(max_guard,float(np.max(abs(error))));assert np.max(abs(error))<1e-4
                checked+=1
        for r in summary['summary']:
            if r['task']!=task:continue
            rr=[lookup[task,s,r['method']] for s in cfg['evaluation']]
            for key,source in [('mean_amplitude_uv','amplitude_uv'),('mean_frozen_ba','frozen_ba'),('mean_adapted_ba','adapted_ba')]:assert abs(np.mean([p[source] for p in rr])-r[key])<1e-12
    values=[]
    for s in cfg['calibration']+cfg['evaluation']:
        v=[]
        for task in cfg['tasks']:
            p=lookup[task,s,'parent']
            for m in cfg['calibration_family']:
                r=lookup[task,s,m];v.extend([abs(r['amplitude_uv']-p['amplitude_uv']),abs(r['frozen_ba']-p['frozen_ba'])/.02])
        values.append(max(v))
    for alpha in [.1,.05]:
        q=sorted(values[:20])[math.ceil(21*(1-alpha))-1];row=summary['calibration']['joint'][f'q{int((1-alpha)*100)}']
        assert abs(q-row['quantile'])<1e-12 and int(np.sum(np.asarray(values[20:])<=q))==row['evaluation_covered']
    probes=json.loads((out/'probes.json').read_text());smooth=json.loads((out/'smooth_probes.json').read_text())
    for record in [probes,smooth]:
        assert len(record['rows'])==11520
        for row in record['summary']:
            g=np.array([r['gain'] for r in record['rows'] if r['task']==row['task'] and r['method']==row['method']])
            assert len(g)==480 and abs(g.mean()-row['mean_gain'])<1e-12
            if row['method'] in ['parent','regression','femto_guard','femto_smooth','icunet_smooth']:assert np.max(abs(g-1))<1e-10
    write_json(out/'verification.json',dict(status='passed',participant_task_method_records_rescored=checked,development_calibration_evaluation=[10,20,10],
        strict_split_isolation=True,decoder_predictions_recomputed=True,classical_operation_voltages_recomputed=True,
        decoder_numerical_convention='Float32 delivered-window means; scaler parameters cast to feature dtype before in-place standardization, as in the fitted sklearn pipeline; saved LDA coefficient dtype preserved. Voltage endpoints accumulated in float64.',
        maximum_guard_constraint_error_uv=max_guard,rank_quantiles_and_evaluation_coverage_recomputed=True,
        probe_records_checked=23040,probe_dispersion_aggregates_recomputed=True,model_montage_pz_identity_checked=True,
        config_sha256=sha256(ROOT/'tim_revision_config.json'),verifier_sha256=sha256(Path(__file__)),
        scope='Numerical validation of delivered digital outputs; source purity and prospective population performance remain separate claims.'))
    print('Verified',checked,'records and 23040 probe records; maximum guard error',max_guard,flush=True)

if __name__=='__main__':main()
