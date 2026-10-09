"""Reviewer-directed ERP assessment; fitted stages use ten development people only."""
import json,warnings,csv
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.metrics import balanced_accuracy_score
from erp_validation import baseline,feature,fit_decoder,weights,guard_operator,stem
from measurement_math import preserve_measurement,conformal_quantile
from revision_math import smooth_guard

C=json.loads((ROOT/'tim_revision_config.json').read_text())
OUT=ROOT/'results/revision';CACHE=ROOT/'data/tim_revision';OLD=ROOT/'data/erp_validation'
STATIC={'parent','hp0.5','hp1','hp2','femto'}

def path(task,subject,method):
    return (OLD if method in STATIC else CACHE)/f'{stem(task,subject)}_{method}.npy'

def partition(s):return next(k for k in ['development','calibration','evaluation'] if s in C[k])

def ci(values):
    a=np.asarray(values,float);rng=np.random.default_rng(C['seed'])
    return np.quantile(rng.choice(a,(C['bootstrap_draws'],len(a)),replace=True).mean(1),[.025,.975]).tolist()

def operations(task):
    dst=OUT/f'{task}_operations.npz'
    if dst.exists():return dict(np.load(dst))
    p=np.concatenate([np.load(OLD/f'{stem(task,s)}_parent_fit.npy') for s in C['development']]).astype(float)
    h=np.concatenate([np.load(OLD/f'{stem(task,s)}_hp1_fit.npy') for s in C['development']]).astype(float)
    x,e=p[:,:30],p[:,30:];me=e.mean(0);ec=e-me;gram=ec.T@ec
    beta=np.linalg.solve(gram+1e-5*np.trace(gram)/3*np.eye(3),ec.T@(x-x.mean(0)))
    ica=FastICA(n_components=29,whiten='unit-variance',random_state=C['seed'],max_iter=1000,tol=.001)
    with warnings.catch_warnings(record=True) as warning:
        z=ica.fit_transform(h[:,:30])
    corr=np.corrcoef(np.c_[z,h[:,30:]],rowvar=False)[:29,29:];strength=np.max(abs(corr),axis=1)
    bad=np.flatnonzero(strength>.3);bad=bad[np.argsort(strength[bad])[::-1]][:3]
    op=dict(beta=beta,eog_mean=me,ica_remove=ica.mixing_[:,bad]@ica.components_[bad],ica_mean=ica.mean_)
    np.savez(dst,**op)
    write_json(OUT/f'{task}_operations.json',dict(development=C['development'],n_fit_samples=len(p),
        ica_iterations=int(ica.n_iter_),ica_removed=bad.tolist(),warnings=[str(w.message) for w in warning],
        config_sha256=sha256(ROOT/'tim_revision_config.json')))
    print('Fitted revision operations',task,'ICA iterations',ica.n_iter_,flush=True)
    return op

def transform(task,subject,op):
    x=np.load(path(task,subject,'parent')).astype(float);e=np.load(OLD/f'{stem(task,subject)}_eog.npy').astype(float)
    for method in ['regression','ica','femto_guard','femto_smooth','femto_bandlimited','icunet_smooth']:
        target=path(task,subject,method)
        if target.exists():continue
        if method=='regression':
            z=baseline(x-np.einsum('net,ec->nct',e-op['eog_mean'][None,:,None],op['beta']))
        elif method=='ica':z=baseline(x-np.einsum('ck,nkt->nct',op['ica_remove'],x-op['ica_mean'][None,:,None]))
        else:
            base_method='icunet' if method.startswith('icunet') else 'femto'
            z=np.load(path(task,subject,base_method)).astype(float)
            if method=='femto_guard':z=preserve_measurement(x,z,guard_operator(task))
            else:
                if method=='femto_bandlimited':
                    mask=np.fft.rfftfreq(512,1/256)<=30
                    z=np.fft.irfft(np.fft.rfft(z,axis=-1)*mask,n=512,axis=-1)
                z=smooth_guard(x,z,guard_operator(task))
        np.save(target,z.astype(np.float32))

def assess(task):
    dest=OUT/f'{task}_metrics.json'
    if dest.exists():return json.loads(dest.read_text())
    ids={s:dict(np.load(OLD/f'{stem(task,s)}_identity.npz')) for s in range(1,41)}
    labels=np.concatenate([ids[s]['label'] for s in C['development']])
    methods=C['teachers']+['femto_bandlimited','eog'];features={};decoders={}
    for m in methods:
        feats=[feature(np.load(OLD/f'{stem(task,s)}_eog.npy' if m=='eog' else path(task,s,m))) for s in range(1,41)]
        features[m]=feats;train=np.concatenate([feats[s-1] for s in C['development']])
        scaler,clf=fit_decoder(train,labels);decoders[m]=(scaler,clf)
        np.savez(OUT/f'{task}_{m}_decoder.npz',scale_mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
    offsets=np.r_[0,np.cumsum([len(ids[s]['label']) for s in range(1,41)])]
    np.savez(OUT/f'{task}_features.npz',offsets=offsets,label=np.concatenate([ids[s]['label'] for s in range(1,41)]),**{m:np.concatenate(f) for m,f in features.items()})
    names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in C['tasks'][task]['roi']]
    rows=[];waves=[];guardqa=[]
    for s in range(1,41):
        y=ids[s]['label'];parent=np.load(path(task,s,'parent')).astype(float)
        epred=decoders['eog'][1].predict(decoders['eog'][0].transform(features['eog'][s-1]))
        saved=dict(label=y,sample=ids[s]['sample'],eog_only=epred)
        for m in methods[:-1]:
            f=features[m][s-1];pred=decoders['parent'][1].predict(decoders['parent'][0].transform(f))
            adapt=decoders[m][1].predict(decoders[m][0].transform(f));x=np.load(path(task,s,m)).astype(float)
            contrast=x[y==1].mean(0)-x[y==0].mean(0);wave=contrast[roi].mean(0);amp=float(wave@weights(task))
            waves.append(wave);saved[m]=pred;saved[m+'_adapted']=adapt
            rows.append(dict(task=task,subject=s,split=partition(s),method=m,n=len(y),amplitude_uv=amp,
                frozen_ba=float(balanced_accuracy_score(y,pred)),adapted_ba=float(balanced_accuracy_score(y,adapt)),
                eog_only_ba=float(balanced_accuracy_score(y,epred)),changed_voltage_rms_uv=float(np.sqrt(np.mean((x-parent)**2)))))
            if m in ['femto_guard','femto_smooth','femto_bandlimited','icunet_smooth']:
                source_method='icunet' if m.startswith('icunet') else 'femto'
                raw=np.load(path(task,s,source_method)).astype(float)
                if m=='femto_bandlimited':raw=np.fft.irfft(np.fft.rfft(raw,axis=-1)*(np.fft.rfftfreq(512,1/256)<=30),n=512,axis=-1)
                correction=x-raw;power=np.abs(np.fft.rfft(correction,axis=-1))**2
                residual=(x-parent)@guard_operator(task).T
                guardqa.append(dict(task=task,subject=s,method=m,maximum_constraint_error_uv=float(np.max(abs(residual))),
                    above30_correction_energy_fraction=float(power[...,np.fft.rfftfreq(512,1/256)>30].sum()/max(power.sum(),1e-30)),
                    correction_rms_uv=float(np.sqrt(np.mean(correction**2)))))
        np.savez_compressed(OUT/f'{stem(task,s)}_predictions.npz',**saved)
    np.save(OUT/f'{task}_roi_contrasts.npy',np.asarray(waves).reshape(40,len(methods)-1,512))
    report=dict(rows=rows,guardqa=guardqa,methods=methods[:-1],development=C['development'],config_sha256=sha256(ROOT/'tim_revision_config.json'))
    write_json(dest,report);print('Assessed revision',task,len(rows),'records',flush=True);return report

def calibrate(rows):
    lookup={(r['task'],r['subject'],r['method']):r for r in rows};scores=[];result={}
    family=C['calibration_family']
    for s in C['calibration']+C['evaluation']:
        values=[]
        for m in family:
            for task in C['tasks']:
                r,p=lookup[task,s,m],lookup[task,s,'parent']
                for key,scale in [('amplitude_uv',1.),('frozen_ba',.02)]:
                    values.append(dict(method=m,task=task,endpoint=key,score=abs(r[key]-p[key])/scale))
        worst=max(values,key=lambda r:r['score']);scores.append(dict(subject=s,split=partition(s),joint_score=worst['score'],dominating=worst))
    def envelope(c,t):
        return {f'q{int((1-alpha)*100)}':dict(quantile=conformal_quantile(c,alpha),calibration_n=len(c),evaluation_n=len(t),
                evaluation_covered=int(np.sum(np.asarray(t)<=conformal_quantile(c,alpha))),
                evaluation_coverage=float(np.mean(np.asarray(t)<=conformal_quantile(c,alpha)))) for alpha in [.1,.05]}
    joint=envelope([r['joint_score'] for r in scores if r['split']=='calibration'],[r['joint_score'] for r in scores if r['split']=='evaluation'])
    for m in family:
        v=[];amp=[];ba=[]
        for s in C['calibration']+C['evaluation']:
            av=[abs(lookup[task,s,m]['amplitude_uv']-lookup[task,s,'parent']['amplitude_uv']) for task in C['tasks']]
            bv=[abs(lookup[task,s,m]['frozen_ba']-lookup[task,s,'parent']['frozen_ba']) for task in C['tasks']]
            amp.append(max(av));ba.append(max(bv));v.append(max(max(av),max(bv)/.02))
        result[m]=dict(joint=envelope(v[:20],v[20:]),voltage_uv=envelope(amp[:20],amp[20:]),balanced_accuracy=envelope(ba[:20],ba[20:]))
    return dict(joint=joint,method_specific=result,scores=scores,family=family,
        guarantee='Conditional on the ten development participants and frozen fitting steps, joint coverage concerns one new exchangeable participant across the declared seven candidates and four endpoints. Individual voltage envelopes cover both tasks only for the prespecified operation.',
        decoder_development_isolation='Scaling, LDA, ICA and regression fitting use only development subjects; none uses calibration or evaluation samples.',
        status='Follow-up exploratory assessment; broad-family envelope is a deliberate stress test, not the only actionable output.')

def partition_sensitivity():
    methods=['hp0.5','hp1','hp2','femto','mapping','icunet','femto_guard','femto_smooth'];results=[]
    for seed in C['partition_sensitivity_seeds']:
        perm=np.random.default_rng(seed).permutation(np.arange(1,41));dev=perm[:10];evaluation=perm[30:]
        for task in C['tasks']:
            store=np.load(OUT/f'{task}_features.npz');offset=store['offsets']
            slices={s:slice(offset[s-1],offset[s]) for s in range(1,41)}
            labels=np.concatenate([store['label'][slices[s]] for s in dev])
            parent_train=np.concatenate([store['parent'][slices[s]] for s in dev]);dec=fit_decoder(parent_train,labels)
            for m in methods:
                delta=[];ad=[]
                for s in evaluation:
                    y=store['label'][slices[s]]
                    pred=dec[1].predict(dec[0].transform(store[m][slices[s]]));p=dec[1].predict(dec[0].transform(store['parent'][slices[s]]))
                    delta.append(float(balanced_accuracy_score(y,pred)-balanced_accuracy_score(y,p)))
                results.append(dict(seed=seed,task=task,method=m,development=dev.tolist(),evaluation=evaluation.tolist(),mean_frozen_ba_difference=float(np.mean(delta))))
    write_json(OUT/'partition_sensitivity.json',dict(rows=results,seeds=C['partition_sensitivity_seeds'],
        scope='Twenty alternative 10/20/10 partitions for fixed, non-ERP-fitted filters, mapping and neural models; scaling and parent decoder refitted each time. ICA/regression are excluded because their operators would also require refitting. The resulting range is partition sensitivity, not a confidence interval.'))
    return results

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    write_json(OUT/'execution_start.json',dict(config_sha256=sha256(ROOT/'tim_revision_config.json'),code_sha256=sha256(Path(__file__)),
        original_config_sha256=sha256(ROOT/'tim_validation_config.json'),data_manifest_sha256=sha256(ROOT/'data/erp_core/download_manifest.json')))
    reports=[]
    for task in C['tasks']:
        op=operations(task)
        for s in range(1,41):transform(task,s,op)
        reports.append(assess(task))
    rows=[r for report in reports for r in report['rows']];lookup={(r['task'],r['subject'],r['method']):r for r in rows};summary=[]
    for task in C['tasks']:
        for m in reports[0]['methods']:
            rr=[lookup[task,s,m] for s in C['evaluation']];p=[lookup[task,s,'parent'] for s in C['evaluation']]
            ad=np.array([r['amplitude_uv']-q['amplitude_uv'] for r,q in zip(rr,p)]);bd=np.array([r['frozen_ba']-q['frozen_ba'] for r,q in zip(rr,p)])
            summary.append(dict(task=task,method=m,mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),
                amplitude_difference_uv=float(ad.mean()),amplitude_difference_ci95=ci(ad),mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr])),
                frozen_ba_difference=float(bd.mean()),frozen_ba_difference_ci95=ci(bd),mean_adapted_ba=float(np.mean([r['adapted_ba'] for r in rr])),
                mean_eog_only_ba=float(np.mean([r['eog_only_ba'] for r in rr]))))
    calibration=calibrate(rows);sensitivity=partition_sensitivity()
    write_json(OUT/'summary.json',dict(summary=summary,calibration=calibration,config=C,config_sha256=sha256(ROOT/'tim_revision_config.json'),
        code_sha256=sha256(Path(__file__)),guardqa=[r for report in reports for r in report['guardqa']],
        assessed_records=len(rows),complete_cohort_trials=14400,evaluation_trials=3600,
        scope='Digital measurement validation; no cortical-error, acquisition-uncertainty or causal attribution certificate.'))
    with (OUT/'participant_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Revision complete',len(rows),'records; joint envelopes',calibration['joint'],flush=True)

if __name__=='__main__':main()
