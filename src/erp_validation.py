"""Independent ERP measurement assessment. All fitted operations use development people."""
import csv,json,sys,warnings
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,resample_poly
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import mne,torch
from measurement_math import endpoint_weights,preserve_measurement,conformal_quantile

C=json.loads((ROOT/'tim_validation_config.json').read_text())
D=ROOT/'data'/'erp_core'; CACHE=ROOT/'data'/'erp_validation'; OUT=ROOT/'results'/'erp_validation'
TIMES=np.arange(512)/256-.5
torch.set_num_threads(2)

def baseline(x):return x-x[..., (TIMES>=-.2)&(TIMES<0)].mean(-1,keepdims=True)
def weights(task):return endpoint_weights(TIMES,C['tasks'][task]['window_s'],C['baseline_s'])
def guard_operator(task):
    b=((TIMES>=-.2)&(TIMES<0)).astype(float);b/=b.sum()
    return np.stack([weights(task),b])
def split(sub):return next(k for k in ['development','calibration','evaluation'] if sub in C[k])
def stem(task,sub):return f'{task}_{sub:03d}'
def data_path(task,sub,method):return CACHE/f'{stem(task,sub)}_{method}.npy'

def continuous_filter(x,hp,bounds):
    sos=butter(4,[hp,30],btype='bandpass',fs=256,output='sos')
    out=np.full_like(x,np.nan)
    for a,b in zip(bounds[:-1],bounds[1:]):
        if b-a>3*256:out[:,a:b]=sosfiltfilt(sos,x[:,a:b])
    return out

def prepare(task,sub):
    prefix=stem(task,sub);meta=CACHE/f'{prefix}.json'
    if meta.exists():return json.loads(meta.read_text())
    p=D/f'sub-{sub:03d}'/'eeg'/f'sub-{sub:03d}_task-{task}_eeg.set'
    raw=mne.io.read_raw_eeglab(p,preload=True,verbose='ERROR')
    assert raw.info['sfreq']==1024 and len(raw.ch_names)==33
    x=resample_poly(raw.get_data()*1e6,1,4,axis=-1)
    x[:30]-=x[:30].mean(0,keepdims=True)
    boundary_mask=np.array([str(d).lower()=='boundary' for d in raw.annotations.description])
    boundary=np.round(raw.annotations.onset[boundary_mask]*256).astype(int)
    bounds=np.unique(np.r_[0,boundary,x.shape[-1]]).clip(0,x.shape[-1])
    events_path=p.with_name(p.name.replace('_eeg.set','_events.tsv'))
    events=pd.read_csv(events_path,sep='\t');events['code']=events['value'].astype(int)
    if task=='N170':
        valid=events['event_type'].isin(['face','car']);ev=events[valid].copy();y=(ev['event_type']=='face').astype(int).to_numpy()
    else:
        code=events['code'];valid=(code//10>=1)&(code//10<=5)&(code%10>=1)&(code%10<=5)
        ev=events[valid].copy();code=ev['code'].to_numpy();y=(code//10==code%10).astype(int)
    samples=np.round(ev['sample'].to_numpy()/4).astype(int)
    offsets=np.arange(512)-128;idx=samples[:,None]+offsets
    keep=(idx.min(1)>=0)&(idx.max(1)<x.shape[1]);exclusions=[]
    for i in np.flatnonzero(~keep):exclusions.append(dict(event_sample=int(samples[i]),reason='epoch beyond recording'))
    # No epoch may cross a recording boundary or contain a nonfinite sample.
    for i in np.flatnonzero(keep):
        if any(idx[i,0]<b<=idx[i,-1] for b in bounds[1:-1]) or not np.isfinite(x[:,idx[i]]).all():
            keep[i]=False;exclusions.append(dict(event_sample=int(samples[i]),reason='boundary or nonfinite'))
    idx=idx[keep];y=y[keep];samples=samples[keep]
    for method,hp in [('parent',.1),('hp0.5',.5),('hp1',1),('hp2',2)]:
        f=continuous_filter(x,hp,bounds)
        epochs=baseline(f[:,idx].transpose(1,0,2))
        assert np.isfinite(epochs).all(),f'Nonfinite filtered epoch {prefix}'
        np.save(data_path(task,sub,method),epochs[:,:30].astype(np.float32))
        if method=='parent':np.save(data_path(task,sub,'eog'),epochs[:,30:].astype(np.float32))
        if sub in C['development'] and method in ['parent','hp1']:
            good=np.flatnonzero(np.isfinite(f).all(0));inds=good[np.linspace(0,len(good)-1,C['ica_samples_per_development_person']).astype(int)]
            np.save(CACHE/f'{prefix}_{method}_fit.npy',f[:,inds].T.astype(np.float32))
    np.savez(CACHE/f'{prefix}_identity.npz',label=y,sample=samples)
    m=dict(task=task,subject=sub,split=split(sub),channels=raw.ch_names,sampling_hz=1024,
           n_epochs=len(y),class_counts=np.bincount(y,minlength=2).tolist(),bounds_256=bounds.tolist(),exclusions=exclusions,
           source_set_sha256=sha256(p),source_fdt_sha256=sha256(p.with_suffix('.fdt')),event_sha256=sha256(events_path),
           config_sha256=sha256(ROOT/'tim_validation_config.json'),code_sha256=sha256(Path(__file__)))
    write_json(meta,m);print(f'Prepared {prefix}: {m["class_counts"]}',flush=True);return m

def fit_operations(task):
    dst=OUT/f'{task}_operations.npz';info=OUT/f'{task}_operations.json'
    if dst.exists():return dict(np.load(dst))
    parent=np.concatenate([np.load(CACHE/f'{stem(task,s)}_parent_fit.npy') for s in C['development']]).astype(float)
    hi=np.concatenate([np.load(CACHE/f'{stem(task,s)}_hp1_fit.npy') for s in C['development']]).astype(float)
    x,e=parent[:,:30],parent[:,30:];mx=x.mean(0);me=e.mean(0);ec=e-me
    gram=ec.T@ec;beta=np.linalg.solve(gram+C['regression_ridge_fraction']*np.trace(gram)/3*np.eye(3),ec.T@(x-mx))
    ica=FastICA(n_components=29,whiten='unit-variance',random_state=C['seed'],max_iter=C['ica_max_iter'],tol=.001)
    with warnings.catch_warnings(record=True) as ww:
        z=ica.fit_transform(hi[:,:30]);messages=[str(w.message) for w in ww]
    ehi=hi[:,30:];corr=np.corrcoef(np.c_[z,ehi],rowvar=False)[:29,29:]
    strength=np.max(abs(corr),axis=1);bad=np.flatnonzero(strength>C['ica_eog_correlation_threshold'])
    bad=bad[np.argsort(strength[bad])[::-1]][:C['ica_max_removed']]
    # Apply the fixed ICA decomposition to the 0.1-Hz parent, retaining its mean.
    remove=ica.mixing_[:,bad]@ica.components_[bad] if len(bad) else np.zeros((30,30))
    operations=dict(beta=beta,eog_mean=me,ica_remove=remove,ica_mean=ica.mean_)
    np.savez(dst,**operations)
    write_json(info,dict(development=C['development'],n_fit_samples=len(parent),ica_iterations=int(ica.n_iter_),
        ica_removed=bad.tolist(),ica_eog_correlations=corr.tolist(),warnings=messages,
        ica_fit_input='Continuous 1-30 Hz development EEG; fixed matrix applied to parent',
        config_sha256=sha256(ROOT/'tim_validation_config.json'),code_sha256=sha256(Path(__file__))))
    print(f'Fitted {task} operations; ICA excluded {bad.tolist()}; warnings {messages}',flush=True)
    return operations

def load_femto():
    sys.path.insert(0,str(ROOT/'src'/'legacy'))
    from models import conv6SingleHeadNet,deeperHead,deeperEncoder,DWConv1d
    p=ROOT/'models'/'femto_2561_state_dict.pt'
    model=conv6SingleHeadNet(ch=8,groups=8,scale_up=1,device='cpu')
    allow=[conv6SingleHeadNet,deeperHead,deeperEncoder,DWConv1d,torch.nn.Conv1d,torch.nn.GroupNorm,torch.nn.Sequential,torch.nn.SiLU]
    with torch.serialization.safe_globals(allow):state=torch.load(p,map_location='cpu',weights_only=True)
    if isinstance(state,torch.nn.Module):state=state.state_dict()
    model.load_state_dict(state);model.eval()
    write_json(OUT/'femto_checkpoint.json',dict(path=str(p),sha256=sha256(p),parameters=sum(p.numel() for p in model.parameters()),device='cpu',
        input='Parent epochs, per-channel window standard deviation normalization, scale restored, baseline reapplied',
        independence='ERP CORE is a separate acquisition cohort from BCI IV 2a; no ERP fine tuning. Original training source-person identities unavailable.'))
    return model

def femto(x,model):
    shape=x.shape;a=x.reshape(-1,512);scale=a.std(1,keepdims=True)
    if np.any(scale<1e-9):raise ValueError('Degenerate model input')
    out=[]
    with torch.no_grad():
        for i in range(0,len(a),256):out.append(model(torch.from_numpy((a[i:i+256]/scale[i:i+256]).astype(np.float32))[:,None])[0][:,0].numpy())
    return baseline((np.concatenate(out)*scale).reshape(shape))

def apply_operation(x,e,method,op,model=None,task=None):
    if method=='parent':return x
    if method in ['regression','regression_guard']:
        out=baseline(x-np.einsum('net,ec->nct',e-op['eog_mean'][None,:,None],op['beta']))
    elif method=='ica':
        out=baseline(x-np.einsum('ck,nkt->nct',op['ica_remove'],x-op['ica_mean'][None,:,None]))
    elif method in ['femto','femto_guard']:out=femto(x,model)
    else:raise ValueError(method)
    return preserve_measurement(x,out,guard_operator(task)) if method.endswith('_guard') else out

def transform(task,sub,op,model):
    x=np.load(data_path(task,sub,'parent'));e=np.load(data_path(task,sub,'eog'))
    for method in ['regression','ica','femto','regression_guard','femto_guard']:
        p=data_path(task,sub,method)
        if p.exists():continue
        if method.endswith('_guard'):
            out=preserve_measurement(x,np.load(data_path(task,sub,method[:-6])),guard_operator(task))
        else:out=apply_operation(x,e,method,op,model,task)
        np.save(p,out.astype(np.float32))
    print(f'Transformed {stem(task,sub)}',flush=True)

def feature(x):
    return np.concatenate([x[:,:,(TIMES>=a)&(TIMES<a+.1)].mean(-1) for a in np.arange(0,.8,.1)],axis=1)

def fit_decoder(features,labels):
    scaler=StandardScaler().fit(features)
    clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]).fit(scaler.transform(features),labels)
    return scaler,clf

def predict(dec,x):return dec[1].predict(dec[0].transform(feature(x)))

def summary_ci(values):
    a=np.asarray(values,float);rng=np.random.default_rng(C['seed'])
    return np.quantile(rng.choice(a,(C['bootstrap_draws'],len(a)),replace=True).mean(1),[.025,.975]).tolist()

def assess(task):
    out=OUT/f'{task}_metrics.json'
    if out.exists():return json.loads(out.read_text())
    ids={s:dict(np.load(CACHE/f'{stem(task,s)}_identity.npz')) for s in range(1,41)}
    ytr=np.concatenate([ids[s]['label'] for s in C['development']])
    train_feature={m:np.concatenate([feature(np.load(data_path(task,s,m))) for s in C['development']]) for m in C['teachers']+['eog']}
    dec={m:fit_decoder(f,ytr) for m,f in train_feature.items()}
    for m,(sc,clf) in dec.items():np.savez(OUT/f'{task}_{m}_decoder.npz',scale_mean=sc.mean_,scale=sc.scale_,coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
    quality=lambda x,e:np.log(np.maximum(np.ptp(x,axis=-1).max(1),1e-9))+np.log(np.maximum(np.ptp(e,axis=-1).max(1),1e-9))
    qtrain=np.concatenate([quality(np.load(data_path(task,s,'parent')),np.load(data_path(task,s,'eog'))) for s in C['development']])
    threshold=float(np.quantile(qtrain,C['quality_quantile']));rows=[];roi=[json.loads((CACHE/f'{stem(task,1)}.json').read_text())['channels'].index(c) for c in C['tasks'][task]['roi']]
    erps=[]
    for s in range(1,41):
        y=ids[s]['label'];parent=np.load(data_path(task,s,'parent'));e=np.load(data_path(task,s,'eog'));keep=quality(parent,e)<=threshold
        saved=dict(label=y,sample=ids[s]['sample'],retained=keep)
        epredict=predict(dec['eog'],e);saved['eog_only']=epredict
        econ=e[y==1].mean(0)-e[y==0].mean(0)
        eamp=float(np.max(abs(econ@weights(task))))
        for m in C['teachers']:
            x=np.load(data_path(task,s,m));pred=predict(dec['parent'],x);adapt=predict(dec[m],x)
            contrast=x[y==1].mean(0)-x[y==0].mean(0);amp=float(contrast[roi].mean(0)@weights(task));erps.append(contrast[roi].mean(0))
            saved[m]=pred;saved[m+'_adapted']=adapt
            rows.append(dict(task=task,subject=s,split=split(s),method=m,n=len(y),class_counts=np.bincount(y,minlength=2).tolist(),
                amplitude_uv=amp,frozen_ba=float(balanced_accuracy_score(y,pred)),adapted_ba=float(balanced_accuracy_score(y,adapt)),
                eog_only_ba=float(balanced_accuracy_score(y,epredict)),eog_max_contrast_uv=eamp,
                retained_fraction=float(keep.mean()),class_retained=[float(keep[y==k].mean()) for k in [0,1]],
                retained_frozen_ba=float(balanced_accuracy_score(y[keep],pred[keep])) if len(np.unique(y[keep]))==2 else None,
                changed_voltage_rms_uv=float(np.sqrt(np.mean((x-parent).astype(float)**2)))))
        np.savez_compressed(OUT/f'{stem(task,s)}_predictions.npz',**saved)
    np.save(OUT/f'{task}_roi_contrasts.npy',np.asarray(erps).reshape(40,len(C['teachers']),512))
    report=dict(rows=rows,quality_threshold=threshold,training_subjects=C['development'],roi=roi,
                config_sha256=sha256(ROOT/'tim_validation_config.json'),code_sha256=sha256(Path(__file__)))
    write_json(out,report);print(f'Assessed {task}',flush=True);return report

def probes(task,op,model):
    dst=OUT/f'{task}_probes.json'
    if dst.exists():return json.loads(dst.read_text())
    cfg=C['tasks'][task];names=json.loads((CACHE/f'{stem(task,1)}.json').read_text())['channels'];roi=[names.index(c) for c in cfg['roi']]
    spatial=np.full(30,-len(roi)/(30-len(roi)));spatial[roi]=1
    tlong=np.arange(32*256)/256-16
    g=np.exp(-.5*((tlong-cfg['probe_center_s'])/cfg['probe_sd_s'])**2)
    full=spatial[:,None]*g;sl=np.arange(512)+int(15.5*256)
    probe=baseline(full[:,sl]);den=float(probe[roi].mean(0)@weights(task));rows=[]
    for s in C['evaluation']:
        ids=np.load(CACHE/f'{stem(task,s)}_identity.npz');selected=np.r_[np.flatnonzero(ids['label']==0)[:4],np.flatnonzero(ids['label']==1)[:4]]
        x=np.load(data_path(task,s,'parent'))[selected].astype(float);e=np.load(data_path(task,s,'eog'))[selected].astype(float)
        base={m:apply_operation(x,e,m,op,model,task) for m in ['parent','regression','ica','femto','regression_guard','femto_guard']}
        for a in C['probe_amplitudes_uv']:
            for m in C['teachers']:
                if m.startswith('hp'):
                    hp=float(m[2:]);response=baseline(sosfiltfilt(butter(4,[hp,30],btype='bandpass',fs=256,output='sos'),a*full)[:,sl])
                    gain=np.repeat(float(response[roi].mean(0)@weights(task))/(a*den),len(x))
                else:
                    response=apply_operation(x+a*probe,e,m,op,model,task)-base[m]
                    gain=(response[:,roi].mean(1)@weights(task))/(a*den)
                rows.append(dict(task=task,subject=s,method=m,amplitude_uv=a,n_backgrounds=len(x),mean_endpoint_gain=float(gain.mean()),
                                 minimum_gain=float(gain.min()),maximum_gain=float(gain.max())))
    result=dict(rows=rows,probe_endpoint_per_unit_uv=den,spatial=spatial.tolist(),
        control='Known digital voltage, not independently measured cortical truth. Filter arm measures continuous filter transfer on 32-second probe; other arms paired after parent filtering.',
        code_sha256=sha256(Path(__file__)),config_sha256=sha256(ROOT/'tim_validation_config.json'))
    write_json(dst,result);print(f'Probe controls complete {task}',flush=True);return result

def finalize(reports):
    allrows=[r for report in reports for r in report['rows']];lookup={(r['task'],r['subject'],r['method']):r for r in allrows};summary=[]
    for task in C['tasks']:
        for m in C['teachers']:
            rows=[lookup[(task,s,m)] for s in C['evaluation']];parents=[lookup[(task,s,'parent')] for s in C['evaluation']]
            amp=np.array([r['amplitude_uv'] for r in rows]);ad=amp-np.array([r['amplitude_uv'] for r in parents]);ba=np.array([r['frozen_ba'] for r in rows]);bd=ba-np.array([r['frozen_ba'] for r in parents])
            summary.append(dict(task=task,method=m,evaluation_participants=10,mean_amplitude_uv=float(amp.mean()),amplitude_difference_uv=float(ad.mean()),amplitude_difference_ci95=summary_ci(ad),
                mean_frozen_ba=float(ba.mean()),frozen_ba_difference=float(bd.mean()),frozen_ba_difference_ci95=summary_ci(bd),mean_adapted_ba=float(np.mean([r['adapted_ba'] for r in rows])),
                mean_eog_only_ba=float(np.mean([r['eog_only_ba'] for r in rows])),mean_eog_contrast_uv=float(np.mean([r['eog_max_contrast_uv'] for r in rows])),
                mean_retained_fraction=float(np.mean([r['retained_fraction'] for r in rows])),mean_retained_frozen_ba=float(np.mean([r['retained_frozen_ba'] for r in rows])),
                mean_changed_voltage_rms_uv=float(np.mean([r['changed_voltage_rms_uv'] for r in rows]))))
    scores=[];method_scores={m:[] for m in C['teachers'][1:]}
    for s in C['calibration']+C['evaluation']:
        vals=[]
        for m in C['teachers'][1:]:
            v=[]
            for task in C['tasks']:
                r,p=lookup[(task,s,m)],lookup[(task,s,'parent')]
                v.extend([abs(r['amplitude_uv']-p['amplitude_uv'])/C['endpoint_tolerances']['amplitude_uv'],abs(r['frozen_ba']-p['frozen_ba'])/C['endpoint_tolerances']['balanced_accuracy']])
            method_scores[m].append(max(v));vals+=v
        scores.append(dict(subject=s,split=split(s),joint_score=max(vals)))
    cal=np.array([r['joint_score'] for r in scores if r['split']=='calibration']);te=np.array([r['joint_score'] for r in scores if r['split']=='evaluation']);q=conformal_quantile(cal,C['conformal_alpha'])
    method_cal={m:dict(q90=conformal_quantile(v[:10],.1),evaluation_coverage=float(np.mean(np.asarray(v[10:])<=conformal_quantile(v[:10],.1))),evaluation_tolerance_pass_fraction=float(np.mean(np.asarray(v[10:])<=1))) for m,v in method_scores.items()}
    write_json(OUT/'summary.json',dict(summary=summary,conformal=dict(alpha=.1,calibration_n=10,evaluation_n=10,joint_q90=q,evaluation_joint_coverage=float(np.mean(te<=q)),scores=scores,
        q95='infinite: ceil(11*0.95)=11 exceeds ten calibration observations',method_marginal=method_cal,
        scope='Simultaneous parent-relative observable endpoint changes for one new exchangeable participant; no cortical-error coverage'),
        config=C,config_sha256=sha256(ROOT/'tim_validation_config.json'),code_sha256=sha256(Path(__file__))))
    pd.DataFrame(allrows).to_csv(OUT/'participant_metrics.csv',index=False)
    print(f'ERP validation complete; joint q90 {q:.3f}; held-out coverage {np.mean(te<=q):.2f}',flush=True)

def main():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    assert (D/'download_manifest.json').exists(),'Require checksum-verified complete cohort before execution'
    write_json(OUT/'execution_start.json',dict(config_sha256=sha256(ROOT/'tim_validation_config.json'),code_sha256=sha256(Path(__file__)),data_manifest_sha256=sha256(D/'download_manifest.json')))
    model=load_femto();reports=[]
    for task in C['tasks']:
        for s in range(1,41):prepare(task,s)
        op=fit_operations(task)
        for s in range(1,41):transform(task,s,op,model)
        reports.append(assess(task));probes(task,op,model)
    finalize(reports)

if __name__=='__main__':main()
