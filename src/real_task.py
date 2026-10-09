"""Exploratory held-out-participant motor-imagery evaluation on real EEG."""
import csv
import itertools
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.linalg import eigh
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from common import ROOT, config, sha256, write_json
import mne
import torch

torch.set_num_threads(2)

def load_data():
    cache=ROOT/'data'/'bci_epochs.npz'
    manifest_path=ROOT/'data'/'bci_manifest.json'
    if cache.exists() and manifest_path.exists():
        return dict(np.load(cache))
    c=config(); xs=[];fs=[];es=[];ys=[];subjects=[];artifacts=[];samples=[];man=[];exclusions=[]
    for sub in c['bci_subjects']:
        path=Path(c['bci_dir'])/f'A{sub:02d}T.gdf'
        raw=mne.io.read_raw_gdf(path,preload=True,verbose='ERROR')
        if raw.info['sfreq']!=250 or len(raw.ch_names)!=25:
            raise ValueError('BCI acquisition metadata mismatch')
        data=raw.get_data()*1e6 # physical microvolts
        onset=np.round(raw.annotations.onset*250).astype(int)
        desc=raw.annotations.description
        boundaries=np.unique(np.r_[0,onset[desc=='32766'],data.shape[1]])
        filt=np.full(data.shape,np.nan)
        sos=butter(4,c['bci_filter_hz'],btype='bandpass',fs=250,output='sos')
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            block=data[:,start:end]
            good=np.isfinite(block).all(0)
            edges=np.flatnonzero(np.diff(np.r_[False,good,False]))
            for a,b in zip(edges[::2],edges[1::2]):
                if b-a>50:
                    filt[:,start+a:start+b]=sosfiltfilt(sos,block[:,a:b])
        counts={}
        for cue,label in zip(onset,desc):
            if label not in ['769','770','771','772']:
                continue
            start=cue+round(c['bci_epoch_after_cue_s'][0]*250)
            end=cue+round(c['bci_epoch_after_cue_s'][1]*250)
            a,b=data[:,start:end],filt[:,start:end]
            if a.shape[1]!=500 or not np.isfinite(a).all() or not np.isfinite(b).all():
                exclusions.append(dict(subject=sub,cue_sample=int(cue),reason='short or nonfinite epoch'))
                continue
            a=resample_poly(a,128,125,axis=-1);b=resample_poly(b,128,125,axis=-1)
            a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
            if np.any(a.std(1)<1e-10):
                exclusions.append(dict(subject=sub,cue_sample=int(cue),reason='zero variance channel'))
                continue
            xs.append(a[:22].astype(np.float32));fs.append(b[:22].astype(np.float32));es.append(b[22:].astype(np.float32))
            ys.append(int(label)-769);subjects.append(sub);samples.append(cue)
            bad=bool(np.any((desc=='1023')&(onset>=cue-500)&(onset<=end)))
            artifacts.append(bad)
            counts[label]=counts.get(label,0)+1
        man.append(dict(subject=sub,path=str(path),sha256=sha256(path),sfreq=250,channels=raw.ch_names,
                        cue_counts=counts,n_artifact_annotations=int(np.sum(desc=='1023'))))
        print(f'Loaded BCI participant {sub}: {counts}',flush=True)
    result=dict(raw=np.asarray(xs),minimal=np.asarray(fs),eog=np.asarray(es),
                label=np.asarray(ys),subject=np.asarray(subjects),artifact=np.asarray(artifacts),cue_sample=np.asarray(samples))
    np.savez_compressed(cache,**result)
    write_json(manifest_path,dict(recordings=man,exclusions=exclusions,config=c,
                                  warning='T sessions only. Labels from class cues. E sessions not used.'))
    return result

def femto_predict(x):
    # An immutable source-code snapshot is installed by the preparation command.
    sys.path.insert(0,str(ROOT/'src'/'legacy'))
    from models import conv6SingleHeadNet, deeperHead, deeperEncoder, DWConv1d
    path=ROOT/'models'/'femto_2561_state_dict.pt'
    model=conv6SingleHeadNet(ch=8,groups=8,scale_up=1,device='cpu')
    # The inherited checkpoint stores a complete module. Explicitly allow only
    # the inspected architecture classes, retaining the restricted unpickler.
    allowed=[conv6SingleHeadNet,deeperHead,deeperEncoder,DWConv1d,
             torch.nn.Conv1d,torch.nn.GroupNorm,torch.nn.Sequential,torch.nn.SiLU]
    with torch.serialization.safe_globals(allowed):
        state=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(state,torch.nn.Module):
        state=state.state_dict()
    if isinstance(state,dict) and 'state_dict' in state:
        state=state['state_dict']
    model.load_state_dict(state)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=model.to(device);model.eval()
    flat=x.reshape(-1,512)
    scale=flat.std(1,keepdims=True)
    normalized=(flat/scale).astype(np.float32)
    preds=[]
    with torch.no_grad():
        for i in range(0,len(flat),128):
            preds.append(model(torch.from_numpy(normalized[i:i+128,None,:]).to(device))[0][:,0].cpu().numpy())
    return (np.concatenate(preds)*scale).reshape(x.shape),dict(path=str(path),sha256=sha256(path),
        parameters=sum(p.numel() for p in model.parameters()),device=str(device),input='minimal filtered EEG, per-window std scale restored')

def reference_regression(xtr,etr,xte,ete):
    x=xtr.transpose(0,2,1).reshape(-1,22).astype(float)
    e=etr.transpose(0,2,1).reshape(-1,3).astype(float)
    mx,me=x.mean(0),e.mean(0);ec=e-me
    gram=ec.T@ec
    beta=np.linalg.solve(gram+1e-5*np.trace(gram)/3*np.eye(3),ec.T@(x-mx))
    def apply(a,b):
        return a-np.einsum('nkt,kc->nct',b-me[:,None],beta)
    return apply(xtr,etr),apply(xte,ete),beta

def ica_clean(xtr,etr,xte):
    c=config()
    x=xtr.transpose(0,2,1).reshape(-1,22).astype(float)
    e=etr.transpose(0,2,1).reshape(-1,3).astype(float)
    ids=np.linspace(0,len(x)-1,min(c['ica_fit_samples'],len(x)),dtype=int)
    scale=x[ids].std(0)
    ica=FastICA(n_components=22,whiten='unit-variance',random_state=c['seed'],max_iter=300,tol=.001)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        z=ica.fit_transform(x[ids]/scale)
    corr=np.corrcoef(z.T,e[ids].T)[:22,22:]
    scores=np.max(np.abs(corr),1)
    excluded=np.flatnonzero(scores>c['ica_correlation_threshold'])
    excluded=excluded[np.argsort(scores[excluded])[::-1]][:c['ica_max_removed']]
    def apply(a):
        flat=a.transpose(0,2,1).reshape(-1,22).astype(float)
        z=ica.transform(flat/scale);z[:,excluded]=0
        return (ica.inverse_transform(z)*scale).reshape(len(a),512,22).transpose(0,2,1).astype(np.float32)
    info=dict(excluded=excluded.tolist(),training_eog_correlation=scores.tolist(),iterations=int(ica.n_iter_),
              warnings=[str(w.message) for w in caught],fit_timepoints=len(ids))
    return apply(xtr),apply(xte),info

def fit_csp(x,y):
    cov=np.einsum('nct,ndt->ncd',x.astype(float),x.astype(float))/x.shape[-1]
    cov/=np.trace(cov,axis1=1,axis2=2)[:,None,None]
    filters=[]
    for label in range(4):
        a=cov[y==label].mean(0);b=cov[y!=label].mean(0)
        reg=1e-3*np.trace(a+b)/len(a)
        _,vectors=eigh(a+reg*np.eye(len(a)),a+b+2*reg*np.eye(len(a)))
        filters.append(vectors[:,np.r_[0,1,len(a)-2,len(a)-1]].T)
    return np.concatenate(filters)

def features(x,w):
    projected=np.einsum('kc,nct->nkt',w,x)
    power=np.var(projected,axis=-1)
    return np.log(np.maximum(power/power.sum(1,keepdims=True),1e-12))

def decode(xtr,ytr,xte):
    w=fit_csp(xtr,ytr)
    clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    clf.fit(features(xtr,w),ytr)
    fte=features(xte,w)
    return clf.predict(fte),clf.predict_proba(fte),w,clf

def eog_features(x):
    f=np.fft.rfftfreq(512,1/256);p=abs(np.fft.rfft(x))**2
    feats=[np.log(np.maximum(p[:,:,(f>=lo)&(f<hi)].mean(-1),1e-12)) for lo,hi in [(1,4),(4,8),(8,13),(13,30),(30,40)]]
    return np.concatenate(feats+[np.log(np.maximum(np.ptp(x,axis=-1),1e-12))],axis=1)

def subject_ci(values):
    rng=np.random.default_rng(config()['seed'])
    boot=np.mean(rng.choice(values,(10000,len(values)),replace=True),1)
    return np.quantile(boot,[.025,.975]).tolist()

def sign_flip(values):
    values=np.asarray(values);observed=abs(values.mean())
    null=[abs(np.mean(values*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=len(values))]
    return float(np.mean(np.asarray(null)>=observed-1e-12))

def main():
    data=load_data();dst=ROOT/'results'/'real_task';dst.mkdir(parents=True,exist_ok=True)
    fm,legacy=femto_predict(data['minimal']);write_json(dst/'legacy_checkpoint.json',legacy)
    rows=[];predictions=[];folds=[]
    for sub in config()['bci_subjects']:
        fold_path=dst/f'fold_{sub}.json';pred_path=dst/f'predictions_{sub}.npz'
        if fold_path.exists() and pred_path.exists():
            prior=__import__('json').loads(fold_path.read_text());rows+=prior['metrics'];folds.append(prior['fold'])
            continue
        tr=data['subject']!=sub;te=~tr;ytr=data['label'][tr];yte=data['label'][te]
        xtr,xte=data['minimal'][tr],data['minimal'][te];etr,ete=data['eog'][tr],data['eog'][te]
        rtr,rte,beta=reference_regression(xtr,etr,xte,ete)
        itr,ite,info=ica_clean(xtr,etr,xte)
        quality=lambda x,e:np.log(np.maximum(np.ptp(x,axis=-1).max(1),1e-9))+np.log(np.maximum(np.ptp(e,axis=-1).max(1),1e-9))
        threshold=float(np.quantile(quality(xtr,etr),config()['severe_quality_train_quantile']))
        keeptr=quality(xtr,etr)<=threshold;keepte=quality(xte,ete)<=threshold
        candidates=dict(raw=(data['raw'][tr],data['raw'][te]),minimal=(xtr,xte),
                        regression=(rtr,rte),ica=(itr,ite),femto_frozen=(fm[tr],fm[te]))
        predictions=dict(label=yte,artifact=data['artifact'][te],cue_sample=data['cue_sample'][te],retained=keepte)
        metrics=[]
        for method,(train,test) in candidates.items():
            pred,prob,w,clf=decode(train,ytr,test)
            predictions[method]=pred;predictions[method+'_prob']=prob
            np.savez_compressed(dst/f'decoder_subject{sub}_{method}.npz',csp=w,coef=clf.coef_,intercept=clf.intercept_)
            for stratum,mask in [('all',np.ones(len(yte),bool)),('retained',keepte),('artifact_marked',data['artifact'][te]),('unmarked',~data['artifact'][te])]:
                if not mask.sum():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ba=balanced_accuracy_score(yte[mask],pred[mask])
                metrics.append(dict(subject=sub,method=method,stratum=stratum,n=int(mask.sum()),
                                    accuracy=float(np.mean(pred[mask]==yte[mask])),balanced_accuracy=float(ba),
                                    coverage=float(mask.mean())))
        pred,prob,w,clf=decode(xtr[keeptr],ytr[keeptr],xte)
        predictions['minimal_selective']=pred
        metrics.append(dict(subject=sub,method='minimal_selective',stratum='retained',n=int(keepte.sum()),
                            accuracy=float(np.mean(pred[keepte]==yte[keepte])),
                            balanced_accuracy=float(balanced_accuracy_score(yte[keepte],pred[keepte])),coverage=float(keepte.mean())))
        clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(eog_features(etr),ytr)
        pred=clf.predict(eog_features(ete));predictions['eog_only']=pred
        metrics.append(dict(subject=sub,method='eog_only',stratum='all',n=len(yte),accuracy=float(np.mean(pred==yte)),
                            balanced_accuracy=float(balanced_accuracy_score(yte,pred)),coverage=1.))
        fold=dict(subject=sub,n_train=int(tr.sum()),n_test=int(te.sum()),training_subjects=np.unique(data['subject'][tr]).tolist(),
                  ica=info,quality_threshold=threshold,n_training_retained=int(keeptr.sum()),
                  class_test_coverage={str(k):float(keepte[yte==k].mean()) for k in range(4)},
                  reference_beta=beta.tolist())
        np.savez_compressed(pred_path,**predictions)
        write_json(fold_path,dict(fold=fold,metrics=metrics))
        rows+=metrics;folds.append(fold)
        print(f'BCI subject {sub}: {[(r["method"],round(r["accuracy"],3)) for r in metrics if r["stratum"]=="all"]}',flush=True)
    with open(dst/'subject_metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[];contrasts=[]
    for method in ['raw','minimal','regression','ica','femto_frozen','eog_only','minimal_selective']:
        stratum='retained' if method=='minimal_selective' else 'all'
        selected=[r for r in rows if r['method']==method and r['stratum']==stratum]
        values=np.asarray([r['accuracy'] for r in selected])
        summary.append(dict(method=method,stratum=stratum,subjects=len(values),mean_accuracy=float(values.mean()),
                            ci95_subject_bootstrap=subject_ci(values),mean_coverage=float(np.mean([r['coverage'] for r in selected]))))
        if method not in ['minimal','minimal_selective','eog_only']:
            base={r['subject']:r['accuracy'] for r in rows if r['method']=='minimal' and r['stratum']=='all'}
            diff=np.asarray([base[r['subject']]-r['accuracy'] for r in selected])
            contrasts.append(dict(contrast='minimal minus '+method,mean_accuracy_difference=float(diff.mean()),
                                  ci95_subject_bootstrap=subject_ci(diff),exact_sign_flip_p=sign_flip(diff),
                                  practical_margin=.02,noninferiority_exploratory=bool(subject_ci(diff)[0]>-.02)))
    write_json(dst/'summary.json',dict(summary=summary,contrasts=contrasts,folds=folds,
        uncertainty='Exploratory participant bootstrap and sign-flip; overlapping training folds and prior cohort use limit confirmatory inference',
        config=config(),code_sha256=sha256(Path(__file__)),data_sha256=sha256(ROOT/'data'/'bci_epochs.npz')))

if __name__=='__main__':
    main()
