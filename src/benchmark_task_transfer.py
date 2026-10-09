"""Post hoc, matched transfer of synthetic-trained estimators to real task data."""
import csv
import json
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common import ROOT,config,write_json,sha256
from mixture_study import TinyCNN,LargeMLP,DEVICE,predict,predict_linear
from real_task import load_data,decode,eog_features,subject_ci

def main():
    data=load_data();dst=ROOT/'results'/'benchmark_task_transfer';dst.mkdir(parents=True,exist_ok=True)
    x=data['minimal'];flat=x.reshape(-1,512);scale=flat.std(1,keepdims=True)
    normal=(flat/scale).astype(np.float32)
    base=ROOT/'results'/'mixtures'/'EOG'
    fit=dict(np.load(base/'linear_parameters.npz'))
    linears=predict_linear(fit,normal)
    selected=json.loads((base/'selection.json').read_text())
    cases=[('minimal',None,None)]+[(k,None,v) for k,v in linears.items() if k in ['fourier','affine']]
    for info in selected:
        name=f'{info["kind"]}_seed{info["seed"]}'
        path=base/f'{name}_lr{info["lr"]}.pt'
        if sha256(path)!=info['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        model=(TinyCNN() if info['kind']=='tiny_cnn' else LargeMLP()).to(DEVICE)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        cases.append((name,info,predict(model,normal)))
    rows=[];metadata=[]
    for method,info,pred in cases:
        transformed=x if pred is None else (pred*scale).reshape(x.shape).astype(np.float32)
        metadata.append(dict(method=method,synthetic_selection=info,
            relative_waveform_alteration=float(np.mean((transformed-x)**2)/np.mean(x*x))))
        power_features=eog_features(transformed)
        saved={}
        for sub in config()['bci_subjects']:
            tr=data['subject']!=sub;te=~tr;ytr=data['label'][tr];yte=data['label'][te]
            csp_pred,csp_prob,*_=decode(transformed[tr],ytr,transformed[te])
            clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(power_features[tr],ytr)
            lp_pred=clf.predict(power_features[te])
            saved[f'csp_subject{sub}']=csp_pred;saved[f'bandpower_subject{sub}']=lp_pred
            for decoder,p in [('CSP',csp_pred),('bandpower',lp_pred)]:
                rows.append(dict(subject=sub,method=method,decoder=decoder,n=int(te.sum()),
                                 accuracy=float(np.mean(p==yte))))
        np.savez_compressed(dst/f'{method}_predictions.npz',**saved)
        print(f'Task transfer {method} done',flush=True)
    with open(dst/'subject_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for decoder in ['CSP','bandpower']:
        for method,_,_ in cases:
            values=np.asarray([r['accuracy'] for r in rows if r['decoder']==decoder and r['method']==method])
            summary.append(dict(method=method,decoder=decoder,mean_accuracy=float(values.mean()),ci95=subject_ci(values)))
        for family in ['tiny_cnn','large_mlp']:
            bysub=np.asarray([[next(r['accuracy'] for r in rows if r['subject']==sub and r['decoder']==decoder and r['method']==f'{family}_seed{seed}')
                               for seed in config()['training_seeds']] for sub in config()['bci_subjects']]).mean(1)
            summary.append(dict(method=family+'_mean_seeds',decoder=decoder,mean_accuracy=float(bysub.mean()),ci95=subject_ci(bysub)))
    write_json(dst/'summary.json',dict(status='post hoc exploratory extension',summary=summary,models=metadata,
        caveat='BCI source overlap with released EOG training banks cannot be excluded; task prediction is not neural truth',
        code_sha256=sha256(__file__),data_sha256=sha256(ROOT/'data'/'bci_epochs.npz')))

if __name__=='__main__':
    main()
