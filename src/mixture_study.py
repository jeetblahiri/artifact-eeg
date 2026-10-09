"""Execute the declared EOG/EMG/ECG source-disjoint reconstruction experiment."""
import argparse
import csv
import copy
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
import wfdb
from common import ROOT, config, sha256, write_json, unique_bank, split_ids, make_mixtures, per_row_metrics, multiway_ci

torch.set_num_threads(4)
DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv1d(1,8,9,padding=4), nn.GELU()]
        for _ in range(3):
            layers += [nn.Conv1d(8,8,9,padding=4), nn.GELU()]
        layers += [nn.Conv1d(8,1,9,padding=4)]
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return x + self.net(x[:,None,:])[:,0,:]

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for _ in range(3):
            layers += [nn.Linear(512,512), nn.GELU()]
        layers += [nn.Linear(512,512)]
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return x+self.net(x)

def prepare():
    c=config()
    dst=ROOT/'data'/'mixtures'
    dst.mkdir(parents=True,exist_ok=True)
    manifest={'fs':c['fs'],'length':c['length'],'sources':{},'pools':{}}
    banks={}
    for typ in ['EEG','EOG','EMG']:
        path=(ROOT/c['source_dir']/f'{typ}_all_epochs.npy').resolve()
        a=np.load(path)
        bank,keep,ids=unique_bank(a)
        banks[typ]=bank
        manifest['sources'][typ]=dict(path=str(path),sha256=sha256(path),shape=list(a.shape),
                                     unique_normalized_rows=len(keep),kept_original_rows=keep.tolist(),
                                     fs_evidence='upstream MAT fs=256 in inherited revision audit')
    # Records are the split unit; two-second ECG segments are not scalp artifacts.
    ecg,record_ids,ecg_sources=[],[],[]
    for record in ['100','101','103','105','106','107','108','109','111','112','113','114','115','116','117']:
        base=Path(c['ecg_dir'])/record
        sig,fields=wfdb.rdsamp(str(base),sampfrom=0,sampto=360*120,channels=[0])
        if fields['fs']!=360:
            raise ValueError('Unexpected ECG rate')
        signal=resample_poly(sig[:,0],32,45)
        chunks=signal[:(len(signal)//512)*512].reshape(-1,512)
        for row in chunks:
            ecg.append(row)
            record_ids.append(record)
        ecg_sources.append(dict(record=record,fs=fields['fs'],lead=fields['sig_name'][0],
                               header_sha256=sha256(base.with_suffix('.hea')),
                               signal_sha256=sha256(base.with_suffix('.dat'))))
    ecg=np.asarray(ecg)
    bank,keep,ids=unique_bank(ecg)
    banks['ECG']=bank
    records=np.asarray(record_ids)[keep]
    manifest['sources']['ECG']=dict(path=c['ecg_dir'],records=ecg_sources,shape=list(ecg.shape),
                                   unique_normalized_rows=len(keep),record_for_row=records.tolist(),
                                   interpretation='recorded ambulatory ECG lead proxy (MLII except record 114 V5); not measured scalp contamination')
    rng=np.random.default_rng(c['seed'])
    splits={k:split_ids(len(v),rng,c['split_fractions']) for k,v in banks.items() if k!='ECG'}
    record_split=split_ids(15,rng,c['split_fractions'])
    record_order=np.unique(records)
    splits['ECG']={part:np.flatnonzero(np.isin(records,record_order[rr])) for part,rr in record_split.items()}
    for typ in banks:
        manifest['pools'][typ]={p:idx.tolist() for p,idx in splits[typ].items()}
        for p1,p2 in [('train','val'),('train','test'),('val','test')]:
            assert not set(splits[typ][p1])&set(splits[typ][p2])
    manifest['ecg_split_records']={p:record_order[rr].tolist() for p,rr in record_split.items()}
    for typ in ['EOG','EMG','ECG']:
        for part,n in c['mixture_counts'].items():
            mix=make_mixtures(banks['EEG'],banks[typ],splits['EEG'][part],splits[typ][part],n,c['snr_power_db'],rng)
            np.savez_compressed(dst/f'{typ}_{part}.npz',**mix)
    write_json(dst/'manifest.json',manifest)
    return manifest

def fit_linear(train,val):
    x,s=train['x'].astype(float),train['s'].astype(float)
    xv,sv=val['x'].astype(float),val['s'].astype(float)
    mx,ms=x.mean(0),s.mean(0)
    xc,sc=x-mx,s-ms
    g=float(np.sum(x*s)/np.sum(x*x))
    fx,fs=np.fft.rfft(xc),np.fft.rfft(sc)
    h=np.sum(fs*fx.conj(),0)/np.maximum(np.sum(abs(fx)**2,0),1e-12)
    gram=xc.T@xc/len(x)
    cross=xc.T@sc/len(x)
    choices=[]
    for ridge in config()['linear_ridge_grid']:
        w=np.linalg.solve(gram+ridge*np.trace(gram)/512*np.eye(512),cross)
        mse=float(np.mean((ms+(xv-mx)@w-sv)**2))
        choices.append((mse,ridge,w))
    best=min(choices,key=lambda item:item[0])
    return dict(scalar_gain=g,mean_x=mx,mean_s=ms,spectral_h=h,affine_w=best[2],ridge=best[1],
                validation_grid=[dict(mse=m,ridge=r) for m,r,_ in choices])

def predict_linear(fit,x):
    x=np.asarray(x,float)
    return dict(identity=x,scalar=x*fit['scalar_gain'],
                fourier=fit['mean_s']+np.fft.irfft(np.fft.rfft(x-fit['mean_x'])*fit['spectral_h'],n=512),
                affine=fit['mean_s']+(x-fit['mean_x'])@fit['affine_w'])

def predict(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.asarray(x[i:i+128],dtype=np.float32)).to(DEVICE)).cpu().numpy()
                               for i in range(0,len(x),128)])

def train_model(kind,seed,lr,train,val,dst):
    tag=f'{kind}_seed{seed}_lr{lr}'
    checkpoint=dst/f'{tag}.pt'
    history_file=dst/f'{tag}_history.json'
    torch.manual_seed(seed)
    model=(TinyCNN() if kind=='tiny_cnn' else LargeMLP()).to(DEVICE)
    if checkpoint.exists() and history_file.exists():
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        return model, __import__('json').loads(history_file.read_text())
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    x=torch.from_numpy(train['x']); s=torch.from_numpy(train['s'])
    gen=torch.Generator().manual_seed(seed)
    best=float('inf'); state=None; history=[]; start=time.perf_counter()
    for epoch in range(config()['epochs']):
        model.train(); total=0
        order=torch.randperm(len(x),generator=gen)
        for ids in order.split(config()['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss=torch.mean((model(x[ids].to(DEVICE))-s[ids].to(DEVICE))**2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            total+=float(loss.detach())*len(ids)
        validation=float(np.mean((predict(model,val['x'])-val['s'])**2))
        history.append(dict(epoch=epoch+1,train_mse=total/len(x),validation_mse=validation))
        if validation<best:
            best=validation; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
        if (epoch+1)%5==0:
            print(f'{dst.name} {tag} epoch {epoch+1} val {validation:.5f}',flush=True)
    model.load_state_dict(state)
    torch.save(state,checkpoint)
    info=dict(kind=kind,seed=seed,lr=lr,parameters=sum(p.numel() for p in model.parameters()),
              best_validation_mse=best,best_epoch=best_epoch,seconds=time.perf_counter()-start,history=history,
              checkpoint_sha256=sha256(checkpoint),device=str(DEVICE))
    write_json(history_file,info)
    return model,info

def run(artifact_types):
    c=config(); data=ROOT/'data'/'mixtures'
    if not (data/'manifest.json').exists():
        prepare()
    all_results=[]
    for typ in artifact_types:
        start=time.perf_counter(); dst=ROOT/'results'/'mixtures'/typ;dst.mkdir(parents=True,exist_ok=True)
        train,val,test=[dict(np.load(data/f'{typ}_{p}.npz')) for p in ['train','val','test']]
        fit=fit_linear(train,val)
        np.savez_compressed(dst/'linear_parameters.npz',**{k:v for k,v in fit.items() if k!='validation_grid'})
        write_json(dst/'linear_selection.json',fit['validation_grid'])
        preds=predict_linear(fit,test['x']); selections=[]
        for kind in ['tiny_cnn','large_mlp']:
            for seed in c['training_seeds']:
                candidates=[train_model(kind,seed,lr,train,val,dst) for lr in c['learning_rates']]
                model,info=min(candidates,key=lambda item:item[1]['best_validation_mse'])
                tag=f'{kind}_seed{seed}';preds[tag]=predict(model,test['x'])
                selections.append(info)
        np.savez_compressed(dst/'predictions.npz',**preds)
        write_json(dst/'selection.json',selections)
        rows=[];rowmetrics={}
        for method,pred in preds.items():
            metr=per_row_metrics(pred,test['s'])
            rowmetrics[method]=metr
            rows.append(dict(artifact=typ,method=method,**{k:float(np.nanmean(v)) for k,v in metr.items()}))
        # Average the losses over seeds, not the outputs (no ensembling).
        for kind in ['tiny_cnn','large_mlp']:
            rowmetrics[kind+'_mean_seeds']={k:np.mean([rowmetrics[f'{kind}_seed{seed}'][k] for seed in c['training_seeds']],0)
                                           for k in rowmetrics['identity']}
            rows.append(dict(artifact=typ,method=kind+'_mean_seeds',**{k:float(np.nanmean(v)) for k,v in rowmetrics[kind+'_mean_seeds'].items()}))
        contrasts=[]
        artifact_clusters=test['artifact_id']
        if typ=='ECG':
            source_manifest=__import__('json').loads((data/'manifest.json').read_text())
            artifact_clusters=np.asarray(source_manifest['sources']['ECG']['record_for_row'])[test['artifact_id']]
        for method in ['fourier','affine','tiny_cnn_mean_seeds','large_mlp_mean_seeds']:
            diff=rowmetrics[method]['mse']-rowmetrics['affine']['mse']
            ci=multiway_ci(diff,test['eeg_id'],artifact_clusters,np.random.default_rng(c['seed']),c['bootstrap_replicates'])
            contrasts.append(dict(method=method,reference='affine',mse_difference=float(diff.mean()),ci95=ci,
                                  uncertainty_unit='released EEG IDs and '+('ECG recording IDs' if typ=='ECG' else 'artifact IDs')+'; conditional on fitted models'))
        # Teacher target sensitivity: retain fixed predictions, only change the nominal label.
        label_rows=[]
        residual=test['x']-test['s']
        for eps in [0.,.1,.25,.5,1.]:
            target=test['s']+eps*residual
            for method,pred in preds.items():
                label_rows.append(dict(artifact=typ,epsilon=eps,method=method,mse=float(np.mean((pred-target)**2))))
        for name,items in [('summary',rows),('label_sensitivity',label_rows)]:
            with open(dst/f'{name}.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
        # SNR and frequency summaries have no interpretation as neural truth.
        strata=[]
        for db in c['snr_power_db']:
            mask=test['snr_db']==db
            for method,metr in rowmetrics.items():
                strata.append(dict(artifact=typ,snr_power_db=db,method=method,n=int(mask.sum()),
                                   **{k:float(np.nanmean(v[mask])) for k,v in metr.items()}))
        with open(dst/'snr_summary.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(strata[0]));writer.writeheader();writer.writerows(strata)
        write_json(dst/'contrasts.json',contrasts)
        write_json(dst/'run.json',dict(config=c,seconds=time.perf_counter()-start,
                                      data_sha256={p:sha256(data/f'{typ}_{p}.npz') for p in ['train','val','test']},
                                      code_sha256=sha256(Path(__file__)),device=str(DEVICE),selections=selections))
        all_results+=rows
        print(f'{typ} complete: {[(r["method"],round(r["mse"],5)) for r in rows]}',flush=True)
    return all_results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--types',nargs='+',default=['EOG','EMG','ECG'])
    args=parser.parse_args()
    run(args.types)
