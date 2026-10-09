"""Exploratory teacher-target changes on identical real BCI observations."""
import csv
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common import ROOT,sha256,write_json
from real_task import load_data,decode,eog_features,subject_ci
from target_validity import settings

def teacher_projection(x,variant):
    freq=np.fft.rfftfreq(x.shape[-1],1/256)
    keep=np.ones(len(freq),bool)
    if variant=='remove_8_30':
        keep[(freq>=8)&(freq<30)]=False
    elif variant=='retain_1_8':
        keep=(freq>=1)&(freq<8)
    elif variant=='remove_1_4':
        keep[(freq>=1)&(freq<4)]=False
    elif variant!='parent':
        raise ValueError(variant)
    return np.fft.irfft(np.fft.rfft(x,axis=-1)*keep,n=x.shape[-1],axis=-1).astype(np.float32)

def main():
    data=load_data();x=data['minimal'];dst=ROOT/'results'/'teacher_intervention';dst.mkdir(parents=True,exist_ok=True)
    rows=[];conditions=[]
    for variant in settings()['real_teacher_variants']:
        transformed=teacher_projection(x,variant)
        energy=float(np.mean((transformed-x)**2))
        bp=eog_features(transformed)
        saved={};methodrows=[]
        for sub in range(1,10):
            tr=data['subject']!=sub;te=~tr;ytr=data['label'][tr];yte=data['label'][te]
            pred,*_=decode(transformed[tr],ytr,transformed[te])
            power=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(bp[tr],ytr).predict(bp[te])
            saved[f'CSP_subject{sub}']=pred;saved[f'bandpower_subject{sub}']=power
            for decoder,p in [('CSP',pred),('bandpower',power)]:
                row=dict(subject=sub,teacher=variant,decoder=decoder,n=int(te.sum()),accuracy=float(np.mean(p==yte)))
                rows.append(row);methodrows.append(row)
        np.savez_compressed(dst/f'{variant}_predictions.npz',**saved)
        conditions.append(dict(teacher=variant,parent_against_teacher_mse_uv2=energy,
            teacher_against_itself_mse_uv2=0.,deleted_scalp_energy_fraction=energy/float(np.mean(x*x))))
        print(f'Teacher intervention {variant} completed',flush=True)
    with (dst/'subject_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for condition in conditions:
        for decoder in settings()['real_teacher_decoders']:
            values=np.asarray([r['accuracy'] for r in rows if r['teacher']==condition['teacher'] and r['decoder']==decoder])
            base=np.asarray([r['accuracy'] for r in rows if r['teacher']=='parent' and r['decoder']==decoder])
            summary.append(dict(**condition,decoder=decoder,mean_accuracy=float(values.mean()),
                ci95=subject_ci(values),difference_from_parent=float((values-base).mean()),paired_ci95=subject_ci(values-base)))
    write_json(dst/'summary.json',dict(config=settings(),summary=summary,
        data_sha256=sha256(ROOT/'data'/'bci_epochs.npz'),code_sha256=sha256(__file__),
        interpretation='Fixed epoch-FFT teacher interventions, not recommended acquisition filters. Deleted scalp energy is not identified neural energy. Reused cohort; exploratory participant intervals.'))

if __name__=='__main__':
    main()
