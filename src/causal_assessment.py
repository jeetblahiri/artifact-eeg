"""Assessment of benchmark reproductions, recipe stages, and target intervention."""
import json
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from erp_validation import baseline,stem,weights,feature,fit_decoder,TIMES
from tim_revision import C,ci
from benchmark_models import load,apply
from smooth_direction_probes import spatial_direction
from measurement_math import conformal_quantile

OUT=ROOT/'results/causal_revision';CACHE=ROOT/'data/causal_revision';OLD=ROOT/'data/erp_validation'
METHODS=['recipe_filter','recipe_brain_argmax','recipe_artifact80','ica_brain_argmax','ica_artifact80',
         'simple_cnn','complex_cnn','target_parent','target_recipe_brain_argmax','target_recipe_artifact80']

def assess():
    rows=[];summary=[];individual_seed=[]
    config=json.loads((ROOT/'causal_revision_config.json').read_text())
    for task in ['N170','P3']:
        store=np.load(ROOT/f'results/revision/{task}_features.npz');o=store['offsets'];sl={s:slice(o[s-1],o[s]) for s in range(1,41)}
        ytr=np.concatenate([store['label'][sl[s]] for s in C['development']]);ptr=np.concatenate([store['parent'][sl[s]] for s in C['development']])
        fixed=fit_decoder(ptr,ytr);names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in C['tasks'][task]['roi']]
        waves=[];features={}
        for m in METHODS:
            features[m]=[feature(np.load(CACHE/f'{stem(task,s)}_{m}.npy')) for s in range(1,41)]
            adap=fit_decoder(np.concatenate([features[m][s-1] for s in C['development']]),ytr)
            np.savez(OUT/f'{task}_{m}_decoder.npz',scale_mean=adap[0].mean_,scale=adap[0].scale_,coef=adap[1].coef_,intercept=adap[1].intercept_,classes=adap[1].classes_)
            for s in range(1,41):
                x=np.load(CACHE/f'{stem(task,s)}_{m}.npy').astype(float);y=store['label'][sl[s]];parent=np.load(OLD/f'{stem(task,s)}_parent.npy').astype(float)
                wave=(x[y==1].mean(0)-x[y==0].mean(0))[roi].mean(0);waves.append(wave)
                pf=fixed[1].predict(fixed[0].transform(features[m][s-1]));pa=adap[1].predict(adap[0].transform(features[m][s-1]));pp=fixed[1].predict(fixed[0].transform(store['parent'][sl[s]]))
                pam=float(((parent[y==1].mean(0)-parent[y==0].mean(0))[roi].mean(0))@weights(task))
                rows.append(dict(task=task,subject=s,method=m,split=next(k for k in ['development','calibration','evaluation'] if s in C[k]),n=len(y),
                    amplitude_uv=float(wave@weights(task)),parent_amplitude_uv=pam,frozen_ba=float(balanced_accuracy_score(y,pf)),
                    adapted_ba=float(balanced_accuracy_score(y,pa)),parent_ba=float(balanced_accuracy_score(y,pp))))
            rr=[r for r in rows if r['task']==task and r['method']==m and r['split']=='evaluation']
            ad=np.array([r['amplitude_uv']-r['parent_amplitude_uv'] for r in rr]);bd=np.array([r['frozen_ba']-r['parent_ba'] for r in rr])
            summary.append(dict(task=task,method=m,mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),
                amplitude_difference_uv=float(ad.mean()),amplitude_difference_ci95=ci(ad),mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr])),
                frozen_ba_difference=float(bd.mean()),frozen_ba_difference_ci95=ci(bd),mean_adapted_ba=float(np.mean([r['adapted_ba'] for r in rr]))))
        np.save(OUT/f'{task}_roi_contrasts.npy',np.asarray(waves).reshape(len(METHODS),40,512))
        np.savez(OUT/f'{task}_features.npz',label=store['label'],offsets=o,**{m:np.concatenate(f) for m,f in features.items()})
        for m in ['simple_cnn','complex_cnn']:
            for seed in config['models']['seeds']:
                for s in C['evaluation']:
                    p=CACHE/f'{stem(task,s)}_{m}_{seed}.npy'
                    if not p.exists():continue
                    x=np.load(p);y=store['label'][sl[s]];wave=(x[y==1].astype(float).mean(0)-x[y==0].astype(float).mean(0))[roi].mean(0)
                    pred=fixed[1].predict(fixed[0].transform(feature(x)));individual_seed.append(dict(task=task,method=m,seed=seed,subject=s,
                        amplitude_uv=float(wave@weights(task)),frozen_ba=float(balanced_accuracy_score(y,pred))))
    # Direct paired target contrasts cancel all shared learner/input choices.
    target_pairs=[];lookup={(r['task'],r['subject'],r['method']):r for r in rows}
    for task in ['N170','P3']:
        for m in ['target_recipe_brain_argmax','target_recipe_artifact80']:
            d=[lookup[task,s,m]['amplitude_uv']-lookup[task,s,'target_parent']['amplitude_uv'] for s in C['evaluation']]
            target_pairs.append(dict(task=task,method=m,comparator='target_parent',amplitude_difference_uv=float(np.mean(d)),ci95=ci(d)))
    calibration={}
    for m in METHODS:
        vs=[];bs=[]
        for s in C['calibration']+C['evaluation']:
            vs.append(max(abs(lookup[t,s,m]['amplitude_uv']-lookup[t,s,m]['parent_amplitude_uv']) for t in ['N170','P3']))
            bs.append(max(abs(lookup[t,s,m]['frozen_ba']-lookup[t,s,m]['parent_ba']) for t in ['N170','P3']))
        qv=conformal_quantile(vs[:20],.1);qb=conformal_quantile(bs[:20],.1);qs=conformal_quantile(np.maximum(vs[:20],np.array(bs[:20])/.02),.1)
        calibration[m]=dict(voltage90_uv=qv,ba90_points=100*qb,joint90=qs,joint_test_covered=int(np.sum(np.maximum(vs[20:],np.array(bs[20:])/.02)<=qs)))
    write_json(OUT/'erp_assessment.json',dict(rows=rows,summary=summary,target_pairs=target_pairs,calibration=calibration,
        individual_seeds=individual_seed,methods=METHODS,config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__)),
        scope='Recorded parent-relative changes; no true cortical source. Paired-target fits isolate target choice in a fixed learner, not the origin of a historical CNN deployment error.'))
    return rows

def probes():
    config=json.loads((ROOT/'causal_revision_config.json').read_text());rows=[];summary=[]
    for task in ['N170','P3']:
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];tc=C['tasks'][task];roi=[names.index(n) for n in tc['roi']]
        s=spatial_direction(names,roi)[:,None]*np.exp(-.5*((TIMES-tc['probe_center_s'])/tc['probe_sd_s'])**2);s=baseline(s);den=float(s[roi].mean(0)@weights(task))
        for m in ['simple_cnn','complex_cnn']:
            models=[load(m,z) for z in config['models']['seeds']]
            for person in C['evaluation']:
                ids=np.load(OLD/f'{stem(task,person)}_identity.npz');selected=np.r_[np.flatnonzero(ids['label']==0)[:4],np.flatnonzero(ids['label']==1)[:4]]
                x=np.load(OLD/f'{stem(task,person)}_parent.npy')[selected].astype(float);zero=apply(x,models)
                for a in C['probe_amplitudes_uv']:
                    v=(apply(x+a*s,models)-zero)[:,roi].mean(1)@weights(task)/(a*den)
                    for idx,g in zip(selected,v):rows.append(dict(task=task,subject=person,method=m,amplitude_uv=a,background_trial=int(idx),gain=float(g)))
            a=np.array([r['gain'] for r in rows if r['task']==task and r['method']==m]);summary.append(dict(task=task,method=m,n=len(a),mean_gain=float(a.mean()),q05_gain=float(np.quantile(a,.05)),q95_gain=float(np.quantile(a,.95))))
        for person in range(1,41):
            mat=np.load(OUT/f'{stem(task,person)}_recipe_projection.npz')
            for policy in ['brain_argmax','artifact80']:
                gain=float((mat[policy]@s)[roi].mean(0)@weights(task)/den)
                rows.append(dict(task=task,subject=person,method='ica_'+policy,gain=gain,scope='Frozen-ICA linear action only; decomposition and labels not refitted after injection'))
        for policy in ['brain_argmax','artifact80']:
            a=np.array([r['gain'] for r in rows if r['task']==task and r['method']=='ica_'+policy and r['subject'] in C['evaluation']])
            summary.append(dict(task=task,method='ica_'+policy,n=len(a),mean_gain=float(a.mean()),q05_gain=float(np.quantile(a,.05)),q95_gain=float(np.quantile(a,.95))))
    write_json(OUT/'probes.json',dict(rows=rows,summary=summary,config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__))))

if __name__=='__main__':assess();probes()
