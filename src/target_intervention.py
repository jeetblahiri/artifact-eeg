"""Matched-input intervention on training targets, with a fixed affine learner.

Changing targets is the sole training intervention. The parent target remains
a measured comparator, not a true neural source. Data never leave development.
"""
import json
from pathlib import Path
from common import ROOT,sha256,write_json,config,normalize
import numpy as np
from erp_validation import baseline,stem

C=json.loads((ROOT/'causal_revision_config.json').read_text());TC=C['target_intervention']
OUT=ROOT/'results/causal_revision';CACHE=ROOT/'data/causal_revision';OLD=ROOT/'data/erp_validation'

def design(task,n,split,rng):
    pools={m:[] for m in TC['targets']};provenance=[]
    for s in C['development']:
        ids=np.load(OLD/f'{stem(task,s)}_identity.npz');perm=np.random.default_rng(TC['seed']+s).permutation(len(ids['label']))
        selected=perm[:len(perm)//2] if split=='train' else perm[len(perm)//2:]
        for m in pools:
            p=(OLD if m=='parent' else CACHE)/f'{stem(task,s)}_{m}.npy'
            pools[m].append(np.load(p)[selected].reshape(-1,512))
        provenance.append(dict(subject=s,trial_indices=selected.tolist()))
    pools={m:np.concatenate(v).astype(float) for m,v in pools.items()}
    eog=normalize(np.load(ROOT/Path(config()['source_dir'])/'EOG_all_epochs.npy'))
    manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text())
    kept=manifest['sources']['EOG']['kept_original_rows']
    source=[kept[i] for i in manifest['pools']['EOG'][split]]
    si=rng.integers(len(pools['parent']),size=n);ai=rng.choice(source,n);db=rng.choice(TC['power_snr_db'],n)
    parent=pools['parent'][si];artifact=eog[ai];coefficient=parent.std(1)*10.**(-db/20.)
    x=parent+coefficient[:,None]*artifact;scale=x.std(1,keepdims=True);x/=scale
    targets={m:v[si]/scale for m,v in pools.items()}
    return x,targets,dict(erp_trials=provenance,eog_original_indices=ai.tolist(),power_snr_db=db.tolist())

def main():
    for task in ['N170','P3']:
        dst=OUT/f'{task}_target_intervention.npz'
        if dst.exists():continue
        rng=np.random.default_rng(TC['seed']);x,targets,provenance=design(task,TC['fit_mixtures'],'train',rng)
        vx,vt,vp=design(task,TC['validation_mixtures'],'val',rng)
        xm=x.mean(0);xc=x-xm;gram=xc.T@xc;ridge=TC['ridge_fraction']*np.trace(gram)/512
        models={};losses={}
        for m,y in targets.items():
            ym=y.mean(0);matrix=np.linalg.solve(gram+ridge*np.eye(512),xc.T@(y-ym));intercept=ym-xm@matrix
            models[m]=(matrix,intercept);losses[m]={ref:float(np.mean((vx@matrix+intercept-z)**2)) for ref,z in vt.items()}
        np.savez(dst,**{m+'_matrix':a for m,(a,b) in models.items()},**{m+'_intercept':b for m,(a,b) in models.items()})
        np.savez_compressed(OUT/f'{task}_target_design.npz',x=x,vx=vx,**targets,**{'validation_'+m:y for m,y in vt.items()})
        for s in range(1,41):
            parent=np.load(OLD/f'{stem(task,s)}_parent.npy');a=parent.reshape(-1,512).astype(float);scale=a.std(1,keepdims=True)
            for target,(matrix,intercept) in models.items():
                z=(a/scale@matrix+intercept)*scale;z=baseline(z.reshape(parent.shape))
                np.save(CACHE/f'{stem(task,s)}_target_{target}.npy',z.astype(np.float32))
        write_json(OUT/f'{task}_target_intervention.json',dict(task=task,losses=losses,ridge=ridge,config=TC,
            source_development_only=True,shared_inputs=True,shared_regularization=True,shared_learner=True,
            provenance=provenance,validation_provenance=vp,config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__))))
        print('Matched-target fits',task,losses,flush=True)

if __name__=='__main__':main()
