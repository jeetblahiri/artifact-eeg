"""Held-out, matched synthetic input control for the target intervention."""
import json
from pathlib import Path
from common import ROOT,config,normalize,write_json,sha256
import numpy as np
from erp_validation import baseline,weights,stem,feature,fit_decoder
from sklearn.metrics import balanced_accuracy_score
from tim_revision import ci

def main():
    out=ROOT/'results/causal_revision';cache=ROOT/'data/causal_revision';old=ROOT/'data/erp_validation'
    c=json.loads((ROOT/'causal_revision_config.json').read_text());tasks=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks'];rows=[];losses=[]
    manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text());kept=manifest['sources']['EOG']['kept_original_rows'];pool=[kept[i] for i in manifest['pools']['EOG']['test']]
    eog=normalize(np.load(ROOT/Path(config()['source_dir'])/'EOG_all_epochs.npy'))
    for j,(task,tc) in enumerate(tasks.items()):
        f=np.load(ROOT/f'results/revision/{task}_features.npz');offset=f['offsets'];sl={s:slice(offset[s-1],offset[s]) for s in range(1,41)}
        dec=fit_decoder(np.concatenate([f['parent'][sl[s]] for s in c['development']]),np.concatenate([f['label'][sl[s]] for s in c['development']]))
        model=np.load(out/f'{task}_target_intervention.npz');names=json.loads((old/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in tc['roi']]
        for s in c['evaluation']:
            parent=np.load(old/f'{stem(task,s)}_parent.npy').astype(float);shape=parent.shape;a=parent.reshape(-1,512)
            rng=np.random.default_rng(94751+1000*j+s);ai=rng.choice(pool,len(a));db=rng.choice(c['target_intervention']['power_snr_db'],len(a))
            mixed=a+a.std(1,keepdims=True)*10.**(-db[:,None]/20.)*eog[ai];scale=mixed.std(1,keepdims=True)
            np.save(cache/f'{stem(task,s)}_intervention_contaminated_inputs.npy',mixed.reshape(shape).astype(np.float32))
            y=f['label'][sl[s]];references={m:np.load((old if m=='parent' else cache)/f'{stem(task,s)}_{m}.npy').reshape(-1,512).astype(float) for m in c['target_intervention']['targets']}
            for target in c['target_intervention']['targets']:
                prediction=(mixed/scale@model[target+'_matrix']+model[target+'_intercept'])*scale
                z=baseline(prediction.reshape(shape)).astype(np.float32);np.save(cache/f'{stem(task,s)}_contaminated_target_{target}.npy',z)
                amp=float(((z[y==1].astype(float).mean(0)-z[y==0].astype(float).mean(0))[roi].mean(0))@weights(task))
                pred=dec[1].predict(dec[0].transform(feature(z)));rows.append(dict(task=task,subject=s,target=target,amplitude_uv=amp,frozen_ba=float(balanced_accuracy_score(y,pred))))
                for ref,truth in references.items():losses.append(dict(task=task,subject=s,target=target,reference=ref,normalized_mse=float(np.mean(((prediction-truth)/scale)**2))))
            write_json(out/f'{stem(task,s)}_target_contamination.json',dict(eog_original_indices=ai.tolist(),power_snr_db=db.tolist(),source_pool='test',input_shared_across_targets=True,seed=94751+1000*j+s))
    summary=[]
    for task in tasks:
        for target in c['target_intervention']['targets']:
            rr=[r for r in rows if r['task']==task and r['target']==target]
            summary.append(dict(task=task,target=target,mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr]))))
    pairs=[]
    for task in tasks:
        for target in ['recipe_brain_argmax','recipe_artifact80']:
            lookup={(r['task'],r['subject'],r['target']):r for r in rows};d=[lookup[task,s,target]['amplitude_uv']-lookup[task,s,'parent']['amplitude_uv'] for s in c['evaluation']]
            pairs.append(dict(task=task,target=target,amplitude_difference_uv=float(np.mean(d)),ci95=ci(d)))
    write_json(out/'target_contamination_control.json',dict(rows=rows,losses=losses,summary=summary,pairs=pairs,config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(__file__),
        scope='All fits see the identical held-out synthetic EOG mixtures. References remain constructed comparators. This control isolates teacher effects under matched contamination, not physical artifact removal or pure cortical recovery.'))
    print('Matched contaminated-input control',len(rows),'endpoint records',flush=True)

if __name__=='__main__':main()
