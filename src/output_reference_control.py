"""Check whether changed output common mode explains voltage/decode changes."""
import json
from common import ROOT,write_json,sha256
import numpy as np
from erp_validation import stem,weights,feature,fit_decoder
from sklearn.metrics import balanced_accuracy_score
from tim_revision import ci

def main():
    c=json.loads((ROOT/'causal_revision_config.json').read_text());tasks=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks'];cache=ROOT/'data/causal_revision';rows=[];summary=[]
    for task,tc in tasks.items():
        f=np.load(ROOT/f'results/revision/{task}_features.npz');o=f['offsets'];sl={s:slice(o[s-1],o[s]) for s in range(1,41)}
        dec=fit_decoder(np.concatenate([f['parent'][sl[s]] for s in c['development']]),np.concatenate([f['label'][sl[s]] for s in c['development']]))
        names=json.loads((ROOT/f'data/erp_validation/{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in tc['roi']]
        methods=['parent','femto','icunet','simple_cnn','complex_cnn','target_parent','target_recipe_brain_argmax','target_recipe_artifact80']
        for m in methods:
            for s in c['evaluation']:
                where=ROOT/'data/erp_validation' if m in ['parent','femto'] else ROOT/'data/tim_revision' if m=='icunet' else cache
                x=np.load(where/f'{stem(task,s)}_{m}.npy').astype(float);y=f['label'][sl[s]]
                for mode in ['delivered','output_average']:
                    z=x if mode=='delivered' else x-x.mean(1,keepdims=True)
                    contrast=(z[y==1].mean(0)-z[y==0].mean(0));amp=float(contrast[roi].mean(0)@weights(task));pred=dec[1].predict(dec[0].transform(feature(z.astype(np.float32))))
                    rows.append(dict(task=task,subject=s,method=m,mode=mode,amplitude_uv=amp,frozen_ba=float(balanced_accuracy_score(y,pred))))
            for mode in ['delivered','output_average']:
                rr=[r for r in rows if r['task']==task and r['method']==m and r['mode']==mode]
                summary.append(dict(task=task,method=m,mode=mode,mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr]))))
    lookup={(r['task'],r['subject'],r['method'],r['mode']):r for r in rows};pairs=[]
    for task in tasks:
        for target in ['target_recipe_brain_argmax','target_recipe_artifact80']:
            d=[lookup[task,s,target,'output_average']['amplitude_uv']-lookup[task,s,'target_parent','output_average']['amplitude_uv'] for s in c['evaluation']]
            pairs.append(dict(task=task,target=target,mean_amplitude_difference_uv=float(np.mean(d)),ci95=ci(d)))
    write_json(ROOT/'results/causal_revision/output_reference_control.json',dict(rows=rows,summary=summary,target_pairs=pairs,code_sha256=sha256(__file__),
        scope='Post-hoc output reference control specified after primary recipe/model outcomes; same delivered arrays and frozen decoder. All thirty output scalp channels are re-referenced together. This tests a common-mode explanation, not cortical purity.'))
    print('Output reference control',len(rows),'records',flush=True)

if __name__=='__main__':main()
