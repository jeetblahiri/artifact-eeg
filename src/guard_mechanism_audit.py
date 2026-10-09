"""Retain the original decoder when testing the guard's proposed alternative mechanism."""
import json
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def main():
    old=json.loads((ROOT/'tim_validation_config.json').read_text());new=json.loads((ROOT/'tim_revision_config.json').read_text())
    t=np.arange(512)/256-.5;rows=[]
    for task in old['tasks']:
        d=dict(np.load(ROOT/f'results/erp_validation/{task}_parent_decoder.npz'))
        for s in old['evaluation']:
            y=np.load(ROOT/f'data/erp_validation/{task}_{s:03d}_identity.npz')['label']
            for m in ['parent','femto','femto_guard','femto_smooth','femto_bandlimited']:
                location='erp_validation' if m in ['parent','femto'] else 'tim_revision'
                x=np.load(ROOT/f'data/{location}/{task}_{s:03d}_{m}.npy').astype(float)
                f=np.concatenate([x[:,:,(t>=a)&(t<a+.1)].mean(-1) for a in np.arange(0,.8,.1)],axis=1)
                z=((f-d['scale_mean'])/d['scale'])@d['coef'].T+d['intercept'];pred=d['classes'][(z[:,0]>0).astype(int)]
                rows.append(dict(task=task,subject=s,method=m,original_decoder_ba=float(balanced_accuracy_score(y,pred))))
    summary=[]
    for task in old['tasks']:
        for m in ['parent','femto','femto_guard','femto_smooth','femto_bandlimited']:
            rr=[r for r in rows if r['task']==task and r['method']==m]
            parent=[r for r in rows if r['task']==task and r['method']=='parent']
            delta=np.array([r['original_decoder_ba']-p['original_decoder_ba'] for r,p in zip(rr,parent)])
            rng=np.random.default_rng(94621);q=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975])
            summary.append(dict(task=task,method=m,mean_ba=float(np.mean([r['original_decoder_ba'] for r in rr])),
                mean_difference=float(delta.mean()),difference_ci95=q.tolist()))
    write_json(ROOT/'results/revision/guard_mechanism_audit.json',dict(rows=rows,summary=summary,
        original_development=old['development'],original_evaluation=old['evaluation'],code_sha256=sha256(Path(__file__)),
        scope='Post-hoc mechanism control retains the original 20-person-trained decoder and ten evaluation people. Not part of the revised 20-person calibration. A residual deficit is sensor-representation change, not measured cortical information loss.'))
    print(summary)

if __name__=='__main__':main()
