"""Same five participant folds for the added external models and recipe arms."""
import json
from common import ROOT,write_json,sha256
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from erp_validation import fit_decoder

def main():
    out=ROOT/'results/causal_revision';methods=['simple_cnn','complex_cnn','recipe_filter','recipe_brain_argmax','recipe_artifact80','ica_brain_argmax','ica_artifact80']
    folds=json.loads((ROOT/'results/revision/crossfit_design.json').read_text())['folds'];rows=[];summary=[]
    for task in ['N170','P3']:
        parent=np.load(ROOT/f'results/revision/{task}_features.npz');store=np.load(out/f'{task}_features.npz');o=store['offsets'];y=store['label']
        np.testing.assert_array_equal(o,parent['offsets']);np.testing.assert_array_equal(y,parent['label'])
        for fold,held in enumerate(folds):
            train=np.setdiff1d(np.arange(1,41),held);idx=np.concatenate([np.arange(o[s-1],o[s]) for s in train]);dec=fit_decoder(parent['parent'][idx],y[idx])
            for s in held:
                sl=slice(o[s-1],o[s]);truth=y[sl];p=dec[1].predict(dec[0].transform(parent['parent'][sl]));pb=balanced_accuracy_score(truth,p)
                for m in methods:
                    pred=dec[1].predict(dec[0].transform(store[m][sl]));ba=balanced_accuracy_score(truth,pred)
                    rows.append(dict(task=task,subject=int(s),method=m,fold=fold,parent_ba=float(pb),method_ba=float(ba),ba_difference=float(ba-pb)))
        for m in methods:
            rr=[r for r in rows if r['task']==task and r['method']==m];summary.append(dict(task=task,method=m,participants=len(rr),mean_ba_difference=float(np.mean([r['ba_difference'] for r in rr]))))
    write_json(out/'crossfit.json',dict(rows=rows,summary=summary,folds=folds,methods=methods,code_sha256=sha256(__file__),
        scope='Descriptive all-40 decoder robustness. Paired-target learners excluded because their ten ERP training participants would enter held-out folds; exact conformal guarantee not claimed.'))
    print('Added cross-fit records',len(rows),flush=True)

if __name__=='__main__':main()
