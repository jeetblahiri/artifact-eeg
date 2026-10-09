"""All-40 cross-fitted sensitivity; do not call pooled CV quantiles split-conformal."""
import json,math
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from erp_validation import fit_decoder

def main():
    out=ROOT/'results/revision';cfg=json.loads((ROOT/'tim_revision_config.json').read_text())
    methods=['hp0.5','hp1','hp2','mapping','icunet','femto','femto_guard','femto_smooth']
    permutation=np.random.default_rng(94641).permutation(np.arange(1,41));folds=np.array_split(permutation,5)
    write_json(out/'crossfit_design.json',dict(seed=94641,folds=[f.tolist() for f in folds],methods=methods,
        status='Additional exploratory robustness check; chosen after primary results; no finite-sample claim for pooled decoder-dependent CV scores'))
    metrics=[];base=json.loads((out/'summary.json').read_text());lookup={}
    for task in cfg['tasks']:
        for r in json.loads((out/f'{task}_metrics.json').read_text())['rows']:lookup[task,r['subject'],r['method']]=r
        store=np.load(out/f'{task}_features.npz');offset=store['offsets'];y=store['label'];f={m:store[m] for m in ['parent']+methods}
        for fold,held in enumerate(folds):
            train=np.setdiff1d(np.arange(1,41),held);idx=np.concatenate([np.arange(offset[s-1],offset[s]) for s in train])
            dec=fit_decoder(f['parent'][idx],y[idx])
            for s in held:
                sl=slice(offset[s-1],offset[s]);truth=y[sl]
                p=dec[1].predict(dec[0].transform(f['parent'][sl]));pba=balanced_accuracy_score(truth,p)
                for m in methods:
                    prediction=dec[1].predict(dec[0].transform(f[m][sl]));ba=balanced_accuracy_score(truth,prediction)
                    a=lookup[task,int(s),m]['amplitude_uv']-lookup[task,int(s),'parent']['amplitude_uv']
                    metrics.append(dict(task=task,subject=int(s),fold=fold,method=m,parent_ba=float(pba),method_ba=float(ba),
                        ba_difference=float(ba-pba),amplitude_difference_uv=float(a)))
    summary=[]
    for task in cfg['tasks']:
        for m in methods:
            rr=[r for r in metrics if r['task']==task and r['method']==m]
            summary.append(dict(task=task,method=m,participants=len(rr),mean_parent_ba=float(np.mean([r['parent_ba'] for r in rr])),
                mean_ba_difference=float(np.mean([r['ba_difference'] for r in rr])),
                mean_amplitude_difference_uv=float(np.mean([r['amplitude_difference_uv'] for r in rr]))))
    method_scores={};fixed_voltage={}
    for m in methods:
        values=[];voltage=[]
        for s in range(1,41):
            rr=[r for r in metrics if r['subject']==s and r['method']==m]
            av=max(abs(r['amplitude_difference_uv']) for r in rr);bv=max(abs(r['ba_difference'])/.02 for r in rr)
            values.append(max(av,bv));voltage.append(av)
        method_scores[m]=dict(empirical_q90=float(np.quantile(values,.9)),empirical_q95=float(np.quantile(values,.95)),
            guarantee='Descriptive pooled cross-fitted score quantiles; no exact split-conformal coverage.')
        # These voltages do not depend on any ERP-fitted decoder or operator.
        # A fixed external operation permits ordinary rank calibration on all
        # forty people; the same cohort gives no fresh coverage evaluation.
        k=math.ceil(41*.95)
        fixed_voltage[m]=dict(q95_uv=float(np.sort(voltage)[k-1]),calibration_n=40,rank=k,
            scope='Prospective marginal voltage-change envelope for a fixed operation under exchangeability, simultaneous over two tasks; not fresh empirical coverage validation or cortical error.')
    write_json(out/'crossfit_assessment.json',dict(rows=metrics,summary=summary,methods=methods,folds=[f.tolist() for f in folds],
        method_scores=method_scores,fixed_operation_voltage_calibration=fixed_voltage,code_sha256=sha256(Path(__file__)),
        training_subjects_per_fold=32,scored_participants=40,
        interpretation='Five disjoint held-out folds use all forty people for the static operations. The primary 10/20/10 split supplies the separate split-conformal assessment. Calibration and CV dispersion are not interchangeable.'))
    print('All-40 cross-fit assessment complete',len(metrics),'records',flush=True)

if __name__=='__main__':main()
