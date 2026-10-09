"""Finite-sample coverage audit and separately labelled fit-size sensitivity."""
import json,warnings
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from scipy.stats import betabinom,binom,hypergeom
from sklearn.decomposition import FastICA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from erp_validation import baseline,stem,feature,fit_decoder,weights
from tim_revision import C,ci
from measurement_math import conformal_quantile

OUT=ROOT/'results/causal_revision';OLD=ROOT/'data/erp_validation';REV=ROOT/'results/revision'

def diagnostics():
    rows=[r for task in ['N170','P3'] for r in json.loads((REV/f'{task}_metrics.json').read_text())['rows']]
    lookup={(r['task'],r['subject'],r['method']):r for r in rows}
    def score(s,family):
        return max(abs(lookup[t,s,m][k]-lookup[t,s,'parent'][k])/scale
                   for t in ['N170','P3'] for m in family for k,scale in [('amplitude_uv',1.),('frozen_ba',.02)])
    family=C['calibration_family'];families={}
    for label,ff in [('with_regression',family),('without_regression',[m for m in family if m!='regression'])]:
        c=[score(s,ff) for s in C['calibration']];e=[score(s,ff) for s in C['evaluation']]
        families[label]={f'q{int(100*(1-a))}':dict(quantile=conformal_quantile(c,a),voltage_uv=conformal_quantile(c,a),
            ba_points=2*conformal_quantile(c,a),test_covered=sum(v<=conformal_quantile(c,a) for v in e)) for a in [.1,.05]}
    subjects=C['calibration']+C['evaluation'];a=np.array([score(s,['ica']) for s in subjects]);q=conformal_quantile(a[:20],.1)
    rng=np.random.default_rng(94709);draws=100000;counts=[]
    for _ in range(draws):
        perm=rng.permutation(30);threshold=np.partition(a[perm[:20]],18)[18];counts.append(int(np.sum(a[perm[20:]]<=threshold)))
    counts=np.array(counts);observed=int(np.sum(a[20:]<=q))
    individual=[]
    for s,v in zip(subjects,a):
        entries=[dict(task=t,endpoint=k,score=abs(lookup[t,s,'ica'][k]-lookup[t,s,'parent'][k])/scale)
                 for t in ['N170','P3'] for k,scale in [('amplitude_uv',1.),('frozen_ba',.02)]]
        individual.append(dict(subject=s,split='calibration' if s in C['calibration'] else 'evaluation',score=float(v),dominating=max(entries,key=lambda r:r['score'])))
    return dict(family_comparison=families,ica=dict(quantile=q,observed_covered=observed,participant_scores=individual,
        fixed_90pct_binomial_probability=float(binom.cdf(observed,10,.9)),
        exchangeable_distinct_rank_probability=float(hypergeom.sf(18,30,20,19+observed)),
        iid_continuous_beta_binomial_probability=float(betabinom.cdf(observed,10,19,2)),
        frozen_score_random_reassignment_probability=float(np.mean(counts<=observed)),
        reassignment_draws=draws,covered_count_histogram=np.bincount(counts,minlength=11).tolist(),
        tie_count=30-len(np.unique(a)),calibration_max=float(a[:20].max()),evaluation_max=float(a[20:].max()),
        interpretation='Conditional rank diagnostic for a fixed fit and an exchangeable remaining population; ties conservative. Rare observed split is flagged, not excused as a heavy-tail effect. Reassignment is a diagnostic on inspected participants, not fresh validation.'))

def refit():
    old=json.loads((ROOT/'tim_validation_config.json').read_text());pool=old['development'];config=json.loads((ROOT/'causal_revision_config.json').read_text())
    rows=[]
    for seed in config['coverage_diagnostics']['refit_seeds']:
        dev=np.random.default_rng(seed).choice(pool,10,replace=False).tolist()
        for task in ['N170','P3']:
            p=np.concatenate([np.load(OLD/f'{stem(task,s)}_parent_fit.npy') for s in dev]).astype(float)
            h=np.concatenate([np.load(OLD/f'{stem(task,s)}_hp1_fit.npy') for s in dev]).astype(float)
            xx,ee=p[:,:30],p[:,30:];ec=ee-ee.mean(0);gram=ec.T@ec
            beta=np.linalg.solve(gram+1e-5*np.trace(gram)/3*np.eye(3),ec.T@(xx-xx.mean(0)))
            ica=FastICA(n_components=29,whiten='unit-variance',random_state=C['seed'],max_iter=1000,tol=.001)
            with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=2):z=ica.fit_transform(h[:,:30])
            cor=np.corrcoef(np.c_[z,h[:,30:]],rowvar=False)[:29,29:];strength=np.max(abs(cor),axis=1)
            bad=np.flatnonzero(strength>.3);bad=bad[np.argsort(strength[bad])[::-1]][:3];remove=ica.mixing_[:,bad]@ica.components_[bad]
            # Fix the primary decoder: this stage measures operator-fit changes.
            f=np.load(REV/f'{task}_features.npz');o=f['offsets'];slices={s:slice(o[s-1],o[s]) for s in range(1,41)}
            dec=fit_decoder(np.concatenate([f['parent'][slices[s]] for s in C['development']]),np.concatenate([f['label'][slices[s]] for s in C['development']]))
            names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(n) for n in C['tasks'][task]['roi']]
            for m in ['ica','regression']:
                amps=[];ba=[];scores=[]
                for s in C['calibration']+C['evaluation']:
                    parent=np.load(OLD/f'{stem(task,s)}_parent.npy').astype(float);e=np.load(OLD/f'{stem(task,s)}_eog.npy').astype(float)
                    x=baseline(parent-np.einsum('ck,nkt->nct',remove,parent-ica.mean_[None,:,None])) if m=='ica' else baseline(parent-np.einsum('net,ec->nct',e-ee.mean(0)[None,:,None],beta))
                    y=f['label'][slices[s]];wave=(x[y==1].mean(0)-x[y==0].mean(0))[roi].mean(0);amp=float(wave@weights(task))
                    pred=dec[1].predict(dec[0].transform(feature(x.astype(np.float32))));pb=dec[1].predict(dec[0].transform(f['parent'][slices[s]]))
                    bv=float(balanced_accuracy_score(y,pred));pa=float(((parent[y==1].mean(0)-parent[y==0].mean(0))[roi].mean(0))@weights(task))
                    if s in C['evaluation']:amps.append(amp);ba.append(bv)
                    scores.append(dict(subject=s,score=max(abs(amp-pa),abs(bv-balanced_accuracy_score(y,pb))/.02)))
                rows.append(dict(seed=seed,task=task,method=m,development=dev,fit_n=10,mean_test_amplitude_uv=float(np.mean(amps)),
                    mean_test_frozen_ba=float(np.mean(ba)),scores=scores,removed_components=bad.tolist(),ica_iterations=int(ica.n_iter_),warnings=[str(w.message) for w in ww]))
            print('Fit-size diagnostic',seed,task,flush=True)
    return dict(rows=rows,original_pool=pool,scope='Five independent ten-of-original-twenty development subsets; primary decoder fixed. These people were inspected previously, and some enter current calibration. Fit sensitivity only, no prospective coverage claim.')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'coverage_diagnostics.json',dict(**diagnostics(),config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__))))
    write_json(OUT/'operator_refit_sensitivity.json',refit())
