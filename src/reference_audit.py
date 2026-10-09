"""Observed reference-construction uncertainty, never an estimate of neural impurity."""
import json
from pathlib import Path
from common import ROOT,normalize,sha256,write_json,config
import numpy as np
from scipy.signal import butter,sosfiltfilt

PROFILES=[('Butterworth 1-20',1,20,'butter'),('Butterworth 1-30',1,30,'butter'),
          ('Butterworth 1-40',1,40,'butter'),('Spectral 1-30',1,30,'spectral'),
          ('Spectral 1-40',1,40,'spectral')]

def operation(x,lo,hi,kind):
    if kind=='butter':return sosfiltfilt(butter(4,[lo,hi],btype='bandpass',fs=256,output='sos'),x,axis=-1)
    f=np.fft.rfftfreq(x.shape[-1],1/256)
    return np.fft.irfft(np.fft.rfft(x,axis=-1)*((f>=lo)&(f<=hi)),n=x.shape[-1],axis=-1)

def crossed_source_ci(values,eeg_ids,artifact_ids):
    """Conditional pigeonhole bootstrap of both component-window banks."""
    _,ei=np.unique(eeg_ids,return_inverse=True);_,ai=np.unique(artifact_ids,return_inverse=True)
    ne,na=int(ei.max()+1),int(ai.max()+1);rng=np.random.default_rng(94621);means=[]
    for _ in range(3000):
        ew=np.bincount(rng.integers(ne,size=ne),minlength=ne)[ei]
        aw=np.bincount(rng.integers(na,size=na),minlength=na)[ai];w=ew*aw
        if w.sum():means.append(float(w@values/w.sum()))
    return np.quantile(means,[.025,.975]).tolist()

def main():
    source=Path(config()['source_dir'])/'EEG_all_epochs.npy'
    bank=normalize(np.load(source));corpus=[];rankings=[]
    for name,lo,hi,kind in PROFILES:
        modified=operation(bank,lo,hi,kind);fraction=np.sqrt(np.mean((bank-modified)**2,axis=1))
        corpus.append(dict(profile=name,reference_disagreement_rms_fraction=float(np.sqrt(np.mean((bank-modified)**2))),
            median_window_fraction=float(np.median(fraction)),q05_window_fraction=float(np.quantile(fraction,.05)),
            q95_window_fraction=float(np.quantile(fraction,.95)),windows=len(bank)))
    for artifact in ['EOG','EMG']:
        test=dict(np.load(ROOT/f'data/mixtures/{artifact}_test.npz'));t=test['s'].astype(float)
        predictions=dict(np.load(ROOT/f'results/mixtures/{artifact}/predictions.npz'))
        families={'affine':np.asarray([predictions['affine']],float),
                  'tiny_cnn':np.asarray([predictions[f'tiny_cnn_seed{s}'] for s in [17,29,43]],float),
                  'large_mlp':np.asarray([predictions[f'large_mlp_seed{s}'] for s in [17,29,43]],float)}
        rms=float(np.sqrt(np.mean(t*t)))
        for name,lo,hi,kind in [('Released target',0,128,'identity')]+PROFILES:
            ref=t if kind=='identity' else operation(t,lo,hi,kind)
            delta=t-ref;risks={k:float(np.mean((p-ref)**2)) for k,p in families.items()}
            for f,g in [('large_mlp','affine'),('tiny_cnn','affine'),('tiny_cnn','large_mlp')]:
                v=families[f].mean(0)-families[g].mean(0)
                d0=float(np.mean((families[f]-t)**2)-np.mean((families[g]-t)**2))
                d1=risks[f]-risks[g];inner=float(2*np.mean(v*delta))
                assert abs(d1-d0-inner)<1e-10
                kappa=float(np.sqrt(np.mean(v*v)));critical=abs(d0)/(2*kappa)
                shifted=float(np.sqrt(np.mean(delta*delta)))
                per_row=np.mean((families[f]-ref)**2,axis=(0,2))-np.mean((families[g]-ref)**2,axis=(0,2))
                interval=crossed_source_ci(per_row,test['eeg_id'],test['artifact_id'])
                rankings.append(dict(artifact=artifact,profile=name,f=f,g=g,risk_f=risks[f],risk_g=risks[g],
                    original_risk_difference=d0,modified_risk_difference=d1,measured_disagreement_fraction=shifted/rms,
                    critical_unrestricted_bias_fraction=critical/rms,
                    ranking_reversed=bool(d0*d1<0),path_tie_fraction=float(-d0/inner) if inner and d0*d1<0 else None,
                    uncertainty_interval=[d0-2*shifted*kappa,d0+2*shifted*kappa],
                    modified_risk_difference_ci95=interval,
                    interpretation='Observed sensitivity to reference definition. Disagreement is not a bound or estimate of latent neural bias.'))
    out=ROOT/'results/revision';out.mkdir(exist_ok=True)
    family_intervals=[]
    for artifact in ['EOG','EMG']:
        for f,g in [('large_mlp','affine'),('tiny_cnn','affine'),('tiny_cnn','large_mlp')]:
            rr=[r for r in rankings if r['artifact']==artifact and r['f']==f and r['g']==g]
            low=min(r['modified_risk_difference'] for r in rr);high=max(r['modified_risk_difference'] for r in rr)
            family_intervals.append(dict(artifact=artifact,f=f,g=g,lower=low,upper=high,strict_ranking_robust=bool(low>0 or high<0),
                scope='Exact ranking range on the convex hull of the six declared reference definitions, conditional on the fixed predictions. Not a neural-bias interval unless the desired target is independently justified to lie in this hull.'))
    write_json(out/'reference_audit.json',dict(source_path=str(source),source_sha256=sha256(source),
        eeg_windows=len(bank),dimensions=list(bank.shape),sampling_hz=256,corpus=corpus,rankings=rankings,family_intervals=family_intervals,
        interval_scope='Descriptive 95% crossed component-window bootstrap, conditional on the three selected fits; unavailable participant identities prevent person-level inference.',
        code_sha256=sha256(Path(__file__)),metadata_missing=['participant crosswalk','channel identity per window','event labels','simultaneous peripheral recordings','pre-ICA parent for each target'],
        source_purity_identifiable=False,
        interpretation='The released windows permit construction-disagreement measurements and exact fixed-prediction ranking checks. They do not identify neural truth. If multiple constructions are admitted for the same measurand, half their separation is a lower bound on the worst-case error of any common estimate; membership of true neural activity in that class is an explicit assumption.'))
    for r in corpus:print(r,flush=True)
    print('Observed ranking reversals:',[(r['artifact'],r['profile'],r['f'],r['g'],r['path_tie_fraction']) for r in rankings if r['ranking_reversed']],flush=True)

if __name__=='__main__':main()
