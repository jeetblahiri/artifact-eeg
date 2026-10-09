"""Automated instantiation of the reported reference recipe on ERP CORE.

The publication leaves thresholds/manual decisions unspecified; both policies
are sensitivity arms, and neither creates a known pure cortical reference.
"""
import json,warnings,os
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from scipy.signal import butter,sosfiltfilt,resample_poly,iirnotch,filtfilt
from threadpoolctl import threadpool_limits
from erp_validation import baseline,stem

C=json.loads((ROOT/'causal_revision_config.json').read_text())
OUT=ROOT/'results/causal_revision';CACHE=ROOT/'data/causal_revision';OLD=ROOT/'data/erp_validation'

def within(x,bounds,operation):
    y=np.full_like(x,np.nan)
    for a,b in zip(bounds[:-1],bounds[1:]):
        if b-a>768:y[:,a:b]=operation(x[:,a:b])
    return y

def prepare(task,subject):
    name=stem(task,subject);dst=OUT/f'{name}_recipe.json'
    if dst.exists():return
    p=ROOT/f'data/erp_core/sub-{subject:03d}/eeg/sub-{subject:03d}_task-{task}_eeg.set'
    original=mne.io.read_raw_eeglab(p,preload=True,verbose='ERROR')
    x=resample_poly(original.get_data()[:30],1,4,axis=-1);x-=x.mean(0,keepdims=True)
    meta=json.loads((OLD/f'{name}.json').read_text());bounds=meta['bounds_256']
    sos=butter(4,[1,80],btype='bandpass',fs=256,output='sos');nb,na=iirnotch(60,30,fs=256)
    high=within(x,bounds,lambda z:filtfilt(nb,na,sosfiltfilt(sos,z)))
    # The full finite stream retains real temporal context for ICLabel features.
    if not np.isfinite(high).all():raise ValueError('Nonfinite recording segments')
    info=mne.create_info(original.ch_names[:30],256,'eeg')
    raw=mne.io.RawArray(high,info,verbose='ERROR').set_montage('standard_1020',match_case=False,verbose='ERROR')
    raw.set_eeg_reference('average',verbose='ERROR')
    with raw.info._unlock():raw.info['highpass']=1.;raw.info['lowpass']=80.
    ica=ICA(n_components=29,method='infomax',fit_params=dict(extended=True),max_iter=500,random_state=C['seed']+subject)
    with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=2):
        ica.fit(raw,decim=4,verbose='ERROR')
        probabilities=iclabel_label_components(raw,ica,inplace=False,backend='torch')
    labels=np.argmax(probabilities,axis=1)
    bad={'brain_argmax':np.flatnonzero(labels!=0).tolist(),
         'artifact80':np.flatnonzero(probabilities[:,1:6].max(1)>=.8).tolist()}
    # Derive the exact linear action of the frozen ICA, eliminating affine
    # offsets by differencing zero and basis inputs. Apply before baseline.
    basis=mne.io.RawArray(np.c_[np.zeros(30),np.eye(30)],info,verbose='ERROR')
    project={}
    for policy,rejected in bad.items():
        z=ica.apply(basis.copy(),exclude=rejected,verbose='ERROR').get_data()
        project[policy]=z[:,1:]-z[:,:1]
    z=ica.apply(basis.copy(),exclude=[],verbose='ERROR').get_data()
    reconstruct=z[:,1:]-z[:,:1]
    control_error=float(np.max(abs(reconstruct-np.eye(30))))
    assert control_error<1e-10
    low=butter(4,30,btype='lowpass',fs=256,output='sos')
    matched=within(high,bounds,lambda z:sosfiltfilt(low,z))*1e6
    ids=np.load(OLD/f'{name}_identity.npz');indices=ids['sample'][:,None]+np.arange(512)-128
    epochs=baseline(matched[:,indices].transpose(1,0,2))
    parent=np.load(OLD/f'{name}_parent.npy').astype(float)
    np.save(CACHE/f'{name}_recipe_filter.npy',epochs.astype(np.float32))
    for policy,matrix in project.items():
        np.save(CACHE/f'{name}_recipe_{policy}.npy',baseline(np.einsum('ck,nkt->nct',matrix,epochs)).astype(np.float32))
        np.save(CACHE/f'{name}_ica_{policy}.npy',baseline(np.einsum('ck,nkt->nct',matrix,parent)).astype(np.float32))
    np.savez(OUT/f'{name}_recipe_projection.npz',probabilities=probabilities,reconstruct=reconstruct,**project)
    write_json(dst,dict(task=task,subject=subject,n_components=29,n_samples=int(raw.n_times),iterations=int(ica.n_iter_),
        rejected=bad,probabilities=probabilities.tolist(),maximum_identity_reconstruction_error=control_error,
        warnings=[str(w.message) for w in ww],source_sha256=sha256(p),source_fdt_sha256=sha256(p.with_suffix('.fdt')),
        config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__)),
        software=dict(mne=mne.__version__,mne_icalabel=__import__('mne_icalabel').__version__),
        scope='Frozen per-recording unsupervised recipe; class labels never enter fit or component selection; automatic threshold sensitivity, not reconstruction of undocumented expert selection'))
    print('Recipe',name,'iterations',ica.n_iter_,'rejected',bad,flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['N170','P3']);args=parser.parse_args()
    for task in ([args.task] if args.task else ['N170','P3']):
        for subject in range(1,41):prepare(task,subject)
