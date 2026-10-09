"""Geometrically smooth probe checks after the local montage-control audit."""
import json
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np,mne
from scipy.signal import butter,sosfiltfilt
from erp_validation import baseline,weights,guard_operator,load_femto,femto
from measurement_math import preserve_measurement
from revision_math import smooth_guard
from independent_denoiser import PublishedICUNet
from tim_revision import C,OUT,CACHE,OLD,stem

def spatial_direction(names,roi,width=.4):
    positions=mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    lookup={k.lower():v for k,v in positions.items()};p=np.asarray([lookup[n.lower()] for n in names[:30]])
    p=p/np.linalg.norm(p,axis=1,keepdims=True)
    distance=np.arccos(np.clip(p@p[roi].T,-1,1));v=np.exp(-.5*(distance/width)**2).mean(1)
    v-=v.mean();v/=v[roi].mean();return v

def main():
    model=PublishedICUNet();tiny=load_femto();rows=[];summary=[]
    for task,tc in C['tasks'].items():
        op=dict(np.load(OUT/f'{task}_operations.npz'));names=json.loads((OLD/f'{task}_001.json').read_text())['channels']
        roi=[names.index(c) for c in tc['roi']];spatial=spatial_direction(names,roi)
        longtime=np.arange(8192)/256-16;full=spatial[:,None]*np.exp(-.5*((longtime-tc['probe_center_s'])/tc['probe_sd_s'])**2)
        longslice=np.arange(512)+int(15.5*256);probe=baseline(full[:,longslice]);den=float(probe[roi].mean(0)@weights(task))
        native_t=np.arange(1024)/256-1.5;native_probe=spatial[:,None]*np.exp(-.5*((native_t-tc['probe_center_s'])/tc['probe_sd_s'])**2)
        l=guard_operator(task)
        for s in C['evaluation']:
            ids=np.load(OLD/f'{stem(task,s)}_identity.npz');selected=np.r_[np.flatnonzero(ids['label']==0)[:4],np.flatnonzero(ids['label']==1)[:4]]
            x=np.load(OLD/f'{stem(task,s)}_parent.npy')[selected].astype(float);e=np.load(OLD/f'{stem(task,s)}_eog.npy')[selected].astype(float)
            ctx=np.load(CACHE/f'{stem(task,s)}_probe_context.npy').astype(float)
            def apply(amplitude):
                source=x+amplitude*probe;learned=femto(source,tiny);independent=model.apply(ctx+amplitude*native_probe)
                return dict(parent=source,regression=baseline(source-np.einsum('net,ec->nct',e-op['eog_mean'][None,:,None],op['beta'])),
                    ica=baseline(source-np.einsum('ck,nkt->nct',op['ica_remove'],source-op['ica_mean'][None,:,None])),
                    femto=learned,icunet=independent,mapping=model.apply(ctx+amplitude*native_probe,network=False),
                    femto_guard=preserve_measurement(source,learned,l),femto_smooth=smooth_guard(source,learned,l),icunet_smooth=smooth_guard(source,independent,l))
            zero=apply(0)
            for a in C['probe_amplitudes_uv']:
                response=apply(a)
                for method in C['teachers']:
                    if method.startswith('hp'):
                        filtered=baseline(sosfiltfilt(butter(4,[float(method[2:]),30],btype='bandpass',fs=256,output='sos'),a*full)[:,longslice])
                        gain=np.repeat(float(filtered[roi].mean(0)@weights(task))/(a*den),len(x))
                    else:gain=((response[method]-zero[method])[:,roi].mean(1)@weights(task))/(a*den)
                    for i,g in enumerate(gain):rows.append(dict(task=task,subject=s,method=method,amplitude_uv=a,background_trial=int(selected[i]),gain=float(g)))
            print('Smooth spatial probes',task,s,flush=True)
        for method in C['teachers']:
            a=np.array([r['gain'] for r in rows if r['task']==task and r['method']==method])
            summary.append(dict(task=task,method=method,n=len(a),mean_gain=float(a.mean()),sd_gain=float(a.std(ddof=1)),
                minimum_gain=float(a.min()),maximum_gain=float(a.max()),q05_gain=float(np.quantile(a,.05)),q95_gain=float(np.quantile(a,.95))))
    write_json(OUT/'smooth_probes.json',dict(rows=rows,summary=summary,code_sha256=sha256(Path(__file__)),
        spatial_width_radians=.4,spatial_construction='Zero-mean geodesic Gaussian, centered on the declared ROI and normalized to ROI mean one',
        status='Additional geometry control specified after seeing local-probe attenuation in the mapping-only arm; width fixed without optimizing model responses',
        scope='A software-voltage direction that is more compatible with the mapped montage. Not an independently validated cortical source model.'))

if __name__=='__main__':main()
