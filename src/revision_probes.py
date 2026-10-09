"""Known-voltage response dispersion for the revised operators and published model."""
import json
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
from scipy.signal import butter,sosfiltfilt
from erp_validation import baseline,weights,guard_operator,load_femto,femto
from measurement_math import preserve_measurement
from revision_math import smooth_guard
from independent_denoiser import PublishedICUNet
from tim_revision import C,OUT,CACHE,OLD,stem

def main():
    model=PublishedICUNet();tiny=load_femto();rows=[];summary=[]
    for task,tc in C['tasks'].items():
        op=dict(np.load(OUT/f'{task}_operations.npz'))
        names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];roi=[names.index(c) for c in tc['roi']]
        spatial=np.full(30,-len(roi)/(30-len(roi)));spatial[roi]=1
        longtime=np.arange(32*256)/256-16
        full=spatial[:,None]*np.exp(-.5*((longtime-tc['probe_center_s'])/tc['probe_sd_s'])**2)
        longslice=np.arange(512)+int(15.5*256)
        probe=baseline(full[:,longslice]);den=float(probe[roi].mean(0)@weights(task))
        native_t=np.arange(1024)/256-1.5
        native_probe=spatial[:,None]*np.exp(-.5*((native_t-tc['probe_center_s'])/tc['probe_sd_s'])**2)
        l=guard_operator(task)
        for s in C['evaluation']:
            ids=np.load(OLD/f'{stem(task,s)}_identity.npz');selected=np.r_[np.flatnonzero(ids['label']==0)[:4],np.flatnonzero(ids['label']==1)[:4]]
            x=np.load(OLD/f'{stem(task,s)}_parent.npy')[selected].astype(float)
            e=np.load(OLD/f'{stem(task,s)}_eog.npy')[selected].astype(float)
            ctx=np.load(CACHE/f'{stem(task,s)}_probe_context.npy').astype(float)
            def apply(a):
                source=x+a*probe
                z=baseline(source-np.einsum('net,ec->nct',e-op['eog_mean'][None,:,None],op['beta']))
                ica=baseline(source-np.einsum('ck,nkt->nct',op['ica_remove'],source-op['ica_mean'][None,:,None]))
                learned=femto(source,tiny);independent=model.apply(ctx+a*native_probe)
                return dict(parent=source,regression=z,ica=ica,femto=learned,icunet=independent,
                    mapping=model.apply(ctx+a*native_probe,network=False),femto_guard=preserve_measurement(source,learned,l),
                    femto_smooth=smooth_guard(source,learned,l),icunet_smooth=smooth_guard(source,independent,l))
            zero=apply(0.)
            for amplitude in C['probe_amplitudes_uv']:
                response=apply(amplitude)
                for method in C['teachers']:
                    if method.startswith('hp'):
                        hp=float(method[2:]);filtered=baseline(sosfiltfilt(butter(4,[hp,30],btype='bandpass',fs=256,output='sos'),amplitude*full)[:,longslice])
                        gain=np.repeat(float(filtered[roi].mean(0)@weights(task))/(amplitude*den),len(x))
                    else:gain=((response[method]-zero[method])[:,roi].mean(1)@weights(task))/(amplitude*den)
                    for i,value in enumerate(gain):rows.append(dict(task=task,subject=s,method=method,amplitude_uv=amplitude,background_trial=int(selected[i]),gain=float(value)))
            print('Revision probes',task,s,flush=True)
        for method in C['teachers']:
            gain=np.array([r['gain'] for r in rows if r['task']==task and r['method']==method])
            summary.append(dict(task=task,method=method,n=len(gain),mean_gain=float(gain.mean()),sd_gain=float(gain.std(ddof=1)),
                minimum_gain=float(gain.min()),maximum_gain=float(gain.max()),q05_gain=float(np.quantile(gain,.05)),q95_gain=float(np.quantile(gain,.95))))
    write_json(OUT/'probes.json',dict(rows=rows,summary=summary,code_sha256=sha256(Path(__file__)),config_sha256=sha256(ROOT/'tim_revision_config.json'),
        scope='Finite paired response for the tested backgrounds and amplitudes; gain of a nonlinear operation is not a transfer function. Regression unity follows from holding EOG references fixed. Guard unity follows from its constraint.'))
    print('Probe records',len(rows),flush=True)

if __name__=='__main__':main()
