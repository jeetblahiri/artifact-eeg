"""Pinned published IC-U-Net, with explicit montage and physical-scale controls."""
import importlib.util,json
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
import mne,torch
from scipy.signal import resample_poly
from mne.channels.interpolation import _make_interpolation_matrix
from erp_validation import baseline,continuous_filter,stem

TEMPLATE=['Fp1','Fp2','F7','F3','Fz','F4','F8','FT7','FC3','FCz','FC4','FT8',
          'T7','C3','Cz','C4','T8','TP7','CP3','CPz','CP4','TP8','P7','P3','Pz','P4','P8','O1','Oz','O2']
SOURCE=ROOT/'vendor/AIEEG/model/cumbersome_model2.py'
CHECKPOINT=ROOT/'vendor/AIEEG/model/ICUNet/modelsave/checkpoint.pth.tar'
OUT=ROOT/'results/revision';CACHE=ROOT/'data/tim_revision'

def mapping(names):
    positions=mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    bylower={k.lower():v for k,v in positions.items()}
    a=np.asarray([bylower[n.lower()] for n in names]);b=np.asarray([bylower[n.lower()] for n in TEMPLATE])
    forward=_make_interpolation_matrix(a,b,alpha=1e-5)
    backward=_make_interpolation_matrix(b,a,alpha=1e-5)
    # Known matching electrodes reproduce their input exactly; only absent
    # electrodes use interpolation. Both directions retain their own order.
    for i,n in enumerate(TEMPLATE):
        if n.lower() in [x.lower() for x in names]:
            forward[i]=0;forward[i,[x.lower() for x in names].index(n.lower())]=1
    for i,n in enumerate(names):
        if n.lower() in [x.lower() for x in TEMPLATE]:
            backward[i]=0;backward[i,[x.lower() for x in TEMPLATE].index(n.lower())]=1
    return forward,backward

class PublishedICUNet:
    def __init__(self):
        spec=importlib.util.spec_from_file_location('published_cumbersome_model2',SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.device='mps' if torch.backends.mps.is_available() else 'cpu'
        torch.set_num_threads(2)
        self.model=module.UNet1(n_channels=30,n_classes=30)
        # The author checkpoint also stores NumPy training-loss metadata.
        # Allow only its array reconstruction/types; retain the safe loader.
        allowed=[(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,
                 type(np.dtype('float64')),type(np.dtype('float32'))]
        with torch.serialization.safe_globals(allowed):
            state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state['state_dict'],strict=True)
        self.model=self.model.to(self.device).eval()
        names=json.loads((ROOT/'data/erp_validation/N170_001.json').read_text())['channels'][:30]
        self.forward,self.backward=mapping(names)
        OUT.mkdir(exist_ok=True)
        np.savez(OUT/'icunet_montage.npz',forward=self.forward,backward=self.backward)
        write_json(OUT/'icunet_provenance.json',dict(model='IC-U-Net',paper_doi='10.1016/j.neuroimage.2022.119586',
            source_repository='https://github.com/roseDwayane/AIEEG',source_commit='7f4f27dbf79c0909a0993f680209cf24c32f7791',
            source_sha256=sha256(SOURCE),checkpoint_sha256=sha256(CHECKPOINT),strict_state_loading=True,
            parameters=sum(p.numel() for p in self.model.parameters()),device=self.device,source_channels=names,template_channels=TEMPLATE,
            preprocessing='Common 0.1-30 Hz continuously filtered parent; native 1024 samples / 4 seconds of actual context; global window mean/std as author code; output scale restored; center 512 samples; original baseline correction',
            mapping='Fixed spherical-spline forward/backward maps with exact matching electrodes; geometry only; mapping-only output supplied as control',
            fine_tuning=False,scope='Published pretrained model under a declared ERP deployment; not a recreation of original training acquisition or an attribution of errors to reference bias'))

    def apply(self,context,network=True):
        mapped=np.einsum('kc,nct->nkt',self.forward,context)
        if network:
            std=mapped.std((1,2),keepdims=True);mean=mapped.mean((1,2),keepdims=True)
            if np.any(std<1e-9):raise ValueError('Degenerate IC-U-Net window')
            chunks=[]
            with torch.no_grad():
                for i in range(0,len(mapped),16):
                    a=torch.from_numpy(((mapped[i:i+16]-mean[i:i+16])/std[i:i+16]).astype(np.float32)).to(self.device)
                    chunks.append(self.model(a).cpu().numpy()*std[i:i+16])
            mapped=np.concatenate(chunks)
        reconstructed=np.einsum('ck,nkt->nct',self.backward,mapped)
        return baseline(reconstructed[:,:,256:768])

def prepare_independent(task,subject,model,force=False):
    CACHE.mkdir(exist_ok=True)
    name=stem(task,subject)
    paths=[CACHE/f'{name}_{m}.npy' for m in ['icunet','mapping']]
    if not force and all(p.exists() for p in paths):return
    metadata=json.loads((ROOT/f'data/erp_validation/{name}.json').read_text())
    p=ROOT/f'data/erp_core/sub-{subject:03d}/eeg/sub-{subject:03d}_task-{task}_eeg.set'
    raw=mne.io.read_raw_eeglab(p,preload=True,verbose='ERROR')
    x=resample_poly(raw.get_data()*1e6,1,4,axis=-1);x[:30]-=x[:30].mean(0,keepdims=True)
    f=continuous_filter(x,.1,np.asarray(metadata['bounds_256']))[:30]
    ids=np.load(ROOT/f'data/erp_validation/{name}_identity.npz')
    indices=ids['sample'][:,None]+np.arange(1024)-384
    assert indices.min()>=0 and indices.max()<f.shape[1]
    bounds=metadata['bounds_256']
    assert not any(np.any((indices[:,0]<b)&(indices[:,-1]>=b)) for b in bounds[1:-1])
    context=f[:,indices].transpose(1,0,2);assert np.isfinite(context).all()
    parent=baseline(context[:,:,256:768])
    np.testing.assert_allclose(parent,np.load(ROOT/f'data/erp_validation/{name}_parent.npy'),atol=1e-4,rtol=2e-7)
    selected=np.r_[np.flatnonzero(ids['label']==0)[:4],np.flatnonzero(ids['label']==1)[:4]]
    np.save(CACHE/f'{name}_probe_context.npy',context[selected].astype(np.float32))
    for network,method in [(False,'mapping'),(True,'icunet')]:
        np.save(CACHE/f'{name}_{method}.npy',model.apply(context,network=network).astype(np.float32))
    write_json(CACHE/f'{name}_independent.json',dict(source_set_sha256=sha256(p),source_fdt_sha256=sha256(p.with_suffix('.fdt')),
        selected_probe_trials=selected.tolist(),full_context_samples=1024,epochs=len(ids['label']),source_code_sha256=sha256(Path(__file__)),
        parent_center_verified=True,recording_boundary_crossings=0,device=model.device))
    print(f'IC-U-Net and mapping control: {name}',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');args=parser.parse_args()
    model=PublishedICUNet()
    for task in ['N170','P3']:
        for subject in range(1,41):prepare_independent(task,subject,model,force=args.force)
