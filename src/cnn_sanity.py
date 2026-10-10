"""No-added-artifact and interface sensitivity checks for the historical CNNs."""
import json
from pathlib import Path
from common import ROOT, config, normalize, sha256, write_json
import numpy as np
from benchmark_models import load, predict, apply
from erp_validation import baseline, stem, weights

OUT = ROOT / 'results/nonlinear_revision'
CACHE = ROOT / 'data/nonlinear_revision'
OLD = ROOT / 'data/erp_validation'
CFG = json.loads((ROOT / 'nonlinear_revision_config.json').read_text())
PART = json.loads((ROOT / 'causal_revision_config.json').read_text())


def metrics(pred, target):
    p, t = np.asarray(pred, float).reshape(-1, 512), np.asarray(target, float).reshape(-1, 512)
    energy = np.sum(t*t, axis=1)
    relative = np.sqrt(np.sum((p-t)**2, axis=1)/energy)
    pc, tc = p-p.mean(1, keepdims=True), t-t.mean(1, keepdims=True)
    corr = np.sum(pc*tc, 1)/np.maximum(np.linalg.norm(pc, axis=1)*np.linalg.norm(tc, axis=1), 1e-12)
    return dict(normalized_mse=float(np.mean((p-t)**2)/np.mean(t*t)),
                pooled_relative_rms=float(np.sqrt(np.sum((p-t)**2)/np.sum(t*t))),
                median_window_relative_rms=float(np.median(relative)),
                mean_correlation=float(corr.mean()),
                through_origin_gain=float(np.sum(p*t)/np.sum(t*t)),
                output_to_input_rms=float(np.sqrt(np.sum(p*p)/np.sum(t*t))),
                fraction_below_10pct_relative_rms=float(np.mean(relative<=.1)))


def physical(x, models, centered=False, fixed_scale=None):
    shape=x.shape; a=np.asarray(x,float).reshape(-1,512)
    if centered: a=a-a.mean(1,keepdims=True)
    scale=a.std(1,keepdims=True) if fixed_scale is None else np.broadcast_to(fixed_scale,shape[:2]).reshape(-1,1)
    assert np.min(scale)>1e-9
    z=np.mean([predict(m,a/scale,batch=CFG['inference_batch_size']) for m in models],axis=0)
    return baseline((z*scale).reshape(shape)).astype(np.float32)


def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data/mixtures/manifest.json').read_text())
    original=np.array(manifest['sources']['EEG']['kept_original_rows'])[manifest['pools']['EEG']['test']]
    bank=normalize(np.load(ROOT/Path(config()['source_dir'])/'EEG_all_epochs.npy'))[original].astype(np.float32)
    np.save(OUT/'uncontaminated_benchmark_inputs.npy',bank)
    rows=[];controls=[];individual=[]
    for architecture in CFG['architectures']:
        models=[load(architecture,seed) for seed in CFG['seeds']]
        predictions=np.stack([predict(m,bank) for m in models]);np.save(OUT/f'{architecture}_uncontaminated_predictions.npy',predictions)
        for seed,z in zip(CFG['seeds'],predictions):individual.append(dict(architecture=architecture,seed=seed,**metrics(z,bank)))
        rows.append(dict(architecture=architecture,n_reference_windows=len(bank),**metrics(predictions.mean(0),bank)))
        for task in CFG['tasks']:
            # Fixed channel SD uses development voltages only; it is an interface
            # sensitivity test, not a replacement learned preprocessing rule.
            sd=np.concatenate([np.load(OLD/f'{stem(task,s)}_parent.npy').std(-1) for s in PART['development']],axis=0)
            fixed=np.median(sd,axis=0)[None,:]
            for person in PART['evaluation']:
                name=stem(task,person);x=np.load(OLD/f'{name}_parent.npy');labels=np.load(OLD/f'{name}_identity.npz')['label']
                names=json.loads((OLD/f'{task}_001.json').read_text())['channels'];tc=json.loads((ROOT/'tim_revision_config.json').read_text())['tasks'][task];roi=[names.index(n) for n in tc['roi']]
                for mode in ['delivered','center_window','fixed_development_sd']:
                    z=np.load(ROOT/f'data/causal_revision/{name}_{architecture}.npy') if mode=='delivered' else physical(x,models,centered=mode=='center_window',fixed_scale=fixed if mode=='fixed_development_sd' else None)
                    if mode!='delivered':np.save(CACHE/f'{name}_{architecture}_{mode}.npy',z)
                    contrast=float((z[labels==1].astype(float).mean(0)-z[labels==0].astype(float).mean(0))[roi].mean(0)@weights(task))
                    controls.append(dict(task=task,subject=person,architecture=architecture,mode=mode,contrast_uv=contrast,**metrics(z,x)))
                meta=json.loads((ROOT/f'data/tim_revision/{name}_independent.json').read_text());ids=np.array(meta['selected_probe_trials'])
                context=np.load(ROOT/f'data/tim_revision/{name}_probe_context.npy').astype(float)
                central=baseline(context[:,:,256:768]);np.testing.assert_allclose(central,x[ids],atol=1e-3,rtol=1e-6)
                # Parent/context caches were rounded independently to float32.
                context_roundoff=float(np.max(np.abs(central-x[ids])))
                # Establish one common baseline on the full context before moving
                # window boundaries; no zero padding or invented context.
                context-=context[:,:,256+np.flatnonzero((np.arange(512)/256-.5>=-.2)&(np.arange(512)/256-.5<0))].mean(-1,keepdims=True)
                summed=np.zeros_like(context);count=np.zeros(1024)
                for offset in [0,128,256,384,512]:
                    chunk=context[:,:,offset:offset+512];a=chunk.reshape(-1,512);scale=a.std(1,keepdims=True)
                    z=np.mean([predict(m,a/scale) for m in models],axis=0)*scale
                    summed[:,:,offset:offset+512]+=z.reshape(chunk.shape);count[offset:offset+512]+=1
                overlap=baseline((summed/count[None,None,:])[:,:,256:768]).astype(np.float32)
                np.save(CACHE/f'{name}_{architecture}_overlap_backgrounds.npy',overlap)
                for mode,z in [('central_backgrounds',np.load(ROOT/f'data/causal_revision/{name}_{architecture}.npy')[ids]),('overlap_backgrounds',overlap)]:
                    y=labels[ids];contrast=float((z[y==1].astype(float).mean(0)-z[y==0].astype(float).mean(0))[roi].mean(0)@weights(task))
                    controls.append(dict(task=task,subject=person,architecture=architecture,mode=mode,contrast_uv=contrast,background_trial_ids=ids.tolist(),context_roundoff_uv=context_roundoff,**metrics(z,x[ids])))
                print('Sanity',architecture,task,person,flush=True)
    summary=[]
    for task in CFG['tasks']:
        for a in CFG['architectures']:
            for mode in ['delivered','center_window','fixed_development_sd','central_backgrounds','overlap_backgrounds']:
                rr=[r for r in controls if (r['task'],r['architecture'],r['mode'])==(task,a,mode)]
                summary.append(dict(task=task,architecture=a,mode=mode,n_people=len(rr),mean_contrast_uv=float(np.mean([r['contrast_uv'] for r in rr])),mean_relative_rms=float(np.mean([r['pooled_relative_rms'] for r in rr]))))
    write_json(OUT/'cnn_sanity.json',dict(benchmark=rows,individual_seeds=individual,erp_rows=controls,erp_summary=summary,heldout_original_rows=original.tolist(),
        source_bank_sha256=sha256(ROOT/Path(config()['source_dir'])/'EEG_all_epochs.npy'),code_sha256=sha256(Path(__file__)),config_sha256=sha256(ROOT/'nonlinear_revision_config.json'),
        scope='Uncontaminated surrogate reference and interface diagnostics, not clean cortical input or a necessity that any MMSE denoiser be the identity.'))


if __name__=='__main__':main()
