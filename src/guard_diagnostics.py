"""Post-hoc QA of the diagnostic guard's unprotected spectral alteration."""
import json
from common import ROOT,write_json,sha256
import numpy as np

def main():
    cfg=json.loads((ROOT/'tim_validation_config.json').read_text());rows=[]
    freq=np.fft.rfftfreq(512,1/256);factor=np.ones(len(freq))*2;factor[[0,-1]]=1
    for task in cfg['tasks']:
        for method in ['regression_guard','femto_guard']:
            for sub in cfg['evaluation']:
                stem=f'{task}_{sub:03d}';p=ROOT/'data/erp_validation'
                guard=np.load(p/f'{stem}_{method}.npy').astype(float);candidate=np.load(p/f'{stem}_{method[:-6]}.npy').astype(float)
                correction=guard-candidate;spec=(abs(np.fft.rfft(correction))**2*factor).sum((0,1))
                rows.append(dict(task=task,method=method,subject=sub,correction_rms_uv=float(np.sqrt(np.mean(correction**2))),
                    above30_correction_energy_fraction=float(spec[freq>30].sum()/spec.sum())))
    summary=[dict(task=t,method=m,mean_correction_rms_uv=float(np.mean([r['correction_rms_uv'] for r in rows if r['task']==t and r['method']==m])),
                 mean_above30_correction_energy_fraction=float(np.mean([r['above30_correction_energy_fraction'] for r in rows if r['task']==t and r['method']==m]))) for t in cfg['tasks'] for m in ['regression_guard','femto_guard']]
    write_json(ROOT/'results/erp_validation/guard_diagnostics.json',dict(rows=rows,summary=summary,code_sha256=sha256(__file__),
        status='Post-hoc waveform QA after the primary outcomes and plots; no models, thresholds, scores or calibration choices changed',
        interpretation='A rectangular-window constraint has a piecewise-constant minimum-energy correction. Its >30-Hz discrete epoch-FFT energy is an unprotected numerical effect, not measured cortical distortion. Guard is a diagnostic control, not a recommended final EEG denoiser.'))
    print(summary)

if __name__=='__main__':main()
