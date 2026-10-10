"""Score the prerecording 1-100 Hz recipe sensitivity on the same ERP trials."""
import json
import numpy as np
from common import ROOT,sha256,write_json
from verify_nonlinear_revision import manual_endpoint
from filter_assessment import intervals

OUT=ROOT/'results/filter_revision';PART=json.loads((ROOT/'causal_revision_config.json').read_text())

def main():
    rows=[];metadata=[]
    for task in ['N170','P3']:
        for subject in range(1,41):
            name=f'{task}_{subject:03d}';labels=np.load(ROOT/f'data/erp_validation/{name}_identity.npz')['label']
            a=json.loads((OUT/f'passband100/{name}_recipe.json').read_text());old=json.loads((ROOT/f'results/causal_revision/{name}_recipe.json').read_text())
            metadata.append(dict(task=task,subject=subject,iterations_100=a['iterations'],iterations_80=old['iterations'],warnings_100=a['warnings'],
                rejected_100=a['rejected'],rejected_80=old['rejected'],configuration_sha256=a['config_sha256'],code_sha256=a['code_sha256']))
            for method in ['recipe_filter','recipe_brain_argmax','recipe_artifact80']:
                p=ROOT/f'data/filter_revision/passband100/{name}_{method}.npy';q=ROOT/f'data/causal_revision/{name}_{method}.npy'
                newer=manual_endpoint(np.load(p),labels,task)[0];primary=manual_endpoint(np.load(q),labels,task)[0]
                rows.append(dict(task=task,subject=subject,method=method,amplitude_100_uv=newer,amplitude_80_uv=primary,difference_uv=newer-primary,
                    output_100_sha256=sha256(p),output_80_sha256=sha256(q)))
    summary=[]
    for task in ['N170','P3']:
        for method in ['recipe_filter','recipe_brain_argmax','recipe_artifact80']:
            for split,people in [('all',list(range(1,41))),('evaluation',PART['evaluation'])]:
                selected=[r for r in rows if r['task']==task and r['method']==method and r['subject'] in people]
                d=np.array([r['difference_uv'] for r in selected])
                summary.append(dict(task=task,method=method,split=split,n_people=len(selected),mean_amplitude_100_uv=float(np.mean([r['amplitude_100_uv'] for r in selected])),
                    mean_amplitude_80_uv=float(np.mean([r['amplitude_80_uv'] for r in selected])),mean_difference_uv=float(d.mean()),ci95=intervals(d)))
    write_json(OUT/'passband_sensitivity.json',dict(rows=rows,summary=summary,recipe_metadata=metadata,
        iteration_limit_100=sum(r['iterations_100']>=500 for r in metadata),iteration_limit_80=sum(r['iterations_80']>=500 for r in metadata),
        configuration_sha256=sha256(ROOT/'filter_revision_config.json'),code_sha256=sha256(__file__),
        scope='Change only recipe upper passband from 80 to 100 Hz; final matched 30-Hz analysis unchanged; component selection is refitted and may change. No retraining on sensitivity labels; no cortical purity certification.'))

if __name__=='__main__':main()
