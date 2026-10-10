"""Paired-target CNN deployment on development and evaluation people only."""
import argparse
import gc
import json
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from common import ROOT, sha256, write_json
from erp_validation import baseline, feature, fit_decoder, stem, weights
from benchmark_models import apply
from matched_cnn import load
from tim_revision import ci

OUT = ROOT / 'results/nonlinear_revision'
CACHE = ROOT / 'data/nonlinear_revision'
OLD = ROOT / 'data/erp_validation'
CFG = json.loads((ROOT / 'nonlinear_revision_config.json').read_text())
PART = json.loads((ROOT / 'causal_revision_config.json').read_text())
TASKS = json.loads((ROOT / 'tim_revision_config.json').read_text())['tasks']


def roi(task):
    names = json.loads((OLD / f'{task}_001.json').read_text())['channels']
    return [names.index(n) for n in TASKS[task]['roi']]


def frozen_prediction(task, f):
    d = np.load(ROOT / f'results/revision/{task}_parent_decoder.npz')
    # Match StandardScaler.transform's in-place float32 arithmetic on the
    # delivered float32 features, including its rounding conventions.
    standardized=f.copy()
    standardized-=d['scale_mean'].astype(f.dtype);standardized/=d['scale'].astype(f.dtype)
    score = standardized @ d['coef'].T + d['intercept']
    return d['classes'][(score[:, 0] > 0).astype(int)]


def voltage(x, labels, task):
    wave = (x[labels == 1].astype(float).mean(0)-x[labels == 0].astype(float).mean(0))[roi(task)].mean(0)
    return float(wave @ weights(task)), wave


def output_tag(architecture, target):
    return f'matched_{architecture}_{target}'


def deployed(task, person, tag, condition, models, checkpoints):
    name = stem(task, person)
    source = OLD / f'{name}_parent.npy' if condition == 'parent_input' else ROOT / f'data/causal_revision/{name}_intervention_contaminated_inputs.npy'
    dst = CACHE / f'{name}_{tag}_{condition}.npy'
    meta = dst.with_suffix('.json')
    fingerprints = dict(input_sha256=sha256(source), checkpoint_sha256=checkpoints,
                        configuration_sha256=sha256(ROOT / 'nonlinear_revision_config.json'))
    if dst.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        assert all(saved[k] == v for k, v in fingerprints.items())
        return np.load(dst)
    x = np.load(source)
    seeds = apply(x, models, return_seeds=True).astype(np.float32)
    ensemble = seeds.astype(float).mean(0).astype(np.float32)
    np.save(dst, ensemble)
    labels = np.load(OLD / f'{name}_identity.npz')['label']
    np.savez_compressed(CACHE / f'{name}_{tag}_{condition}_seed_endpoints.npz',
                        features=np.stack([feature(z) for z in seeds]),
                        roi_contrasts=np.stack([voltage(z, labels, task)[1] for z in seeds]),
                        seed=np.array(CFG['seeds']), label=labels)
    write_json(meta, dict(**fingerprints, output_sha256=sha256(dst),
                         code_sha256=sha256(__file__), equal_seed_average=True,
                         subject=person, task=task, condition=condition,
                         inference='Per-channel input SD; restore physical scale; baseline each seed; equal average'))
    return ensemble


def main(wait_for_models=False):
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    rows=[]; seed_rows=[]; cross_losses=[]; transmission=[]
    people = PART['development'] + PART['evaluation']
    for architecture in CFG['architectures']:
        paths = [OUT / f'{architecture}_{seed}_{target}.json' for seed in CFG['seeds'] for target in CFG['targets']]
        while not all(p.exists() for p in paths):
            if not wait_for_models:
                raise FileNotFoundError('Complete the paired fixed-duration fits before assessment.')
            time.sleep(15)
        for target in CFG['targets']:
            tag = output_tag(architecture, target)
            models = [load(architecture, seed, target) for seed in CFG['seeds']]
            checkpoint_hashes = [sha256(OUT / f'{architecture}_{seed}_{target}.pth') for seed in CFG['seeds']]
            vp = np.mean([np.load(OUT / f'{architecture}_{seed}_{target}_validation_predictions.npy').astype(float)
                          for seed in CFG['seeds']], axis=0)
            design = np.load(OUT / 'matched_design.npz')
            for scored_target in CFG['targets']:
                cross_losses.append(dict(architecture=architecture, training_target=target,
                                         scoring_target=scored_target, ensemble_mse=float(np.mean((vp-design['validation_'+scored_target])**2))))
            for task in CFG['tasks']:
                labels = {s:np.load(OLD / f'{stem(task,s)}_identity.npz')['label'] for s in people}
                features={}; voltages={}; waves={}
                for person in people:
                    z=deployed(task, person, tag, 'parent_input', models, checkpoint_hashes)
                    features[person]=feature(z); voltages[person],waves[person]=voltage(z, labels[person], task)
                    print('Matched inference', architecture, target, task, person, flush=True)
                sc, clf = fit_decoder(np.concatenate([features[s] for s in PART['development']]),
                                      np.concatenate([labels[s] for s in PART['development']]))
                np.savez(OUT / f'{task}_{tag}_adapted_decoder.npz',scale_mean=sc.mean_, scale=sc.scale_,
                         coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
                for person in people:
                    y=labels[person]
                    rows.append(dict(architecture=architecture,target=target,task=task,subject=person,
                                     split='development' if person in PART['development'] else 'evaluation',condition='parent_input',
                                     amplitude_uv=voltages[person],frozen_ba=float(balanced_accuracy_score(y,frozen_prediction(task,features[person]))),
                                     adapted_ba=float(balanced_accuracy_score(y,clf.predict(sc.transform(features[person]))))))
                    seed_data=np.load(CACHE / f'{stem(task,person)}_{tag}_parent_input_seed_endpoints.npz')
                    for seed,f,w in zip(seed_data['seed'],seed_data['features'],seed_data['roi_contrasts']):
                        seed_rows.append(dict(architecture=architecture,target=target,task=task,subject=person,seed=int(seed),
                                              split='development' if person in PART['development'] else 'evaluation',condition='parent_input',
                                              amplitude_uv=float(w @ weights(task)),frozen_ba=float(balanced_accuracy_score(y,frozen_prediction(task,f)))))
                    if person not in PART['evaluation']:continue
                    original=np.load(CACHE / f'{stem(task,person)}_{tag}_parent_input.npy')
                    reref=baseline(original-original.mean(1,keepdims=True)).astype(np.float32)
                    np.save(CACHE / f'{stem(task,person)}_{tag}_output_reference.npy',reref)
                    contaminated=deployed(task,person,tag,'contaminated_input',models,checkpoint_hashes)
                    for condition,z in [('output_reference',reref),('contaminated_input',contaminated)]:
                        rows.append(dict(architecture=architecture,target=target,task=task,subject=person,split='evaluation',condition=condition,
                                         amplitude_uv=voltage(z,y,task)[0],frozen_ba=float(balanced_accuracy_score(y,frozen_prediction(task,feature(z)))),
                                         adapted_ba=float(balanced_accuracy_score(y,clf.predict(sc.transform(feature(z)))))))
                np.savez_compressed(OUT / f'{task}_{tag}_endpoint_features.npz',
                                    **{f'features_{s}':features[s] for s in people},
                                    **{f'wave_{s}':waves[s] for s in people})
            del models;gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
    lookup={(r['architecture'],r['target'],r['task'],r['subject'],r['condition']):r for r in rows}
    summary=[];pairs=[];seed_pairs=[]
    for architecture in CFG['architectures']:
        for task in CFG['tasks']:
            for condition in ['parent_input','output_reference','contaminated_input']:
                for target in CFG['targets']:
                    rr=[lookup[architecture,target,task,s,condition] for s in PART['evaluation']]
                    summary.append(dict(architecture=architecture,target=target,task=task,condition=condition,n_people=len(rr),
                                        mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),
                                        mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr])),
                                        mean_adapted_ba=float(np.mean([r['adapted_ba'] for r in rr]))))
                for target in CFG['targets'][1:]:
                    amp=np.array([lookup[architecture,target,task,s,condition]['amplitude_uv']-lookup[architecture,'parent',task,s,condition]['amplitude_uv'] for s in PART['evaluation']])
                    ba=np.array([lookup[architecture,target,task,s,condition]['frozen_ba']-lookup[architecture,'parent',task,s,condition]['frozen_ba'] for s in PART['evaluation']])
                    pairs.append(dict(architecture=architecture,target=target,task=task,condition=condition,amplitude_difference_uv=float(amp.mean()),
                                      amplitude_ci95=ci(amp),frozen_ba_difference=float(ba.mean()),frozen_ba_ci95=ci(ba)))
                    for person,learned in zip(PART['evaluation'],amp):
                        y=np.load(OLD / f'{stem(task,person)}_identity.npz')['label']
                        parent=np.load(OLD / f'{stem(task,person)}_parent.npy')
                        teacher=np.load(ROOT / f'data/causal_revision/{stem(task,person)}_{target}.npy')
                        direct=voltage(teacher,y,task)[0]-voltage(parent,y,task)[0]
                        # Exact endpoint decomposition; no cortical-truth interpretation.
                        z0=np.load(CACHE / f'{stem(task,person)}_{output_tag(architecture,"parent")}_{condition}.npy')
                        z1=np.load(CACHE / f'{stem(task,person)}_{output_tag(architecture,target)}_{condition}.npy')
                        residual=voltage(z1.astype(float)-teacher,y,task)[0]-voltage(z0.astype(float)-parent,y,task)[0]
                        transmission.append(dict(architecture=architecture,target=target,task=task,subject=person,condition=condition,
                                                 direct_reference_shift_uv=direct,learned_output_shift_uv=float(learned),residual_shift_uv=residual,
                                                 identity_discrepancy_uv=float(learned-direct-residual)))
            for seed in CFG['seeds']:
                by={(r['target'],r['subject']):r for r in seed_rows if (r['architecture'],r['task'],r['seed'],r['split'])==(architecture,task,seed,'evaluation')}
                for target in CFG['targets'][1:]:
                    a=np.array([by[target,s]['amplitude_uv']-by['parent',s]['amplitude_uv'] for s in PART['evaluation']])
                    seed_pairs.append(dict(architecture=architecture,task=task,target=target,seed=seed,
                                           mean_amplitude_difference_uv=float(a.mean()),ci95=ci(a)))
    write_json(OUT / 'matched_assessment.json',dict(rows=rows,individual_seeds=seed_rows,summary=summary,pairs=pairs,seed_pairs=seed_pairs,
               ensemble_cross_target_losses=cross_losses,endpoint_transmission=transmission,
               config_sha256=sha256(ROOT / 'nonlinear_revision_config.json'),code_sha256=sha256(__file__),
               development_people=PART['development'],evaluation_people=PART['evaluation'],
               scope='Fixed training-algorithm paired target effects on digital output, conditional on inspected cohort, checkpoints and chosen recipe. No released-label impurity, cortical preservation or causal account of historical CNN failures.'))
    print('Matched assessment complete:',len(rows),'ensemble and',len(seed_rows),'individual-seed records',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for-models',action='store_true');main(p.parse_args().wait_for_models)
