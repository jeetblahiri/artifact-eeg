"""Four-arm endpoint transmission: filtering, incremental selection, residuals."""
import json
import itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT,sha256,write_json
from erp_validation import stem,feature,fit_decoder,baseline
from matched_cnn_assessment import voltage,frozen_prediction

OUT=ROOT/'results/filter_revision';CACHE=ROOT/'data/filter_revision';OLD=ROOT/'data/erp_validation'
CFG=json.loads((ROOT/'filter_revision_config.json').read_text())
PART=json.loads((ROOT/'causal_revision_config.json').read_text())

def intervals(numerator,denominator=None):
    a=np.asarray(numerator,float)
    index=np.random.default_rng(CFG['bootstrap_seed']).integers(len(a),size=(CFG['bootstrap_draws'],len(a)))
    if denominator is None:return [float(q) for q in np.quantile(a[index].mean(1),[.025,.975])]
    b=np.asarray(denominator,float);den=b[index].mean(1)
    # Ratios are means of paired shifts, not unstable averages of individual ratios.
    stable=np.min(abs(den))>1e-8 and np.all(np.sign(den)==np.sign(b.mean()))
    ratio=a[index].mean(1)/den
    return dict(ratio_of_means=float(a.mean()/b.mean()),ci95=[float(q) for q in np.quantile(ratio,[.025,.975])],
                denominator_mean=float(b.mean()),denominator_ci95=intervals(b),all_bootstrap_denominators_same_sign=bool(stable),
                interpretation='Conditional participant bootstrap; ratio unstable if denominator can approach or cross zero')

def affine_rows():
    rows=[]
    for task in CFG['tasks']:
        features={};labels={}
        for person in PART['development']+PART['evaluation']:
            name=stem(task,person);labels[person]=np.load(OLD/f'{name}_identity.npz')['label']
            features[person]=feature(np.load(CACHE/f'{name}_affine_recipe_filter_parent_input.npy'))
        sc,clf=fit_decoder(np.concatenate([features[s] for s in PART['development']]),np.concatenate([labels[s] for s in PART['development']]))
        np.savez(OUT/f'{task}_affine_recipe_filter_adapted_decoder.npz',scale_mean=sc.mean_,scale=sc.scale_,coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
        for person in PART['development']+PART['evaluation']:
            name=stem(task,person);y=labels[person]
            for condition in (['parent_input'] if person in PART['development'] else ['parent_input','contaminated_input','output_reference']):
                if condition=='output_reference':
                    original=np.load(CACHE/f'{name}_affine_recipe_filter_parent_input.npy');z=baseline(original-original.mean(1,keepdims=True)).astype(np.float32)
                    np.save(CACHE/f'{name}_affine_recipe_filter_{condition}.npy',z)
                else:z=np.load(CACHE/f'{name}_affine_recipe_filter_{condition}.npy')
                f=feature(z)
                rows.append(dict(architecture='affine',target='recipe_filter',task=task,subject=person,condition=condition,
                    split='development' if person in PART['development'] else 'evaluation',amplitude_uv=voltage(z,y,task)[0],
                    frozen_ba=float(balanced_accuracy_score(y,frozen_prediction(task,f))),adapted_ba=float(balanced_accuracy_score(y,clf.predict(sc.transform(f))))))
    return rows

def main():
    cnn=json.loads((OUT/'filter_cnn_assessment.json').read_text())
    earlier=json.loads((ROOT/'results/nonlinear_revision/matched_assessment.json').read_text())
    rows=earlier['rows']+cnn['rows']+affine_rows()
    historical=json.loads((ROOT/'results/causal_revision/erp_assessment.json').read_text())
    for r in historical['rows']:
        if r['method'].startswith('target_') and r['subject'] in PART['development']+PART['evaluation']:
            rows.append(dict(r,architecture='affine',target=r['method'][7:],condition='parent_input'))
    # Previously saved affine deployment controls, scored on exactly the same inputs.
    for task in CFG['tasks']:
        for person in PART['evaluation']:
            name=stem(task,person);y=np.load(OLD/f'{name}_identity.npz')['label']
            for target in ['parent','recipe_brain_argmax','recipe_artifact80']:
                model=np.load(ROOT/f'results/causal_revision/{task}_target_intervention.npz')
                a=np.load(ROOT/f'data/causal_revision/{name}_intervention_contaminated_inputs.npy');flat=a.reshape(-1,512).astype(float);sd=flat.std(1,keepdims=True)
                contaminated=baseline(((flat/sd@model[target+'_matrix']+model[target+'_intercept'])*sd).reshape(a.shape)).astype(np.float32)
                primary=np.load(ROOT/f'data/causal_revision/{name}_target_{target}.npy')
                reref=baseline(primary-primary.mean(1,keepdims=True)).astype(np.float32)
                for condition,z in [('contaminated_input',contaminated),('output_reference',reref)]:
                    np.save(CACHE/f'{name}_affine_{target}_{condition}.npy',z)
                    rows.append(dict(architecture='affine',target=target,task=task,subject=person,condition=condition,split='evaluation',
                        amplitude_uv=voltage(z,y,task)[0],frozen_ba=float(balanced_accuracy_score(y,frozen_prediction(task,feature(z))))))
    lookup={(r['architecture'],r['target'],r['task'],r['subject'],r['condition']):r for r in rows}
    teachers={};source_hashes={}
    for task in CFG['tasks']:
        for person in PART['evaluation']:
            name=stem(task,person);y=np.load(OLD/f'{name}_identity.npz')['label']
            for target in CFG['targets']:
                path=(OLD if target=='parent' else ROOT/'data/causal_revision')/f'{name}_{target}.npy'
                teachers[task,person,target]=voltage(np.load(path),y,task)[0];source_hashes[str(path.relative_to(ROOT))]=sha256(path)
    summary=[];comparisons=[];transmission=[]
    edges=[('parent','recipe_filter'),('recipe_filter','recipe_brain_argmax'),('recipe_filter','recipe_artifact80'),('parent','recipe_brain_argmax'),('parent','recipe_artifact80')]
    for architecture,task,condition in itertools.product(['affine',*CFG['architectures']],CFG['tasks'],['parent_input','contaminated_input','output_reference']):
        for target in CFG['targets']:
            rr=[lookup[architecture,target,task,s,condition] for s in PART['evaluation']]
            summary.append(dict(architecture=architecture,target=target,task=task,condition=condition,
                mean_amplitude_uv=float(np.mean([r['amplitude_uv'] for r in rr])),mean_frozen_ba=float(np.mean([r['frozen_ba'] for r in rr])),
                mean_adapted_ba=float(np.mean([r['adapted_ba'] for r in rr])) if all('adapted_ba' in r for r in rr) else None))
        for first,second in edges:
            output=np.array([lookup[architecture,second,task,s,condition]['amplitude_uv']-lookup[architecture,first,task,s,condition]['amplitude_uv'] for s in PART['evaluation']])
            direct=np.array([teachers[task,s,second]-teachers[task,s,first] for s in PART['evaluation']]);residual=output-direct
            ba=np.array([lookup[architecture,second,task,s,condition]['frozen_ba']-lookup[architecture,first,task,s,condition]['frozen_ba'] for s in PART['evaluation']])
            comparisons.append(dict(architecture=architecture,task=task,condition=condition,from_target=first,to_target=second,
                output_shift_uv=float(output.mean()),output_shift_ci95=intervals(output),direct_reference_shift_uv=float(direct.mean()),
                direct_reference_ci95=intervals(direct),residual_shift_uv=float(residual.mean()),residual_ci95=intervals(residual),
                frozen_ba_shift=float(ba.mean()),frozen_ba_ci95=intervals(ba),transfer=intervals(output,direct)))
            for person,d,l,e in zip(PART['evaluation'],direct,output,residual):
                transmission.append(dict(architecture=architecture,task=task,condition=condition,subject=person,from_target=first,to_target=second,
                    direct_reference_shift_uv=float(d),learned_output_shift_uv=float(l),residual_shift_uv=float(e),identity_discrepancy_uv=float(l-d-e)))
    shares=[]
    for architecture in ['affine',*CFG['architectures']]:
        for second in ['recipe_brain_argmax','recipe_artifact80']:
            filtering=np.array([lookup[architecture,'recipe_filter','P3',s,'parent_input']['amplitude_uv']-lookup[architecture,'parent','P3',s,'parent_input']['amplitude_uv'] for s in PART['evaluation']])
            total=np.array([lookup[architecture,second,'P3',s,'parent_input']['amplitude_uv']-lookup[architecture,'parent','P3',s,'parent_input']['amplitude_uv'] for s in PART['evaluation']])
            shares.append(dict(architecture=architecture,to_target=second,filtering_share_of_learned_shift=intervals(filtering,total)))
    seed_rows=earlier['individual_seeds']+cnn['individual_seeds'];seed_pairs=[]
    for architecture,task,seed in itertools.product(CFG['architectures'],CFG['tasks'],CFG['seeds']):
        by={(r['target'],r['subject']):r for r in seed_rows if (r['architecture'],r['task'],r['seed'],r['split'])==(architecture,task,seed,'evaluation')}
        for first,second in edges:
            diff=[by[second,s]['amplitude_uv']-by[first,s]['amplitude_uv'] for s in PART['evaluation']]
            seed_pairs.append(dict(architecture=architecture,task=task,seed=seed,from_target=first,to_target=second,mean_shift_uv=float(np.mean(diff)),ci95=intervals(diff)))
    # Complete four-by-four loss matrix: rescore all fixed validation predictions.
    design=np.load(OUT/'matched_design.npz');cross_losses=[]
    for architecture in CFG['architectures']:
        for target in CFG['targets']:
            directory=OUT if target=='recipe_filter' else ROOT/'results/nonlinear_revision'
            p=np.mean([np.load(directory/f'{architecture}_{seed}_{target}_validation_predictions.npy').astype(float) for seed in CFG['seeds']],axis=0)
            for scored in CFG['targets']:cross_losses.append(dict(architecture=architecture,training_target=target,scoring_target=scored,ensemble_mse=float(np.mean((p-design['validation_'+scored])**2))))
    write_json(OUT/'four_arm_assessment.json',dict(rows=rows,summary=summary,comparisons=comparisons,participant_decompositions=transmission,
        filter_shares=shares,individual_seeds=seed_rows,seed_pairs=seed_pairs,ensemble_cross_target_losses=cross_losses,
        teacher_endpoint_rows=[dict(task=t,subject=s,target=a,amplitude_uv=v) for (t,s,a),v in teachers.items()],
        source_sha256=source_hashes,configuration_sha256=sha256(ROOT/'filter_revision_config.json'),code_sha256=sha256(__file__),
        scope=CFG['scope'],bootstrap='Paired participants; conditional on fitted operators and inspected cohort; exploratory intervals, no multiplicity correction'))

if __name__=='__main__':main()
