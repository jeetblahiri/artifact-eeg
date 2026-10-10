"""Add filter-only labels using exactly the previously saved mixture draws."""
import json
from common import ROOT, sha256, write_json
import numpy as np
import target_intervention as original
from erp_validation import baseline, stem

OUT=ROOT/'results/filter_revision'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original.TC=dict(original.TC,targets=[*original.TC['targets'],'recipe_filter'])
    for task in ['N170','P3']:
        destination=OUT/f'{task}_filter_design.npz'
        if destination.exists():continue
        rng=np.random.default_rng(original.TC['seed'])
        x,y,provenance=original.design(task,original.TC['fit_mixtures'],'train',rng)
        vx,vy,validation_provenance=original.design(task,original.TC['validation_mixtures'],'val',rng)
        previous=np.load(ROOT/f'results/causal_revision/{task}_target_design.npz')
        np.testing.assert_array_equal(x,previous['x']);np.testing.assert_array_equal(vx,previous['vx'])
        for target in original.C['target_intervention']['targets']:
            np.testing.assert_array_equal(y[target],previous[target])
            np.testing.assert_array_equal(vy[target],previous['validation_'+target])
        np.savez_compressed(destination,recipe_filter=y['recipe_filter'],validation_recipe_filter=vy['recipe_filter'])
        xm=x.mean(0);xc=x-xm;gram=xc.T@xc
        ridge=original.TC['ridge_fraction']*np.trace(gram)/512
        ym=y['recipe_filter'].mean(0)
        matrix=np.linalg.solve(gram+ridge*np.eye(512),xc.T@(y['recipe_filter']-ym));intercept=ym-xm@matrix
        np.savez(OUT/f'{task}_filter_affine.npz',matrix=matrix,intercept=intercept)
        losses={t:float(np.mean((vx@matrix+intercept-z)**2)) for t,z in vy.items()}
        cache=ROOT/'data/filter_revision';cache.mkdir(exist_ok=True)
        for subject in original.C['development']+original.C['evaluation']:
            for condition in ['parent_input','contaminated_input']:
                source=(original.OLD/f'{stem(task,subject)}_parent.npy' if condition=='parent_input' else original.CACHE/f'{stem(task,subject)}_intervention_contaminated_inputs.npy')
                if not source.exists():continue
                a=np.load(source);flat=a.reshape(-1,512).astype(float);sd=flat.std(1,keepdims=True)
                z=baseline(((flat/sd@matrix+intercept)*sd).reshape(a.shape)).astype(np.float32)
                np.save(cache/f'{stem(task,subject)}_affine_recipe_filter_{condition}.npy',z)
        write_json(OUT/f'{task}_filter_design.json',dict(task=task,provenance=provenance,validation_provenance=validation_provenance,
            ridge=ridge,cross_target_validation_mse=losses,original_inputs_and_three_labels_bit_identical=True,
            configuration_sha256=sha256(ROOT/'filter_revision_config.json'),code_sha256=sha256(__file__),
            original_design_sha256=sha256(ROOT/f'results/causal_revision/{task}_target_design.npz'),design_sha256=sha256(destination),
            affine_sha256=sha256(OUT/f'{task}_filter_affine.npz')))
        print('Added exact paired filter design and affine fit',task,losses,flush=True)

if __name__=='__main__':main()
