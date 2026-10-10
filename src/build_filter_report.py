"""Consolidated fourth-arm evidence and submission-ready text supplement."""
import json,csv,shutil
from common import ROOT,sha256,write_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_nonlinear_report import table,interval
from verify_nonlinear_revision import manual_endpoint

OUT=ROOT/'results/filter_revision';FIG=ROOT/'figures/filter_revision'
LABEL={'parent':'Parent','recipe_filter':'Filter-only','recipe_brain_argmax':'Brain-only','recipe_artifact80':'Conservative'}
ARCH={'affine':'Affine','simple_cnn':'Simple CNN','complex_cnn':'Residual CNN'}

def waveform_data():
    part=json.loads((ROOT/'causal_revision_config.json').read_text());waves={}
    for method in ['parent','icunet','femto','femto_smooth']:
        curves=[]
        for person in part['evaluation']:
            name=f'P3_{person:03d}';folder='tim_revision' if method in ['icunet','femto_smooth'] else 'erp_validation'
            x=np.load(ROOT/f'data/{folder}/{name}_{method}.npy')
            labels=np.load(ROOT/f'data/erp_validation/{name}_identity.npz')['label']
            curves.append(manual_endpoint(x,labels,'P3')[1])
        waves[method]=np.mean(curves,axis=0)
    return waves


def main():
    FIG.mkdir(parents=True,exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
    a=json.loads((OUT/'four_arm_assessment.json').read_text());s=json.loads((OUT/'passband_sensitivity.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert v['status']=='passed'
    cfg=json.loads((ROOT/'filter_revision_config.json').read_text());fits=[]
    for arch in cfg['architectures']:
        for seed in cfg['seeds']:
            for target in cfg['targets']:
                directory=OUT if target=='recipe_filter' else ROOT/'results/nonlinear_revision'
                fits.append(json.loads((directory/f'{arch}_{seed}_{target}.json').read_text()))
    parts=['# Fourth supervisory arm and passband sensitivity\n\nJeet Bandhu Lahiri and Siddharth Panwar\n',
        '## D1. Protocol, pairing and scope\n\nThe fourth arm was added after the earlier three-arm outcomes, reviewer feedback and the entire cohort had been inspected. `filter_revision_config.json` was saved before the new fits and sensitivity outcomes. The earlier configurations, code and all 18 checkpoints are preserved. This is exploratory, not a preregistered or fresh-cohort replication. Only six added CNN fits and two added affine fits are required: the original input design and earlier targets are reproduced bit-identically from the original trial/channel/EOG draws. Within architecture and seed, the filter arm matches the original initialized state, all 50 minibatch permutations, dropout seed, RMSprop settings, input-derived scaling and epoch 50 budget. Validation does not select checkpoints. The added labels are the 1–80Hz/notch/30Hz-analysis filter stage without component removal. All ten development participants remain shared between training and diagnostic validation; no claim of participant-independent internal validation is made.\n\nThe affine fit uses all20000/4000 original rows per task and the same regularization; the pooled CNN design uses the same first10000/2000 rows per task as earlier. The same ten evaluation participants, unchanged trial identities, parent inputs, test-pool EOG mixtures and output-reference control are used. Adapted decoders use the same ten development participants. The recipe is a declared automated instantiation, not the original released benchmark labels.\n\nA shared-archive preparation collision on the first parallel launch stopped the residual worker before any fit; the completed archive was checked and the worker restarted. No data, fit settings or outcome-dependent selections changed.\n',
        '## D2. Four-arm endpoints\n']
    parts.append(table(['Task','Learner','Target','Control','Mean voltage(µV)','Frozen BA(%)','Adapted BA(%)'],
        [[r['task'],ARCH[r['architecture']],LABEL[r['target']],r['condition'],f"{r['mean_amplitude_uv']:.6f}",f"{100*r['mean_frozen_ba']:.4f}",
          'not rescored' if r['mean_adapted_ba'] is None else f"{100*r['mean_adapted_ba']:.4f}"] for r in a['summary']]))
    parts.append('\n## D3. Stage differences and transfer fractions\n\nFor each participant, direct reference shift D=ℓ(T₁−T₀), learned shift L=ℓ(F₁−F₀), and residual E=L−D. The transfer fraction is mean(L)/mean(D), **not** mean(L/D). All paired participant data are resampled together using10000 draws, seed 94810. Intervals are descriptive and conditional on fitted operators and the inspected cohort; there is no multiplicity correction. Near-zero or crossing reference-shift denominators make a ratio unstable; the sign diagnostic and denominator interval are retained in the machine-readable record. Residuals can offset or amplify shifts; the ratio need not lie between0 and1 in general. Parent-supervised attenuation is a separate learner/deployment term and must not be attributed to the changed targets. Main-text total recipe-versus-parent intervals retain the original bootstrap realizations in Parts B and C (seed 94709); added stage and ratio intervals use the separately frozen seed 94810. Recomputed versions of the earlier contrasts are recorded here, rather than relabeled as a new replication.\n')
    parts.append(table(['Task','Learner','Control','From','To','Direct(µV)','Learned(µV)','Learned95%','Residual(µV)','Transfer(%)','Transfer95%(%)','Stable denominator'],
        [[r['task'],ARCH[r['architecture']],r['condition'],LABEL[r['from_target']],LABEL[r['to_target']],f"{r['direct_reference_shift_uv']:+.6f}",f"{r['output_shift_uv']:+.6f}",interval(r['output_shift_ci95']),f"{r['residual_shift_uv']:+.6f}",
          f"{100*r['transfer']['ratio_of_means']:.3f}",interval(np.array(r['transfer']['ci95'])*100),r['transfer']['all_bootstrap_denominators_same_sign']] for r in a['comparisons']]))
    parts.append('\nThe share attributable to the added filter-only supervision is the parent-to-filter learned shift divided by the parent-to-recipe learned shift. It measures a computational decomposition under this intervention, not the neural fraction of either target.\n')
    parts.append(table(['Learner','Recipe target','Filtering share(%)','95% descriptive interval(%)'],
        [[ARCH[r['architecture']],LABEL[r['to_target']],f"{100*r['filtering_share_of_learned_shift']['ratio_of_means']:.3f}",interval(np.array(r['filtering_share_of_learned_shift']['ci95'])*100)] for r in a['filter_shares']]))
    parts.append('\n## D4. Individual paired seeds and cross-target losses\n')
    parts.append(table(['Task','Learner','Seed','From','To','Mean shift(µV)','95% participant interval'],
        [[r['task'],ARCH[r['architecture']],r['seed'],LABEL[r['from_target']],LABEL[r['to_target']],f"{r['mean_shift_uv']:+.6f}",interval(r['ci95'])] for r in a['seed_pairs']]))
    for arch in cfg['architectures']:
        by={(r['training_target'],r['scoring_target']):r['ensemble_mse'] for r in a['ensemble_cross_target_losses'] if r['architecture']==arch}
        parts.append('\n### '+ARCH[arch]+' ensemble diagnostic loss matrix\n\n'+table(['Supervision/scoring']+[LABEL[t] for t in cfg['targets']],[[LABEL[t]]+[f'{by[t,u]:.7f}' for u in cfg['targets']] for t in cfg['targets']]))
    parts.append('\nAll24 CNN checkpoints use final epoch 50. Full curves and model reports retain each train/validation value. The following end-of-budget values are diagnostics, not proof of convergence.\n')
    parts.append(table(['Learner','Seed','Target','Train MSE at50','Validation MSE at50','Last-five validation range'],
        [[ARCH[r['architecture']],r['seed'],LABEL[r['target']],f"{r['history'][-1]['train_mse']:.7f}",f"{r['history'][-1]['validation_mse']:.7f}",
          f"{min(q['validation_mse'] for q in r['history'][-5:]):.7f}–{max(q['validation_mse'] for q in r['history'][-5:]):.7f}"] for r in fits]))
    parts.append('\n## D5. 1–100Hz construction sensitivity\n\nAll80 recordings were refitted with the upper recipe passband changed to100Hz, following the documented MNE-ICLabel input range. Scalp-average reference, extended infomax29-component ICA, seed94709 + subject, every-fourth-sample fit,500-iteration limit, notch60Hz/Q30, two automated selection policies, final30Hz analysis lowpass, original epochs and baseline are unchanged. These are recipe endpoints, **not retraining** on100Hz targets. The primary1–80Hz results remain primary. Thirty versus64 channels and omitted expert selection remain limitations; meeting the passband specification does not validate ICLabel probabilities or cortical purity.\n\nPrimary technical source: https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html (checked 10 October 2026).\n')
    parts.append(table(['Task','Policy','People','n','1–80Hz mean(µV)','1–100Hz mean(µV)','Change(µV)','95% interval'],
        [[r['task'],LABEL[r['method']],r['split'],r['n_people'],f"{r['mean_amplitude_80_uv']:.6f}",f"{r['mean_amplitude_100_uv']:.6f}",f"{r['mean_difference_uv']:+.6f}",interval(r['ci95'])] for r in s['summary']]))
    parts.append(f"\nICA iteration-limit records: primary{s['iteration_limit_80']}/80, sensitivity{s['iteration_limit_100']}/80. Complete probabilities, selected components, warning messages and reconstruction checks are retained per record.\n\n## D6. Independent verification and executable order\n")
    parts.append(table(['Check','Outcome'],[[k,z] for k,z in v.items()]))
    parts.append('\n```sh\npython src/filter_design.py\npython src/filter_cnn.py\npython src/filter_cnn_assessment.py\npython src/filter_assessment.py\npython src/reference_passband100.py\npython src/passband_assessment.py\npython src/verify_filter_revision.py\npython src/build_filter_report.py\n```\n\nThe earlier source preparation, recipe and paired-CNN stages are prerequisites. No participant was newly recruited; neural truth, released-label bias and prospective ICA coverage remain unidentified.\n')
    text='\n'.join(parts);(ROOT/'reports/filter_revision_supplement.md').write_text(text)
    master=ROOT/'reports/TIM_supplement.md'
    if master.exists():
        old=master.read_text().split('\n\n# Part D.')[0]
        master.write_text(old+'\n\n# Part D. Filter-only supervision and passband sensitivity\n\n'+text)
        destination=ROOT/'submission/supplementary';destination.mkdir(parents=True,exist_ok=True)
        # TXT is an IEEE-recommended supplementary text format; do not create a separate paper PDF.
        (destination/'TIM_supplementary_material.txt').write_text(master.read_text())
        readme=('SUPPLEMENTARY MATERIAL\nElectroencephalography Artifact Removal: Limits of Validation Against Constructed References\nJeet Bandhu Lahiri and Siddharth Panwar\n\n'
                'Submit TIM_supplementary_material.txt as a separate supplementary file WITH the manuscript for peer review. This local package is prepared, not submitted. '
                'A repository link does not replace the supplementary submission. PartsA–C preserve earlier protocols; PartD reports the fourth-arm comparison and100Hz sensitivity. '
                'Contains proofs, all cross-target loss matrices, seed tables, calibration and provenance. UTF-8 plain text; no special software required. '
                'Recommended formats and separate-file submission: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/\n\n'
                'Code: https://github.com/jeetblahiri/artifact-eeg\nContact: d23146@students.iitmandi.ac.in; siddharthpanwar@iitmandi.ac.in\n')
        (destination/'README.txt').write_text(readme)
    # Four-arm actual P3 means, matching the inline main-paper figure.
    means={(r['architecture'],r['target']):r['mean_amplitude_uv'] for r in a['summary'] if r['task']=='P3' and r['condition']=='parent_input'}
    direct={t:np.mean([r['amplitude_uv'] for r in a['teacher_endpoint_rows'] if r['task']=='P3' and r['target']==t]) for t in cfg['targets']}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(5.6,3.4));x=np.arange(4)
    ax.plot(x,[direct[t] for t in cfg['targets']],color='#ad6e22',marker='s',label='Reference')
    for arch,color,marker in [('affine','#353b42','o'),('simple_cnn','#235f8c','^'),('complex_cnn','#9b415a','d')]:
        ax.plot(x,[means[arch,t] for t in cfg['targets']],color=color,marker=marker,label=ARCH[arch])
    ax.set_xticks(x,[LABEL[t] for t in cfg['targets']]);ax.set_ylabel('Mean P3 contrast (µV)');ax.set_ylim(bottom=0);ax.legend(frameon=False,ncol=2);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'four_arm_p3_transmission.{ext}',dpi=240)
    plt.close(fig)
    # Restore the independent ERP-waveform comparison as an inspectable source-backed figure.
    waves=waveform_data();times=np.arange(512)/256-.5
    fig,ax=plt.subplots(figsize=(5.6,3.4));ax.axvspan(.3,.6,color='#edf1f5')
    for method,name,color,style in [('parent','Parent','#353b42','-'),('icunet','IC-U-Net','#235f8c','-'),('femto','FemtoEOGClean','#9b415a','-'),('femto_smooth','Smooth guard','#ad6e22','--')]:
        ax.plot(times,waves[method],label=name,color=color,ls=style)
    ax.set_xlim(0,.9);ax.set_xlabel('Time after stimulus (s)');ax.set_ylabel('Pz target−nontarget (µV)');ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01));fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'p3_waveform_restored.{ext}',dpi=240)
    plt.close(fig)
    primary=[r for r in a['comparisons'] if r['task']=='P3' and r['condition']=='parent_input']
    response=['# Response: filter-only arm, transfer fraction, and restored figures\n',
        'All requested new fits have been executed: six additional fixed-epoch CNN fits and two affine fits. The earlier18 CNN fits are unchanged. Original input arrays and earlier labels are bit-identical; initialized-state and batch-sequence hashes match across each four-arm group.\n',
        'Filtering accounts for most of the recipe-associated learned P3 shift; this is stated explicitly. The incremental filter-to-ICA contrast is reported separately. The contribution is supervision transmitting reference choices into a deployed digital operation, not discovery of high-pass distortion or proof that removed voltage is neural.\n']
    response.append(table(['Learner','From','To','Direct shift','Learned shift','Learned95%','Residual','Transfer(%)','Transfer95%(%)'],
        [[ARCH[r['architecture']],LABEL[r['from_target']],LABEL[r['to_target']],f"{r['direct_reference_shift_uv']:+.4f}",f"{r['output_shift_uv']:+.4f}",interval(r['output_shift_ci95']),f"{r['residual_shift_uv']:+.4f}",f"{100*r['transfer']['ratio_of_means']:.2f}",interval(np.array(r['transfer']['ci95'])*100)] for r in primary]))
    response.extend(['\nTwo figures have been restored/updated: an independent P3-waveform comparison and the direct-reference versus four-arm learned-output comparison. Benchmark-trained CNN rows carry a confounding footnote. Exploratory history remains in one main limitations paragraph. Complete numerical and mathematical records are consolidated in the supplementary TXT prepared for separate submission; the paper has not been submitted.\n',
        'The100Hz sensitivity refits all 80 records. It does not recreate undocumented original expert choices, prove released-label purity, or resolve the pooled ICA5/10 coverage diagnostic.\n',
        table(['Task','Policy','Primary mean','100Hz mean','Change95%'],[[r['task'],LABEL[r['method']],f"{r['mean_amplitude_80_uv']:.4f}",f"{r['mean_amplitude_100_uv']:.4f}",interval(r['ci95'])] for r in s['summary'] if r['split']=='evaluation']),
        '\nIndependent numerical verification: '+str(v['ensemble_records_rescored'])+' ensemble records, '+str(v['four_arm_comparisons_checked'])+' four-arm differences, '+str(v['transmission_identities_checked'])+' decompositions, six fresh checkpoint runs, and240 sensitivity endpoints.\n'])
    (ROOT/'reports/reviewer_response_filter_revision.md').write_text('\n'.join(response))
    write_json(OUT/'report_provenance.json',dict(code_sha256=sha256(__file__),assessment_sha256=sha256(OUT/'four_arm_assessment.json'),
        sensitivity_sha256=sha256(OUT/'passband_sensitivity.json'),verification_sha256=sha256(OUT/'verification.json'),report_sha256=sha256(ROOT/'reports/filter_revision_supplement.md'),
        figures=[str(p.relative_to(ROOT)) for p in sorted(FIG.glob('*'))],formal_supplement_prepared=master.exists(),submission_performed=False))

if __name__=='__main__':main()
