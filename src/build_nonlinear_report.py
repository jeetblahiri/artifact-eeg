"""Inspectable supplement and plots for the paired-target CNN revision."""
import json
from pathlib import Path
from textwrap import dedent

from common import ROOT, sha256, write_json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/nonlinear_revision'
FIG=ROOT/'figures/nonlinear_revision'
CFG=json.loads((ROOT/'nonlinear_revision_config.json').read_text())
LABEL={'parent':'Parent','recipe_brain_argmax':'Brain-only','recipe_artifact80':'Conservative'}
ARCH={'simple_cnn':'Simple CNN','complex_cnn':'Residual CNN'}
COLORS={'parent':'#353b42','recipe_brain_argmax':'#235f8c','recipe_artifact80':'#9b415a'}


def table(head,rows):
    return '\n'.join(['| '+' | '.join(map(str,head))+' |','|'+'|'.join(['---']*len(head))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def interval(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    sanity=json.loads((OUT/'cnn_sanity.json').read_text())
    fits=json.loads((OUT/'matched_training.json').read_text())['models']
    assessment=json.loads((OUT/'matched_assessment.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='passed'
    parts=['# Supplementary material: Electroencephalography Artifact Removal: Limits of Validation Against Constructed References\n\nJeet Bandhu Lahiri and Siddharth Panwar\n\nPaired CNN supervision and clean-input diagnostics\n',
           'Executed exploratory follow-up, 10 October 2026. This supplements the preceding reference audit and affine intervention; it does not replace or relabel their saved results. All 40 participants and earlier outcomes had already been inspected. The saved `nonlinear_revision_config.json` records the fit settings. Its first public commit follows execution and does not establish prospective specification. No preregistration or fresh-cohort claim is made.\n',
           '## 1. What the controlled comparison identifies\n',
           'Within architecture/seed, three fits share the exact float32 input design, initialized parameters and batch-normalization state, minibatch permutations, dropout seed, RMSprop settings, input-derived target scaling and fixed final epoch. Only the supervisory arrays differ. This identifies the effect of target definition **under this training algorithm and budget** on held-out digital outputs. It is not a claim about an optimizer limit, a population causal effect, cortical source purity or the unique cause of historical CNN deployment failure. Parent voltages are comparators with potentially mixed neural/peripheral content.\n',
           'The same architectures are used as the historical EEGdenoiseNet-trained fits: simple CNN (16,815,552 parameters) and complex residual CNN (8,455,424 parameters), PyTorch translations of pinned public source. These are retrained architecture ports, not the authors\' original checkpoints or published leaderboard replication.\n',
           '## 2. No-added-artifact sanity check\n',
           'All 903 held-out EEGdenoiseNet EEG reference windows are supplied without added artifact, after the same mean-centering and unit-RMS normalization used for their source bank. The references are not accepted cortical truth. A 10% relative waveform RMS criterion is a descriptive engineering identity screen, not physiological equivalence or a required property of an MMSE denoiser.\n']
    fields=['architecture','n_reference_windows','pooled_relative_rms','median_window_relative_rms','mean_correlation','through_origin_gain','fraction_below_10pct_relative_rms']
    parts.append(table(['Architecture','Windows','Pooled relative RMS','Median relative RMS','Mean correlation','Gain','Fraction ≤10%'],
                       [[ARCH[r['architecture']],r['n_reference_windows']]+[f'{r[k]:.4f}' for k in fields[2:]] for r in sanity['benchmark']]))
    parts.append('\nThese fits do not approximately reproduce these inputs under the declared screen. Nonidentity may arise from conditional-mean shrinkage and training/deployment mismatch; it is not proof of physiological harm or impure references.\n')
    parts.append(table(['Architecture','Seed','Relative RMS','Gain'],[[ARCH[r['architecture']],r['seed'],f"{r['pooled_relative_rms']:.4f}",f"{r['through_origin_gain']:.4f}"] for r in sanity['individual_seeds']]))
    parts.append('\n## 3. ERP interface and alignment diagnostics\n\nParent ERP inputs contain no newly added artifact. Every main contrast uses the unchanged ten evaluation participants. Delivered interface uses per-channel window SD, physical restoration and baseline. Two full-trial controls mean-center before inference or replace the local SD with the channel-specific median fitted on development people only. Those are sensitivity checks rather than newly selected deployments.\n\nThe alignment control uses the same eight fixed trial backgrounds (four per class) in actual four-second recording contexts. Five two-second windows start at offsets 0, 128, 256, 384 and 512 samples; restored voltages are overlap-averaged and centrally extracted. No zero padding or fabricated context is used. Background contrasts must be compared with their same eight-trial central-window values, not the full-cohort ERP mean. Independent float32 parent/context caches differ by at most 0.000628 µV after baseline; this numerical tolerance is not physiological equivalence.\n')
    parts.append(table(['Task','Architecture','Mode','Mean contrast (µV)','Mean relative RMS'],[[r['task'],ARCH[r['architecture']],r['mode'],f"{r['mean_contrast_uv']:.4f}",f"{r['mean_relative_rms']:.4f}"] for r in sanity['erp_summary']]))
    parts.append('\nDomain/normalization/window/single-channel effects can coexist. These finite controls do not identify a unique cause of historical near-zero P3 and are not used as causal evidence for the target-definition argument.\n\n## 4. Matched retraining protocol\n\nThe first fixed 10,000 rows per task are taken from the saved affine-intervention training designs and pooled equally, for 20,000 training rows. The first 2,000 validation rows per task give 4,000 diagnostic rows. All training voltages and references come from the same ten development participants; the same trial indices, channels and public EOG draws define the three target conditions. Fit/validation trial indices and EOG source windows are separated, but development participants are shared and neighboring raw epochs may overlap. This is not independent-participant or disjoint-raw-sample validation.\n\nThree paired seeds (17, 29, 43), 50 epochs, batch size 128 and RMSprop (learning rate 5e−5, alpha 0.9, epsilon 1e−7, no weight decay) are fixed. **Epoch 50 is primary for every fit**; validation is diagnostic, without early stopping or selecting a target-specific checkpoint. The three seeds are equally averaged at deployment. Results depend on this finite budget; diagnostic losses and full histories remain available and do not prove convergence.\n')
    parts.append(table(['Architecture','Seed','Target','Epoch','Final train MSE','Final validation MSE','Last-five validation range'],
                       [[ARCH[r['architecture']],r['seed'],LABEL[r['target']],r['selected_epoch'],f"{r['history'][-1]['train_mse']:.6f}",f"{r['history'][-1]['validation_mse']:.6f}",
                         f"{min(q['validation_mse'] for q in r['history'][-5:]):.6f}–{max(q['validation_mse'] for q in r['history'][-5:]):.6f}"] for r in fits]))
    parts.append('\nInitial-state hashes, exact minibatch-order hashes, source-array/checkpoint checksums and final-epoch selection are verified independently in `verification.json`. Individual JSON files retain all 50 train/validation values, not only the last row.\n\n## 5. Held-out endpoints and shared controls\n\nNo calibration participants enter this new training or deployment comparison. Full physical ensemble outputs are saved for ten development and ten evaluation people; individual-seed ROI waves and decoder features are retained. Parent-trained decoders are unchanged from the earlier assessment. Adapted decoders use only the ten development participants processed by the corresponding fit. They assess representational usability, not cortical fidelity. The contamination control uses exactly the previously saved held-out EOG test-pool mixtures at −7/−4/−1/+2 dB for every target and architecture. Here SNR is the ratio of centered signal power (voltage variance) to added EOG variance: coefficient SD(parent)×10^(−SNR/20) multiplies unit-SD, zero-mean EOG. Parent full-window means are retained, so this is not a total-RMS-power ratio when a window has nonzero mean. A separate output-average-reference control acts on all 30 scalp channels and rebaselines.\n')
    parts.append(table(['Task','Architecture','Training target','Input/control','Mean contrast (µV)','Frozen BA (%)','Adapted BA (%)'],
                       [[r['task'],ARCH[r['architecture']],LABEL[r['target']],r['condition'],f"{r['mean_amplitude_uv']:.4f}",f"{100*r['mean_frozen_ba']:.3f}",f"{100*r['mean_adapted_ba']:.3f}"] for r in assessment['summary']]))
    parts.append('\nPaired recipe-supervised minus parent-supervised effects; intervals are descriptive participant bootstraps conditional on the fitted seed ensembles, not seed-as-participant inference or prospective population bounds:\n')
    parts.append(table(['Task','Architecture','Changed target','Input/control','Voltage change (µV)','95% interval','BA change (points)','95% interval'],
                       [[r['task'],ARCH[r['architecture']],LABEL[r['target']],r['condition'],f"{r['amplitude_difference_uv']:+.4f}",interval(r['amplitude_ci95']),f"{100*r['frozen_ba_difference']:+.3f}",interval(np.array(r['frozen_ba_ci95'])*100)] for r in assessment['pairs']]))
    parts.append('\n## 6. Individual paired seeds\n')
    parts.append(table(['Task','Architecture','Target','Seed','Voltage difference (µV)','95% descriptive participant interval'],
                       [[r['task'],ARCH[r['architecture']],LABEL[r['target']],r['seed'],f"{r['mean_amplitude_difference_uv']:+.4f}",interval(r['ci95'])] for r in assessment['seed_pairs']]))
    parts.append('\n## 7. Cross-target loss matrices\n\nRows are supervision targets and columns are scoring targets on the same 4,000 diagnostic validation inputs. These are losses of **equal-seed ensemble predictions**, not averages of individual-seed MSEs. Targets and predictions are in the same input-normalized units; these scores are not neural errors.\n')
    for a in CFG['architectures']:
        by={(r['training_target'],r['scoring_target']):r['ensemble_mse'] for r in assessment['ensemble_cross_target_losses'] if r['architecture']==a}
        parts.append('\n### '+ARCH[a]+'\n\n'+table(['Supervision']+[LABEL[t] for t in CFG['targets']],[[LABEL[t]]+[f'{by[t,u]:.6f}' for u in CFG['targets']] for t in CFG['targets']]))
    parts.append('\n## 8. Endpoint propagation, with residuals retained\n\nFor each participant and linear contrast ℓ, let e_t=F_t−T_t on the identical evaluation arrays. Then\n\n**ℓ(F₁−F₀) = ℓ(T₁−T₀) + ℓ(e₁−e₀).**\n\nAdding/subtracting the references proves this elementary identity for any learner. The direct reference shift, observed learned shift and residual are independently scored from saved physical arrays. The empirical contribution is the controlled transmission test across participants and architectures, not a new algebraic identity. A residual is allowed to offset or amplify the direct shift; ratios are avoided when a reference contrast is small. In the contaminated-input control the corresponding uncontaminated targets remain the intended comparison, so the same decomposition includes contamination/deployment error.\n')
    transmission=[]
    for a in CFG['architectures']:
        for task in CFG['tasks']:
            for t in CFG['targets'][1:]:
                rr=[r for r in assessment['endpoint_transmission'] if (r['architecture'],r['task'],r['target'],r['condition'])==(a,task,t,'parent_input')]
                transmission.append([task,ARCH[a],LABEL[t]]+[f"{np.mean([r[k] for r in rr]):+.4f}" for k in ['direct_reference_shift_uv','learned_output_shift_uv','residual_shift_uv']])
    parts.append(table(['Task','Architecture','Target','Direct shift (µV)','Learned shift (µV)','Residual shift (µV)'],transmission))
    parts.append('\n### Independent mixing in the known-source endpoint\n\nThe earlier known-source control provides a separate logical check (see `src/run_known_source.py`). At alpha=1, T=B*g_b and the constructed artifact is (U+V)*g_u+C*g_a. B is independent of (U,V,C), so these two constructed pools are independent. Their sum is still the same observed Y=B*g_b+(U+V)*g_u+C*g_a. Projection onto g_b recovers T exactly (R_T=0), while the desired source N=B*g_b+U*g_u differs by U*g_u (R_N=1 with unit-RMS modes and U in {−1,+1}). Thus this endpoint does not rely on dependence between the synthetically combined pools. T contains only stipulated neural activity yet omits an intended neural mode: purity alone would not establish completeness. Intermediate alpha values can share U between both pools and need not be independent. These are stipulated source labels and a possibility result, not evidence of real released-label impurity.\n')
    parts.append('\n## 9. Secondary calibration and ICA fit sensitivity\n\nCalibration has been demoted in the manuscript; full historical envelopes remain in `causal_revision_supplement.md` and `tim_revision_supplement.md`. For fixed development procedure and exchangeable participant scores S_i=max_j,k |d_ijk|/τ_k, the new score\'s rank among n+1 scores is uniform (ties conservative). Rank ceil((n+1)(1−alpha)) therefore yields marginal simultaneous change coverage within the fixed family; the order statistic is infinite if that rank exceeds n. It is not cortical-error coverage or conditional coverage for every participant.\n\nThe pooled EOG-guided ICA envelope\'s 5/10 result is unusual under the frozen-score rank model: P(K≤5)=0.00884173, with 100,000 tied-score reassignments giving 0.00857. Four failures are driven by N170 decoding and one by P3 voltage. This ICA is distinct from the per-record ICLabel recipes in the target intervention. Five development-subset refits change mean P3 from 3.02 to 8.09 µV and N170 frozen BA from 53.44% to 60.81%, connecting fit sensitivity to the measured endpoints. This is a practical sensitivity clue, **not a demonstration that exchangeability breaks**: conditional on one frozen operator, instability across other operators does not invalidate the rank guarantee. The refits may change fit/test roles and are not a fresh coverage study. Rare allocation, participant heterogeneity and procedure dependence remain unresolved.\n\n## 10. Verification, remaining limits and execution\n')
    parts.append(table(['Independent check','Result'],[[k,v] for k,v in verification.items() if k not in ['scope','code_sha256','configuration_sha256']]))
    parts.append('\nThe original benchmark\'s preconstruction parents, event identities, simultaneous peripheral recordings and expert decisions are unavailable; neither this audit nor the two automated recipes establish impurity of the released labels. Spectral overlap alone is not a proof that constrained multichannel separation is impossible. No acquisition uncertainty, physical artifact reduction, clinical endpoint or universal sufficiency of minimal preprocessing is validated. Positive IC-U-Net findings and robust MLP benchmark advantages remain in the main study. Prospective validation requires a new cohort and a frozen full procedure.\n\n```sh\npython src/cnn_sanity.py\npython src/matched_cnn.py\npython src/matched_cnn_assessment.py\npython src/verify_nonlinear_revision.py\npython src/build_nonlinear_report.py\n```\n\nThe new fits reuse prior public-data caches and designs; see the code repository for their prerequisites. New large weights, signal arrays and raw recordings are not bundled in the code-only release.\n')
    (ROOT/'reports/nonlinear_revision_supplement.md').write_text('\n'.join(parts))
    if (ROOT/'reports/tim_revision_supplement.md').exists() and (ROOT/'reports/causal_revision_supplement.md').exists():
        first=(ROOT/'reports/tim_revision_supplement.md').read_text().replace('The six reviewer priorities','The six revision priorities')
        first=first[first.index('## S1.'):]
        # Auxiliary ECG/BCI studies remain archived separately, outside the core
        # consolidated supplement and manuscript claim.
        first=first[:first.index('## S8.')]+first[first.index('## S9.'):]
        prior=(ROOT/'reports/causal_revision_supplement.md').read_text()
        prior=prior[prior.index('## 1.'):]
        current='\n'.join(parts);current=current[current.index('## 1.'):]
        consolidated=dedent('''
    # Supplementary material for Electroencephalography Artifact Removal: Limits of Validation Against Constructed References

    Jeet Bandhu Lahiri and Siddharth Panwar, School of Computing and Electrical Engineering, Indian Institute of Technology Mandi.

    This consolidated numerical record identifies three successive exploratory protocols. Original outputs and configurations are preserved; historical observations are not relabeled as new or prospective validation. Part C contains the current paired nonlinear supervision comparison. Parts A and B retain the reference audit, known-source derivation, independent ERP assessment and preceding controls. Auxiliary ECG and motor-imagery studies are outside the core manuscript and remain in their separate archived reports. The main manuscript states the essential claims and results without requiring this full record.

    # Part A. Reference audit, digital measurement assessment and mathematical detail

    ''')+first+'\n\n# Part B. Automated reference construction and affine supervision intervention\n\n'+prior+'\n\n# Part C. Paired CNN supervision and clean-input diagnostics\n\n'+current
        (ROOT/'reports/TIM_supplement.md').write_text(consolidated)
    response=['# Response to the remaining reviewer concerns\n',
              '## 1. Separate historical CNN deployment from the causal target claim\n',
              'The abstract no longer juxtaposes synthetic MSE with near-zero P3. All 903 held-out EEG reference windows were supplied without added artifact; the original simple/residual CNN ensemble relative RMS errors are 32.5%/30.7%, with only 5.8%/3.7% passing a declared 10% identity screen. Per-window centering, development-fitted fixed scales and genuine-context window overlap are reported separately. Neither reference purity nor a unique cause of historical failure is inferred.\n',
              'Both architectures were retrained on the same development-only inputs with three supervisory targets and three paired seeds (18 fits). Input arrays, initialized state and minibatch hashes match within each pair; the optimizer, dropout seed, scaling and final epoch50 are fixed. Parent/recipe effects, individual seeds, contaminated inputs, output reference controls and complete loss curves are independently checked.\n',
              '## 2. State what the intervention adds beyond filtering\n',
              'The ridge identity is explicitly described as standard. The contribution is the measured transfer of a target choice through a fixed training procedure to participants excluded from training, extended to nonlinear CNNs. A new linear-endpoint decomposition separates direct construction changes from fit/deployment residuals without assuming cortical truth. Filter-only and ICA-only controls distinguish passband and component effects. The abstract and methods explicitly identify two automated recipe instantiations, not the released labels. Parent-supervised attenuation remains visible.\n',
              '## 3. Reduce density and preserve the cleanest theoretical result\n',
              'The abstract has three connected claims: target-conditional validation, scoped controlled evidence, and independent measurement assessment. The main text retains the unchanged-observation known-source result R_T=0/R_N=1. Elementary algebra is not represented as a novel theorem. The exploratory history and archived execution settings remain disclosed; no preregistration, fresh cohort or confirmatory population claim is made.\n',
              '## 4. Demote calibration and connect ICA sensitivity carefully\n',
              'Calibration is a secondary short methods/result diagnostic; the main calibration table is replaced by paired CNN supervision results. Full envelopes and rank proofs remain available in the supplement. The 5/10 ICA result is tied to refit sensitivity (P3 means3.02–8.09µV; N170 BA53.44–60.81%). This does not establish why the inspected split failed: instability across refits does not break exchangeable ranks conditional on a frozen operator. Rare allocation and heterogeneity remain unresolved, and prospective validation is required.\n',
              '## 5. Explain the measurement-journal contribution\n',
              'Three new verified I&M sources discuss ML uncertainty and GUM-based learned measurement models (DOIs10.1109/MIM.2021.9436102,10.1109/MIM.2025.10982089,10.1109/TIM.2025.3643048). The introduction distinguishes their uncertainty question from supervisory-reference validity. The scope cover letter explains the defined digital voltage chain, independent endpoints, public-data necessity and absence of an acquisition-level uncertainty claim.\n',
              '## Remaining scientific boundaries\n',
              'The released EEGdenoiseNet labels still lack independently available cortical truth and preconstruction parents. These experiments cannot identify their actual impurity, prove that all deep denoisers are redundant, or promise future-cohort coverage. They support the narrower central claim that reconstruction alone does not establish neural preservation. No revision can honestly guarantee absence of all reviewer objections or journal acceptance.\n',
              '## Actual paired CNN results\n']
    response.append(table(['Task','Architecture','Target','Input/control','Voltage change(µV)','95% descriptive interval'],
                          [[r['task'],ARCH[r['architecture']],LABEL[r['target']],r['condition'],f"{r['amplitude_difference_uv']:+.4f}",interval(r['amplitude_ci95'])] for r in assessment['pairs']]))
    response.append('\nIndependent verification: '+str(verification['ensemble_records_rescored'])+' ensemble records, '+str(verification['individual_seed_records'])+' seed endpoint records, '+str(verification['transmission_identities_checked'])+' endpoint decompositions, '+str(verification['checkpoints_independently_rerun'])+' directly rerun checkpoints.\n')
    (ROOT/'reports/reviewer_response_2026-10-10.md').write_text('\n'.join(response))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10,5.5),sharex=True)
    for row,a in enumerate(CFG['architectures']):
        for col,seed in enumerate(CFG['seeds']):
            ax=axes[row,col]
            for r in fits:
                if r['architecture']==a and r['seed']==seed:
                    ax.plot([q['epoch'] for q in r['history']],[q['validation_mse'] for q in r['history']],color=COLORS[r['target']],label=LABEL[r['target']])
            ax.set_title(f'{ARCH[a]}, seed {seed}');ax.set_ylim(bottom=0);ax.set_xlabel('Epoch')
            if col==0:ax.set_ylabel('Target-specific validation MSE')
    axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(FIG/f'fixed_budget_learning_curves.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),sharey=True)
    by={(r['architecture'],r['target'],r['task'],r['subject'],r['condition']):r for r in assessment['rows']}
    for ax,a in zip(axes,CFG['architectures']):
        ids=assessment['evaluation_people']
        for s in ids:
            ax.plot(range(3),[by[a,t,'P3',s,'parent_input']['amplitude_uv'] for t in CFG['targets']],color='#b9bfc4',alpha=.6,lw=.8)
        means=[np.mean([by[a,t,'P3',s,'parent_input']['amplitude_uv'] for s in ids]) for t in CFG['targets']]
        ax.plot(range(3),means,color='#235f8c',marker='o',lw=2,label='Ten-person mean')
        ax.set_xticks(range(3),[LABEL[t] for t in CFG['targets']]);ax.set_title(ARCH[a]);ax.set_xlabel('Training target');ax.axhline(0,color='gray',lw=.6)
    axes[0].set_ylabel('P3 contrast (µV)');axes[0].legend(frameon=False);fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(FIG/f'matched_p3_participants.{suffix}',dpi=180)
    plt.close(fig)
    write_json(OUT/'report_provenance.json',dict(report_sha256=sha256(ROOT/'reports/nonlinear_revision_supplement.md'),
               code_sha256=sha256(__file__),figures=[str(p.relative_to(ROOT)) for p in sorted(FIG.glob('*'))],
               input_assessment_sha256=sha256(OUT/'matched_assessment.json'),verification_sha256=sha256(OUT/'verification.json')))
    print('Built nonlinear supplement and saved training/participant plots')


if __name__=='__main__':main()
