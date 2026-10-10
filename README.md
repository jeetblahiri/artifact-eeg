# EEG benchmark validity and measurement preservation

Analysis code by Jeet Bandhu Lahiri and Siddharth Panwar, Indian Institute of Technology Mandi.

This repository tests a specific inference: recovery of a constructed EEG reference does not, by itself, establish preservation of neural activity. It implements reference-bias sensitivity bounds, a known-source counterexample, synthetic reconstruction comparisons, and independent ERP measurement assessments. It does not measure actual EEGdenoiseNet target impurity or establish that artifact correction is always harmful.

The release contains executable code, numerical protocols, scientific contract tests, public-source checksum metadata, and the frozen compact baseline's tensor-only checkpoint. Manuscript files, raw recordings, generated signal arrays, and private research reports are excluded.

## Paired CNN supervision and clean-input checks

`nonlinear_revision_config.json` records the follow-up settings before the new fits. Both published CNN architectures are retrained on identical development-only ERP/EOG mixtures with parent, brain-only, or conservative recipe targets. Within each architecture and seed, initialized state, minibatch order, dropout seed, optimizer, input-derived scaling and **fixed final epoch 50** are shared; validation is diagnostic, without target-specific checkpoint selection. Three seeds give 18 fits. Pooled N170/P3 designs contain 20,000 training and 4,000 diagnostic validation rows. Fit/validation trial indices and EOG windows are separated, while development participants are shared; neighboring raw epochs may overlap. Mixture SNR uses centered signal power (variance) relative to added, zero-mean EOG variance; full-window means are retained, so this is not a total-RMS-power ratio when those means are nonzero.

The original EEGdenoiseNet-trained CNNs are also checked on all 903 held-out EEG references **without added artifact**, and on ERP parent inputs with alternative centering, development-fitted channel scales and overlap-averaged genuine recording contexts. An identity screen is descriptive, not a requirement that an MMSE denoiser reproduce its input or proof that the references are pure neural voltage.

After the preceding public-data caches, historical CNNs and affine designs are prepared:

```sh
python src/cnn_sanity.py
python src/matched_cnn.py
python src/matched_cnn_assessment.py
python src/verify_nonlinear_revision.py
python src/build_nonlinear_report.py
```

`matched_cnn_assessment.py --wait-for-models` can wait for complete architecture groups while training continues. It uses the same ten development and ten evaluation participants, frozen parent decoders, separately development-fitted adapted decoders, equal seed output averages, the exact saved EOG-test mixtures, and output-average-reference controls. Individual-seed ROI waveforms/features and all physical ensemble arrays are saved under ignored output directories. ERP-trained models are excluded from all-participant cross-fitting. No validation or ERP result is used to select their checkpoints.

The independent checker regenerates initial-state and minibatch hashes, verifies fixed-duration pairing and source separation, rescores voltage/decoder endpoints and target-loss matrices, checks the exact reference/fit-residual contrast decomposition, and independently reruns every new checkpoint on four evaluation trials. The generated supplement and plots expose all seeds, histories and controls. These establish target-definition effects conditional on the fixed learning procedure and two automated constructions, not impurity of released labels or the cause of historical CNN failure. All analyses use an already inspected cohort and remain exploratory. Raw data, generated reports, large retrained weights and manuscript files are excluded from this code-only release.

## Direct reference and controlled-target follow-up

`causal_revision_config.json` records the new exploratory experiments requested after the preceding review. These add the published EEGdenoiseNet simple CNN and complex residual CNN architectures, translated from pinned official source and retrained on source-isolated EOG mixtures, without ERP tuning. Three seeds, 50 epochs, RMSprop 5e-5, and validation-MSE checkpoint selection are explicit; these are architecture reproductions, not original pretrained weights or claims to replicate the authors' leaderboard. Inference uses equal seed averages and records individual evaluation-seed outputs.

The reported reference-construction recipe is instantiated on all 80 ERP CORE recordings with 1–80-Hz filtering, a 60-Hz notch, extended-infomax ICA, and ICLabel. Brain-argmax retention and conservative artifact-probability ≥0.8 exclusion are separate fixed policies. Filter-only and ICA-only arms distinguish the stages. Original expert decisions and thresholds are not available; this is not a recreation of released clean labels or a cortical-purity test. Thirteen fits reach the 500-iteration limit; their flags and converged-only sensitivity remain visible.

A matched-input affine-ridge experiment changes only training targets: parent voltage versus the two constructed references. ERP development trial indices and public EOG source pools are separated before fitting/validation mixtures. The same ten evaluation people and a matched synthetic-contamination control test the resulting target-definition effect. The parent remains a comparator, not accepted neural truth.

Reproduce after preparing the original and first-revision ERP caches and EEGdenoiseNet mixtures:

```sh
python src/download_benchmark_source.py
python src/benchmark_models.py
python src/reference_recipe.py
python src/target_intervention.py
python src/target_contamination_control.py
python src/coverage_diagnostics.py
python src/causal_assessment.py
python src/causal_crossfit.py
python src/output_reference_control.py
python src/verify_causal_assessment.py
python src/build_causal_figures.py
```

All new means, intervals, individual seeds, cross-target loss matrices and coverage diagnostics are generated under the ignored `results/causal_revision/` folder. Independent rescoring checks 800 new primary records, 60 contaminated records, 320 output-reference records and 2080 probe aggregates; 32 actual model probe responses are independently rerun. All 21 contract tests pass.

`mne-icalabel==0.9.0` is included in the dependency lock. Public benchmark code licensing is reproduced in `THIRD_PARTY_NOTICES.md`. Third-party source/weights, the six newly trained CNN weights, raw data, generated results and manuscript files are excluded from this code release. The previously included compact baseline is retained. Model training uses a local GPU when available; CPU reproduction is substantially slower.

Coverage diagnostics include the original seven-candidate envelope with/without regression, the exact exchangeable-rank distribution for the ICA failure count, 100,000 frozen-score reassignments with ties, and separate development-fit sensitivity. The unusual 5/10 ICA coverage is flagged, not rationalized as a heavy-tail effect or used to claim prospective validation. All-40 cross-fitting excludes the paired-target learners, whose ERP training participants would otherwise enter held-out folds.

## Environment and checks

Run from the repository root. The executed environment used Python 3.12; installed package versions are recorded in `requirements.lock`. Neural training used Apple MPS where available, with CPU fallback. The ERP baseline runs on CPU. Hardware and numerical-library differences can affect fitted outcomes.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
python src/mathematical_controls.py
python src/run_known_source.py
```

The tests check power-SNR, source isolation, model capacities, affine estimation, reference regression, decoding, reference-risk identities, sharp ranking bounds, measurement projection, and conformal coverage. The known-source calculation needs no downloaded recordings. Its source labels are stipulated simulation quantities, not inferred neural ground truth.

## Revised measurement assessment (10/20/10 and all 40)

The current procedure adds an independently published pretrained **IC-U-Net**, a fixed montage-mapping control, reference-construction ranges, two spatial digital-probe directions, bandlimited endpoint guards, and participant-level calibration/robustness checks. It retains favorable denoiser outcomes rather than treating neural correction as uniformly harmful. The paper's physiological claim is limited: reconstruction alone does not validate neural preservation; actual EEGdenoiseNet neural impurity remains unidentified.

Prepare the public ERP recordings and frozen-baseline caches with the historical runner below, then execute the follow-up:

```sh
python src/download_icunet.py
python src/independent_denoiser.py
python src/tim_revision.py
python src/revision_probes.py
python src/smooth_direction_probes.py
python src/crossfit_assessment.py
python src/guard_mechanism_audit.py
python src/verify_tim_revision.py
```

The pinned downloader fetches the architecture and checkpoint from the original IC-U-Net authors, commit `7f4f27dbf79c0909a0993f680209cf24c32f7791`, and verifies SHA-256 before inference. Neither file is bundled here. The scholarly model is [Chuang et al., NeuroImage 263, 119586 (2022)](https://doi.org/10.1016/j.neuroimage.2022.119586). The deployed 2,669,854-parameter model uses four-second genuine recording contexts and global window normalization, with no ERP fine-tuning. A fixed spherical-spline montage map, with exact shared electrodes, is accompanied by mapping-only outputs. This deployment uses the common parent passband; it does not recreate the model's training preprocessing. GPU inference is much faster than CPU on this cohort.

`tim_revision_config.json` preserves 10 development, 20 calibration and 10 evaluation participants. These follow-up choices were made after earlier cohort outcomes were inspected; the study is exploratory, not fresh confirmatory validation. Fitting remains numerically isolated to development people. Seven genuine candidates enter calibration; montage and guard controls are excluded. Finite 90% and 95% rank envelopes use calibration ranks 19 and 20. Operation-specific bounds are provided separately; pooled cross-fitted decoder scores are descriptive and have no exact split-conformal guarantee. All-40 voltage-only calibration applies only to fixed non-ERP-fitted operations and has no fresh same-cohort coverage test.

The numerical checker independently recomputes 1,040 participant/task/method rows and 23,040 probe responses. Delivered float32 feature/scaler/LDA conventions are explicit; voltage endpoints use float64 accumulation. Nonlinear probe means, dispersion and extrema are saved. The unit response of regression with EOG fixed and of endpoint guards is algebraic. Probe geometry, montage interpolation, decoder fitting and source purity are distinct limitations.

After placing the acknowledged EEGdenoiseNet banks and executing the reconstruction runner, run:

```sh
python src/reference_audit.py
python src/build_revision_figures.py
```

The audit scores five explicit alternative passbands/projections and the released target against the **same predictions and inputs**. It computes exact risk-contrast ranges over the six-reference convex hull, plus descriptive crossed-component-window bootstrap intervals. Construction disagreement is not δ, a purified target, or evidence of cortical loss. Window identities, not historical participant identities, are available. Different passbands may imply different measurands. The figure builder expects both reference-audit and ERP outputs; run it after both stages.

## Historical ERP assessment (20/10/10)

The pinned public acquisition is [ERP CORE / NEMAR nm000132 v1.1.1](https://data.nemar.org/nm000132/v1.1.1/), N170 and P3, all 40 participants. The downloader verifies the published checksums for every selected file. The raw download is approximately 5.37 GB; processing and saved outputs require additional space.

```sh
python src/download_erp_core.py
python src/erp_validation.py
python src/verify_erp_validation.py
python src/guard_diagnostics.py
python src/build_erp_figures.py
```

`tim_validation_config.json` fixes the same 20 development, 10 calibration, and 10 evaluation people for both tasks. Development alone fits regression, ICA, scalers, decoders, and screening thresholds. The compact denoiser is frozen throughout. Evaluation compares voltage contrasts, frozen and adapted decoding, EOG-only decoding, known digital EEG-only probe response, retention, and joint parent-relative change calibration. An endpoint guard is a diagnostic projection, not a recommended physiological denoiser.

The minimally processed parent is a common comparator, not clean cortical truth. Digital probes test processing response, not acquisition calibration. Ten calibration participants permit the specified finite 90% envelope but not a finite distribution-free 95% envelope. Engineering tolerances are illustrative. See `PROTOCOL.md` for timing and interpretation.

## Auxiliary reconstruction and motor-imagery experiments

1. Obtain `EEG_all_epochs.npy`, `EOG_all_epochs.npy`, and `EMG_all_epochs.npy` from the [EEGdenoiseNet authors' repository](https://github.com/ncclabsustech/EEGdenoiseNet/tree/master/data) and place them in `data/eegdenoisenet/`. Use the 256-Hz, 512-sample banks identified by `data_sources.json`; another release is not interchangeable. The recorded SHA-256 and Git blob hashes identify the exact public inputs.
2. Obtain MIT-BIH records 100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, and 117 (`.hea` and `.dat`) from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and place them in `data/mit-bih/`. Lead 0 supplies a cardiac proxy, not simultaneous scalp contamination.
3. Obtain A01T–A09T `.gdf` files from [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) and place them in `data/bci_iv_2a/`. The evaluation sessions are not used. These public recordings retain their source terms.

```sh
python src/mixture_study.py
python src/real_task.py
python src/benchmark_task_transfer.py
python src/target_validity.py
python src/teacher_intervention.py
python src/verify_results.py
python src/build_figures.py
python src/build_target_figures.py
```

To reproduce the introductory six-panel synthetic-mixture illustration after placing the three EEGdenoiseNet banks in `data/eegdenoisenet/`, run `python src/build_mixture_example.py`. It uses fixed source rows, verifies their checksums, and adds the same EEG reference separately to ocular and muscle segments at 0 dB power-SNR. It exports PNG/SVG figures, plot coordinates, and source metadata to ignored output directories. Normalized amplitudes are dimensionless; the illustration does not validate neural purity.

The reconstruction analysis deduplicates source content, isolates windows before mixing, selects hyperparameters on validation data, and averages losses across fitted seeds rather than ensembling predictions. ECG sources are split by recording. Released EEGdenoiseNet windows lack complete participant identities; window isolation is not participant isolation. Historical source-person overlap with the BCI cohort cannot be excluded. The transfer and teacher studies are exploratory.

## Provenance and fresh runs

`models/provenance.json` identifies the previously trained 2561-parameter Femto baseline and its original publication. Its original full-object checkpoint was converted to a tensor-only state dictionary; all tensors were checked for exact equality. This baseline is reused for assessment, not presented as a new architecture or ERP-trained model.

`source_provenance.json` records the executed-source and release-source fingerprints. Portability changes relocate inputs and the checkpoint, create fresh figure directories, and let the verifier check newly generated runner fingerprints. Analysis settings and estimators are preserved. Dataset and baseline sources should be acknowledged in reuse.

Runners reuse completed caches and checkpoints. Start a separate checkout with empty generated `data/` subdirectories and `results/` for a fresh replication. Preserve `data/erp_core/manifest.json`. After changing code or configuration, use fresh output directories; do not reuse historical caches. The code records new source/configuration hashes for new runs. No historical outputs are relabeled as results from the adapted release.

Original dataset licenses govern downloaded data. The ERP release's license file and metadata use differing Creative Commons labels; consult both at the source before redistributing recordings or derivatives. This repository redistributes no raw recordings.

## Fourth supervisory arm and reference-transmission decomposition

`filter_revision_config.json` records a reviewer-directed addition made after inspection of the earlier outcomes and cohort. It preserves the earlier three-arm configurations and 18 fixed-epoch CNN fits. Six additional fits add **filter-only supervision** to the same two CNN architectures and three paired seeds. Two affine fits use the exact original task designs and regularization. Raw-source regeneration verifies identical input mixtures and earlier target labels; initialized-state and minibatch hashes verify pairing. All fits use final epoch 50 without outcome-selected checkpoints.

After the automated recipe, affine, and paired-CNN stages above, run the following in order from the repository root:

```sh
python src/filter_design.py
python src/filter_cnn.py
python src/filter_cnn_assessment.py
python src/filter_assessment.py
python src/reference_passband100.py
python src/passband_assessment.py
python src/verify_filter_revision.py
python src/build_filter_report.py
```

The added design is prepared once before training. Use the sequential default runner; concurrent initial design writers can race. Do not change earlier configurations or reuse stale caches after source changes. The CNN design uses the first fixed 10,000/2,000 rows per task from the saved affine designs, pooled equally across tasks. Development participants are shared between fitting and diagnostic validation; this is not participant-independent internal validation.

Outputs distinguish parent-to-filter transmission from the filter-to-component-selection increment, as well as the total parent-to-recipe shift. Transfer fractions are **ratios of paired participant-mean shifts**, with 10,000 paired bootstrap draws and a denominator stability diagnostic; individual ratios with small contrasts are avoided. Parent-supervised attenuation and fit/deployment residuals remain separate. The full four-by-four cross-target loss matrices, seed effects, primary and shared-input controls are generated under ignored `results/filter_revision/`.

The separate 1–100 Hz sensitivity refits both automated reference policies on all 80 ERP recordings, with final 30-Hz endpoint analysis unchanged. It does not retrain the denoisers on the sensitivity labels, reconstruct expert choices, resolve the 30-versus-64-channel limitation, or establish cortical purity. The pooled ICA coverage diagnostic remains unresolved. Reconstruction against constructed targets still does not identify neural error in the original released EEGdenoiseNet labels.

The report runner exports two numerical figures and an inspectable text report. A consolidated journal supplement, when preceding local reports are present, is prepared as a separate supplementary TXT for author submission. No manuscript or supplementary manuscript text is included in this code-only repository.
