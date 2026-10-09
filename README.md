# EEG benchmark validity and measurement preservation

Analysis code by Jeet Bandhu Lahiri and Siddharth Panwar, Indian Institute of Technology Mandi.

This repository tests a specific inference: recovery of a constructed EEG reference does not, by itself, establish preservation of neural activity. It implements reference-bias sensitivity bounds, a known-source counterexample, synthetic reconstruction comparisons, and independent ERP measurement assessments. It does not measure actual EEGdenoiseNet target impurity or establish that artifact correction is always harmful.

The release contains executable code, numerical protocols, scientific contract tests, public-source checksum metadata, and the frozen compact baseline's tensor-only checkpoint. Manuscript files, raw recordings, generated signal arrays, and private research reports are excluded.

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

## Independent ERP assessment

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
