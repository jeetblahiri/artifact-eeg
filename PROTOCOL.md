# Revised protocol and inferential scope

The current assessment is a reviewer-directed follow-up dated 9 October 2026. The earlier ERP outcomes had been inspected. The split keeps both tasks together: 10 development participants, 20 calibration, and the same 10 evaluation participants as the original analysis. It does not constitute a fresh confirmatory test. The explicit participant IDs are in `tim_revision_config.json`.

Development alone fits regression and ICA (50,000 continuous samples, 5,000 per person), scalers, and frozen/adapted LDA decoders. The pretrained IC-U-Net and FemtoEOGClean are fixed. All 14,400 planned epochs are retained, including 3,600 primary evaluation trials. The independent-model mapping is geometry-only; missing montage channels require interpolation, with mapping-only outcomes reported. Genuine four-second context is used for IC-U-Net. All native windows are verified not to cross recording boundaries.

The revision's seven calibration candidates are HP 0.5/1/2 Hz, EOG regression, ICA/EOG, IC-U-Net, and FemtoEOGClean. Parent, mapping and guards are controls. Joint scores maximize both tasks and amplitude/frozen-BA changes with illustrative 1-µV and two-point scales. Twenty calibration people permit finite 90% and 95% envelopes. This coverage statement requires a fixed procedure and exchangeable future people; the inspected sample's observed coverage is descriptive. Operation-specific and endpoint-specific bounds do not protect unrestricted method/endpoint selection. Fivefold decoder cross-fitting over all forty has no exact guarantee for pooled quantiles. Voltage-only all-forty calibration concerns fixed non-ERP-fitted operations and has no new same-cohort coverage test.

The localized digital scalp direction was assessed first. A broad geodesic Gaussian direction (0.4-rad width, zero scalp mean, unit ROI mean) was added after seeing strong localized montage attenuation. Both sets retain all 480 background/amplitude responses per operation/task. Nonlinear gains are finite paired responses, not transfer functions. EOG is held fixed, which forces unity for EEG-only regression response. Guard unity is likewise an exact contract. Neither implies cortical purity.

Smooth guards use the minimum-norm correction in a ≤30-Hz orthogonal Fourier subspace. A full-output variant first bandlimits the candidate. The mechanism audit holds the original decoder fixed while changing guard construction; the revised primary decoder and all-participant cross-fitted decoder assess its dependence. A decoding change is not assigned to cortical loss.

The reference audit tests five deterministic alternative constructions on all 4,514 released EEG windows and rescored EOG/EMG test targets. It leaves model fits, inputs, and predictions fixed. Its convex-hull ranking range is exact conditional construction sensitivity. It is a neural-bias interval only under independently justified target-family membership. Butterworth filtering on two-second windows may introduce boundary effects; Fourier projections provide an additional reference family. Crossed source-window bootstrap intervals condition on selected seed fits and cannot make participant-level claims.

The config's inherited text saying eight nonparent teachers was corrected to the seven explicit `calibration_family` candidates; the numerical algorithm used those seven throughout. No numerical choice changed. The local evidence package preserves the initial snapshot and confirms exact equality of the rerun classical operator tensors.

The executed-source/release-source hashes in `source_provenance.json` distinguish portable path changes. Reproduce in a fresh checkout/caches after changes; generated historical arrays are not bundled or relabeled. IC-U-Net third-party source/weights are obtained by a pinned downloader, not redistributed. Manuscript and supplementary narrative files are excluded from this repository.

---

# Historical numerical protocol and chronology

The reconstruction protocol was fixed locally on 8 October 2026 after earlier related results were known. It uses source-window-disjoint 60/20/20 splits, power-SNR −7/−4/−1/+2 dB, 10000/2000/3000 mixtures, three training seeds, two learning rates, and twenty epochs. The ECG proxy uses recording-disjoint splits. The matched reconstruction-to-task transfer was a declared post hoc analysis. Settings are in `config.json`.

The known-source and reference-bias controls were fixed after those outcomes and before their own execution. Settings are in `target_validity_config.json`. Bias radii are hypothetical sensitivity assumptions, not estimates of biological contamination. The known-source generator preserves the observed waveform across source allocations.

The independent ERP protocol was fixed on 9 October 2026 before scoring new ERP EEG outcomes, while previous synthetic and BCI outcomes were known. It is exploratory and was not externally registered. All forty participants and both N170/P3 tasks were selected as a complete cohort. Participants stay in the same development/calibration/evaluation split across tasks. `tim_validation_config.json` preserves the executed settings and chronology.

ERP processing uses scalp-average reference, resampling to 256 Hz, continuous forward–backward 0.1–30 Hz parent filtering, −0.5 to 1.5 s epochs, and a −0.2 to 0 s baseline. N170 is the face-minus-car mean at PO7/PO8 over 110–170 ms. P3 is the target-minus-nontarget mean at Pz over 300–600 ms. Operations include higher high-pass cutoffs, EOG regression, EOG-guided ICA, the frozen compact model, and diagnostic endpoint guards. All fitted operations and decoders use development participants only.

Independent rescoring uses saved voltages, identities, decoders, source checksums, and protocol/code hashes. Participant bootstrap intervals are descriptive and conditional on fixed development fits. Joint conformal scores include both tasks, voltage and frozen-decoder endpoints, and all eight nonparent operations. A participant, not a task or trial, is the calibration unit. Parent-relative stability does not certify cortical fidelity. EOG-only decoding tests peripheral task association; retaining it cannot be equated with retaining neural information.

Guard spectral diagnostics were added after the ERP results and are explicitly post hoc. The original executed numerical code remains in the local evidence package; the public release documents its portability adaptations separately.
