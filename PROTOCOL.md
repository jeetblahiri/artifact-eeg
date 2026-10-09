# Numerical protocol and chronology

The reconstruction protocol was fixed locally on 8 October 2026 after earlier related results were known. It uses source-window-disjoint 60/20/20 splits, power-SNR −7/−4/−1/+2 dB, 10000/2000/3000 mixtures, three training seeds, two learning rates, and twenty epochs. The ECG proxy uses recording-disjoint splits. The matched reconstruction-to-task transfer was a declared post hoc analysis. Settings are in `config.json`.

The known-source and reference-bias controls were fixed after those outcomes and before their own execution. Settings are in `target_validity_config.json`. Bias radii are hypothetical sensitivity assumptions, not estimates of biological contamination. The known-source generator preserves the observed waveform across source allocations.

The independent ERP protocol was fixed on 9 October 2026 before scoring new ERP EEG outcomes, while previous synthetic and BCI outcomes were known. It is exploratory and was not externally registered. All forty participants and both N170/P3 tasks were selected as a complete cohort. Participants stay in the same development/calibration/evaluation split across tasks. `tim_validation_config.json` preserves the executed settings and chronology.

ERP processing uses scalp-average reference, resampling to 256 Hz, continuous forward–backward 0.1–30 Hz parent filtering, −0.5 to 1.5 s epochs, and a −0.2 to 0 s baseline. N170 is the face-minus-car mean at PO7/PO8 over 110–170 ms. P3 is the target-minus-nontarget mean at Pz over 300–600 ms. Operations include higher high-pass cutoffs, EOG regression, EOG-guided ICA, the frozen compact model, and diagnostic endpoint guards. All fitted operations and decoders use development participants only.

Independent rescoring uses saved voltages, identities, decoders, source checksums, and protocol/code hashes. Participant bootstrap intervals are descriptive and conditional on fixed development fits. Joint conformal scores include both tasks, voltage and frozen-decoder endpoints, and all eight nonparent operations. A participant, not a task or trial, is the calibration unit. Parent-relative stability does not certify cortical fidelity. EOG-only decoding tests peripheral task association; retaining it cannot be equated with retaining neural information.

Guard spectral diagnostics were added after the ERP results and are explicitly post hoc. The original executed numerical code remains in the local evidence package; the public release documents its portability adaptations separately.
