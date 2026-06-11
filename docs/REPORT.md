# Input-level behavioural biometrics for cheat detection: what works at small N

> **Status: SKELETON (created at Phase 6 start).** This is the grow-as-you-go
> tech report (deliverable **F** in [docs/ROADMAP.md](ROADMAP.md)). Each phase's
> definition of done includes drafting its section here; post to arXiv after
> Phase 8, with a blog-post condensation. Most §§ already have their evidence in
> [docs/FINDINGS.md](FINDINGS.md) and the per-topic docs — this report *condenses
> and frames*, it doesn't re-derive.

**Thesis.** Player identification and automation detection from raw mouse/keyboard
telemetry is feasible, but the honest story at small data scale is one of
*measured limits and checked claims* as much as headline metrics. We report what
works, what doesn't, and how we verified each.

---

## Abstract
<!-- 150 words, written last. Problem, method, the 2-3 numbers that survive
scrutiny (scale-up EER, chunk-AUC, the serving-fidelity + ablation rigor),
and the honest framing. -->

## 1. Introduction & problem framing
<!-- Why input-level biometrics for anti-cheat; identification vs verification;
the cost-asymmetry of a false ban. Source: README, MODEL_CARD, ETHICS.md. -->

## 2. Data
<!-- Real GTA recordings (18 sessions / 3 players), the synthetic-cheat harness,
external corpora (CS2CD, and Phase 6's Balabit/SapiMouse). Collection ethics +
the hardware-confound caveat. Source: ETHICS, RECORDING_INSTRUCTIONS, FINDINGS#1. -->

## 3. Features: windowed aggregates vs raw sequences
<!-- The 30 window features + sens/DPI/polling normalisation; why windowing
averages away a 150 ms aimbot snap; the ID/cheat feature-set decoupling.
Source: FEATURES.md, SIGNALS.md. -->

## 4. Models
<!-- LightGBM identifier; the LSTM-AE (+ TCN/Transformer tie); the Naive-Bayes
session aggregator. Source: LSTM_AE.md, ARCHITECTURE_COMPARISON.md, STREAMING.md. -->

## 5. Rigor at small N (the core contribution)
<!-- This is the section that distinguishes the report. Source: FINDINGS.md. -->
- 5.1 Hardware confound isolated (same-hardware pair = the honest biometric)
- 5.2 Drift measured, not assumed (KS+PSI, 20/25 features)
- 5.3 Ablation → over-parameterisation at N=18 (more features would hurt)
- 5.4 Calibration helps but not blindly (isotonic ok, Platt hurts)
- 5.5 Bootstrap CIs on every headline number
- 5.6 Serving-fidelity bug: found → root-caused (float32 + overfit margins) → fixed (bit-faithful float64 export) → CI-gated

## 6. Scaling identification + verification *(Phase 6)*
<!-- Users-curve to 120 (Balabit/SapiMouse), EER/DET, open-set rejection, the
smurf/account-sharing reframe + continuous-auth generalisation. Source:
notebooks/19, docs/VERIFICATION.md. -->

## 7. The detection-vs-evasion frontier *(Phase 7)*
<!-- Humanizer λ-sweep; detector-AUC vs cheat-utility equilibrium. Source:
notebooks/20, ADVERSARIAL.md arms-race section. -->

## 8. Self-supervised pretraining & data efficiency *(Phase 8)*
<!-- Masked-step pretraining on CaptchaSolve30k; domain-gap measurement; the
data-efficiency curve (pretrained vs scratch). The "data is the bottleneck"
thesis tested directly. Source: notebooks/21, PRETRAINING.md. -->

## 9. Limitations & ethics
<!-- 3 players / 1 cheat recorder; input-level only (no kernel/memory/network);
out-of-scope = real bans. Source: MODEL_CARD, ETHICS.md. -->

## 10. Conclusion
<!-- What survived scrutiny; what's data-gated; the one-line takeaway for an
anti-cheat R&D reader. -->

---

### Appendix — reproducibility
<!-- DVC pipeline, CI gates (tests + README-results staleness + ONNX parity),
MLflow registry. Source: MLOPS.md, DEPLOY.md. -->
