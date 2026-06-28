# Findings

The honest results, in one place. The theme: **measure, don't assume — and
report the limitation even when it's inconvenient.** Every number here is on
real GTA5 data (22 sessions, 4 players) unless stated; treat them as directional
at this scale, not production guarantees.

---

### 1. A real behavioural biometric — once you control for hardware

4-class player identification scores **0.72** test accuracy (95% CI 0.60–0.83,
window bootstrap — the interval is wide because the test set is 53 windows, and
we say so). Two of the four players (hydra, dninix) were recorded on the **same
PC with identical settings** — only the human differs. Evaluating *just that
pair* gives **0.75** (vs a 0.65 majority baseline). The other two sit on their
own hardware: one (shotik) is near-perfectly separable (a hardware tell), but the
newest (ropyk) is **not** trivially separable despite a distinct rig — 13/20 test
windows, confused with dninix/shotik — evidence the sens/DPI normalisation
*removes* most of the hardware artefact rather than masking it. Adding the 4th
player lowered the headline from 0.85 (3-class) to 0.72: a harder, more honest
task. (Catching this needed fixing a **100× sensitivity data-entry error** in
ropyk's recordings — left in, the bad `norm_factor` would have made ropyk
trivially separable by a pure normalisation artefact and *inflated* the number.)

**So the honest claim is the narrower one:** on identical hardware, where the
only variable is the person, the model distinguishes two players at ~75% from
raw mouse/keyboard behaviour. SHAP shows the separation is driven by
**timing/rhythm** features (`click_interval_std`, `keystroke_periodicity`,
`burst_rate`) — i.e. a *behavioural* fingerprint, not a hardware tell.
→ `notebooks/12_explainability.ipynb`

### 2. We measure drift; we don't assume it

When real recordings replaced the original mock data (desktop mouse-wiggling),
a per-feature KS + PSI report showed **20 of 25 features drifted significantly**
(`wasd_rhythm` PSI 9.4, `speed`/`accel` 6.6–8.0). That measured shift — not a
hunch — is what triggered retraining everything on real data.
→ `pipeline/monitoring/drift.py`, `notebooks/14_drift.ipynb`

### 3. The deep model earns its place — on real data

Hand-crafted 30 s-window features detect aimbot at **~0.50 AUC (chance)**: a
150 ms snap is averaged away by windowing. The chunk-level **LSTM autoencoder**
reaches **0.79 (aimbot)** and **0.93 (triggerbot)** on real gameplay.
Per-channel reconstruction attribution shows triggerbot flags are driven **~16×**
by the mouse-click channel — exactly what triggerbot automates.
→ `docs/LSTM_AE.md`, `reports/figures/phase4_chunk_detection.png`

### 4. Chunk-level works; session-level has a ceiling (and we proved it)

The chunk detector is strong, but lifting it to a single **session** risk score
fails: every aggregation (max / p95 / fraction-above-threshold) scores **~0.50**
legit-vs-cheat. Reason: legit gameplay has its *own* natural high-reconstruction
chunks (rare fast flicks) indistinguishable from sparse injected cheat chunks
once aggregated. This was **prototyped and verified before building**, so no
saturated "live risk" score was shipped. The unblock is real *continuous* cheat
data (a real cheater cheats throughout → most chunks elevated), for which a safe,
controllable capture harness was built.
→ `docs/STREAMING.md` (Phase 4.1), `docs/CHEAT_DATA_COLLECTION.md`

### 5. More features would hurt — ablation says stop

Splitting the 25 features into 5 families and ablating each (8-seed-averaged):
single families already classify well alone (mouse-kinematics, keyboard each
≈ 0.75–0.79), and **dropping a whole family often *raises* validation accuracy**.
The model is **over-parameterised at this data scale**. So the planned
feature-expansion phase was **deferred on evidence** — the lever at this scale is
more data or feature *reduction*, not more features.
→ `notebooks/15_ablation.ipynb`

### 6. Calibration helps — but not blindly

Measured ECE + multiclass Brier, then tried post-hoc scaling: **isotonic**
improved Brier (0.275 → 0.224) while holding accuracy; **Platt made it worse**
— the small-data fragility you'd predict at 46 calibration windows. Reported as
found, not cherry-picked. (Same small-N calibration fragility is why the Phase-4
aggregator saturates.)
→ `notebooks/13_calibration.ipynb`

### 7. Validate your serving artifacts — the ONNX export was lying (found → diagnosed → fixed)

The exported `model.onnx` (shipped since Phase 4, never checked) was **numerically
unfaithful**: probability MAE **0.27** vs the sklearn model, ~38% of labels
flipped on a probe. The first hypothesis ("converter bug") was **wrong** — the
trees converted faithfully. The real diagnosis is more interesting:

1. The ai.onnx.ml `Scaler` op standardises in **float32**, perturbing scaled
   features by up to ~2e-4 (worst where feature stds are tiny).
2. The model is **razor-margin sensitive**: trained to 100% train accuracy on
   187 windows, it memorises via hairline splits — feeding *sklearn itself* the
   float32-scaled inputs reproduces the exact same 0.27 MAE. Even a 1e-7 input
   perturbation flips ~15% of test predictions. (The serving bug is finding #5
   — over-parameterisation — wearing a different hat.)

**Fix:** a composed float64 graph (`pipeline/onnx_export.py`) — `Sub`/`Div`
scaling in double precision and an ai.onnx.ml v3 `TreeEnsembleClassifier`
carrying the booster's original float64 thresholds/leaf weights
(`nodes_values_as_tensor`), re-derived from the booster dump and checksummed
against the converter's structure. Result: probability MAE **~1e-8**, **100%
label agreement** on train/val/test and random probes. Regression-gated in CI
(`tests/test_onnx_export.py`), re-validated at every export, and the inference
benchmark gates on MAE < 1e-6. sklearn remains the reference path
(p50 1.40 ms, ~89k windows/s — real-time).
→ `pipeline/onnx_export.py`, `scripts/benchmark_inference.py`

### 8. Pretraining didn't transfer — and we can point to exactly why (Phase 8)

The "data is the bottleneck" thesis suggested an obvious lever: pretrain the sequence
encoder self-supervised on a large unlabelled human-mouse corpus (CaptchaSolve30k,
≈17.7k sessions), then fine-tune. We masked-denoising-pretrained the LSTM-AE and ran a
**data-efficiency curve** — pretrained-init vs from-scratch chunk-AUC vs fine-tuning
budget — on **both** CS2CD (real cheats) and GTA. **Verdict: a rigorous null.**
Pretrained ≈ scratch at every budget (CS2CD Δ ≈ 0.000; GTA Δ = −0.001…−0.005, within
±std over 5 seeds).

The value is that we **measured the domain gap first** (the project's own KS/PSI drift
tooling, reference = captcha) and it predicts the null mechanistically: the temporal
channel `dt` is **PSI ≈ 10–12** mismatched against *both* games (fixed-tick captcha vs
event-driven game streams), and GTA's mouse-delta geometry differs (`dx` PSI 0.37) —
while captcha→CS2 movement geometry actually transfers well (`dx/dy` PSI < 0.1). So a
**generic human-mouse corpus is not a drop-in foundation for game-input biometrics** —
the data-not-capacity thesis confirmed from the other side: *out-of-domain* data doesn't
substitute for in-domain data. Honest negative result, with the next levers named
(matched temporal encoding / in-domain corpus / contrastive objective).
→ `docs/PRETRAINING.md`, `notebooks/21_pretraining.ipynb`, `pipeline/pretraining/`

---

**What this set of findings is meant to show:** the engineering is end-to-end
and the results are real, but more importantly that claims are *checked* —
hardware confounds isolated, drift quantified, a non-working approach proven
non-working before shipping, an over-fit caught by ablation, and a silent
serving bug caught by validation. For anti-cheat, where a wrong "ban" is the
expensive failure, that skepticism is the point.
