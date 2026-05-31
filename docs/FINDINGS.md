# Findings

The honest results, in one place. The theme: **measure, don't assume — and
report the limitation even when it's inconvenient.** Every number here is on
real GTA5 data (18 sessions, 3 players) unless stated; treat them as directional
at this scale, not production guarantees.

---

### 1. A real behavioural biometric — once you control for hardware

3-class player identification scores **0.853** test accuracy. But two of the
players (hydra, dninix) were recorded on the **same PC with identical settings**
— only the human differs. Evaluating *just that pair* gives **0.75** (vs a 0.65
majority baseline). The third player sits on different hardware and is trivially
separable, which inflates the 3-class number.

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
The model is **over-parameterised at 18 sessions**. So the planned
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

### 7. Validate your serving artifacts — the ONNX export was lying

The exported `model.onnx` (shipped since Phase 4, never checked) is **numerically
unfaithful**: probability MAE **0.13** vs the sklearn model, ~17% of labels flip,
from an onnxmltools/LightGBM multiclass-converter incompatibility. Caught by a
probability-parity check (a sklearn-native export, by contrast, matches to
<1e-5). The benchmark now gates on this, the figure flags it, and `export_onnx`
self-validates at export time. sklearn inference is the trustworthy path
(p50 1.40 ms, ~89k windows/s — real-time).
→ `scripts/benchmark_inference.py`

---

**What this set of findings is meant to show:** the engineering is end-to-end
and the results are real, but more importantly that claims are *checked* —
hardware confounds isolated, drift quantified, a non-working approach proven
non-working before shipping, an over-fit caught by ablation, and a silent
serving bug caught by validation. For anti-cheat, where a wrong "ban" is the
expensive failure, that skepticism is the point.
