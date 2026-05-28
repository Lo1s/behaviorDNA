# Feature Catalogue

> Per-feature documentation for the 25 production features in [`pipeline/features/run.py`](../pipeline/features/run.py).
>
> Features are computed per 30-second window per session. Detection benchmarks against synthetic cheats are tracked in [docs/ADVERSARIAL.md](ADVERSARIAL.md); the phased roadmap is in [docs/ROADMAP.md](ROADMAP.md).

## Quick reference

| Group | Features | Primary signal |
|---|---|---|
| **Mouse kinematics** | `speed_mean`, `speed_std`, `accel_mean`, `accel_std`, `jitter`, `click_interval_mean`, `click_interval_std` | DPI-normalised motion magnitudes |
| **Mouse trajectory** *(Phase 1)* | `mouse_curvature_mean`, `mouse_curvature_std`, `path_efficiency`, `direction_changes_per_sec` | Geometry — distinguishes smooth aimbot snaps from human micro-corrections |
| **Keyboard patterns** | `hold_mean`, `hold_std`, `iki_mean`, `iki_std`, `burst_rate`, `wasd_rhythm` | Press cadence and hold dynamics |
| **Reaction timing** *(Phase 1)* | `click_reaction_mean`, `inter_click_movement` | Latency-based cheat signature — triggerbots fire at ~0 ms |
| **Keystroke geometry** *(Phase 1)* | `keystroke_periodicity` | Regularity of key-press intervals — macros produce CV → 0 |
| **Session aggregates** | `event_rate`, `mouse_key_ratio`, `active_time_pct`, `scroll_count`, `scroll_direction_ratio` | Coarse behavioural summary |

All Phase-1 features were added to close the detection gap demonstrated in Phase 3, where the original 18 features scored AUC ≈ 0.5 against synthetic cheats. After Phase 1 the benchmark improved to AUC 0.87 (triggerbot), 0.68 (macro), 0.53 (aimbot — left for Phase 2 LSTM).

---

## Mouse kinematics

DPI-normalised motion magnitudes. All speed/acceleration values are divided by `norm_factor = sensitivity × dpi / 800.0` so different hardware setups are comparable.

| Feature | Definition | Anti-cheat relevance |
|---|---|---|
| `speed_mean`, `speed_std` | Mean / std of $\|\Delta p\| / \Delta t$ across consecutive `mouse_move` events | Bots often have flatter speed distributions; humans show bursty acceleration around aim |
| `accel_mean`, `accel_std` | Mean / std of $\Delta\text{speed} / \Delta t$ | High `accel_std` correlates with manual aim correction |
| `jitter` | Total path length / Euclidean displacement (ratio ≥ 1.0) | Aimbots produce ratios near 1.0 during snaps; humans 1.2–3.0+ |
| `click_interval_mean`, `click_interval_std` | Mean / std of time between consecutive click presses | Slow auto-fire and macros show low variance |

## Mouse trajectory (Phase 1)

Geometric features designed to capture the *direction* of motion, not just the magnitude. The mean-aggregation operator dilutes the signal of brief cheat episodes inside a 30 s window — these features are most discriminative when combined with per-session aggregation (see [`pipeline.adversarial.benchmark.run_benchmark`](../pipeline/adversarial/benchmark.py) with `aggregation="session_max"`).

### `mouse_curvature_mean`, `mouse_curvature_std`

For consecutive triples $(p_{i-1}, p_i, p_{i+1})$ of mouse-move samples, the turn angle is

$$\theta_i = \arccos\!\left(\frac{\vec{v_1}\cdot\vec{v_2}}{\|\vec{v_1}\|\,\|\vec{v_2}\|}\right)$$

with $\vec{v_1} = p_i - p_{i-1}$, $\vec{v_2} = p_{i+1} - p_i$. Reported as the mean and std of $\theta_i$ across the window.

**What it catches.** Aimbots interpolate smoothly toward target — every triple along the snap has $\theta \approx 0$. Humans constantly micro-correct, producing $\theta$ distributed broadly between 0 and $\pi/2$.

**Failure mode.** A 150 ms snap is <1 % of a 30 s window's mouse events. Mean curvature stays close to the human baseline even when the aimbot is active. This motivates Phase 2.

### `path_efficiency`

$$\text{efficiency} = \frac{\|p_{\text{last}} - p_{\text{first}}\|}{\sum_i \|\vec{v_i}\|}$$

Euclidean displacement divided by total path length. `1.0` = perfectly straight; `~0.0` = highly self-intersecting.

**What it catches.** Aimbot snaps drive efficiency toward 1.0 during their brief window. Humans cover the same ground inefficiently due to corrections.

### `direction_changes_per_sec`

Count of velocity-vector sign flips on either axis per second of window duration. Humans flip often (overshoot, correct); aimbots progress monotonically to target during snaps.

## Keyboard patterns

Original keystroke aggregates from Phase 0.

| Feature | Definition | Anti-cheat relevance |
|---|---|---|
| `hold_mean`, `hold_std` | Mean / std of key-down duration (`release_t − press_t`) | Bots holding keys at fixed durations show very low `hold_std` |
| `iki_mean`, `iki_std` | Mean / std of inter-key-press intervals | High-CV humans, low-CV macros (see `keystroke_periodicity`) |
| `burst_rate` | Key presses per second in the window | Distinguishes idle vs active play more than cheat vs legit |
| `wasd_rhythm` | Variance of intervals between consecutive WASD presses | Movement-macros show pathologically low variance |

## Reaction timing (Phase 1)

### `click_reaction_mean`

For each `mouse_click` press, time delta to the most recent `mouse_move` in the same window. Mean over all clicks.

**What it catches.** Triggerbots fire when the crosshair crosses a target — zero motor latency. Humans take 100–250 ms (visual → motor cortex). This is the **single most discriminative feature** in the benchmark, lifting triggerbot detection from AUC 0.50 → 0.87.

### `inter_click_movement`

Mean Euclidean distance between consecutive click coordinates in the window.

**What it catches.** Macros / triggerbots firing without repositioning produce ~0 inter-click movement. Humans aim between shots. NaN when the window contains <2 clicks.

## Keystroke geometry (Phase 1)

### `keystroke_periodicity`

Coefficient of variation of inter-key-press intervals:

$$\text{CV} = \frac{\text{std}(\Delta t_i)}{\text{mean}(\Delta t_i)}$$

**What it catches.** Macros press at fixed intervals → CV → 0. Humans press irregularly → CV typically > 0.5. This is the **time-domain version of the FFT analysis** in [notebook 10](../notebooks/10_adversarial_bots.ipynb) — same signal, cheaper to compute per-window.

## Session aggregates

Coarse summary statistics.

| Feature | Definition |
|---|---|
| `event_rate` | Total events per second of window duration |
| `mouse_key_ratio` | Mouse-event count / (key-event count + ε) |
| `active_time_pct` | Fraction of 1-second sub-buckets in the window containing at least one event |
| `scroll_count` | Number of `mouse_scroll` events |
| `scroll_direction_ratio` | Fraction of scrolls in the "down" (positive `dy`) direction |

These exist mainly to characterise the activity context of a window. They are individually weak cheat signals but help models calibrate to different gameplay phases (combat vs menu navigation vs idle).

---

## Design decisions

**No z-score scaling at the feature stage.** Scaling is applied only inside [`pipeline/training/run.py`](../pipeline/training/run.py) using a `StandardScaler` fit on the training fold. This prevents train/test leakage.

**NaN handling.** Each helper returns `float('nan')` for degenerate windows (e.g. `click_interval_*` when the window has 0–1 clicks). Downstream stages call `.fillna(0.0)` to get a clean feature matrix. Models are tree-based or kernel-based and tolerate this.

**DPI normalisation, not feature-level standardisation.** `speed_*` and `accel_*` are divided by `sensitivity × dpi / 800.0` *during* feature computation, so different hardware setups are comparable across sessions in absolute terms. Geometric ratios (`mouse_curvature_*`, `path_efficiency`) are inherently scale-invariant and need no normalisation.

**Polling-rate normalisation.** Three features scale ~linearly with the mouse polling rate, because a 1000 Hz mouse emits ~8× more `mouse_move` events per second than a 125 Hz mouse for *identical* behaviour:

| Feature | Why it scales with polling rate |
|---|---|
| `event_rate` | dominated by `mouse_move` count, which is ~polling_rate |
| `mouse_key_ratio` | numerator (mouse events) scales with polling rate; denominator (keys) doesn't |
| `direction_changes_per_sec` | more samples capture more velocity sign-flips |

These are multiplied by `rate_norm = REFERENCE_POLLING_RATE / polling_rate` (reference = 1000 Hz) in `compute_session_aggregates` / `compute_trajectory_features`, so two recordings of the same behaviour on different hardware land at the same value. When `polling_rate` is missing or non-positive, `rate_norm = 1.0` (no-op, backwards-compatible). The same `rate_norm` is applied consistently in the training pipeline (`pipeline/features/run.py:run`), the adversarial benchmark, and the streaming engine so train/inference features don't diverge.

**Left unnormalised on purpose:** `speed_*` / `accel_*` are `dist/dt` ratios where higher polling shrinks both numerator and denominator → approximately rate-invariant already (and DPI-normalised). Keyboard features (`burst_rate`, `iki_*`, `hold_*`, `keystroke_periodicity`, `wasd_rhythm`) are driven by key events, not the mouse polling clock. `jitter` (path/displacement) is *mildly* polling-sensitive but not linearly — perfectly cancelling it would need event-stream resampling, which is out of scope; the linear scaling above handles the dominant effect. The drift-detection tool ([docs/MONITORING.md](MONITORING.md)) is the way to verify, on real mixed-hardware data, that normalisation actually cancels the gap.

**Window size = 30 s, non-overlapping.** Hardcoded as `WINDOW_MS` in `pipeline/features/run.py`. Larger windows give more stable feature estimates but dilute brief cheat signals further; smaller windows go the other way. 30 s was chosen empirically — see [notebooks/02_features.ipynb](../notebooks/02_features.ipynb).
