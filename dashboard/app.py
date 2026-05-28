"""BehaviorDNA — Streamlit demo dashboard."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features.run import FEATURE_COLS

MODEL_PATH = ROOT / "models" / "model.pkl"
SPLITS = ROOT / "data" / "splits"

FEATURE_GROUPS = {
    "Mouse kinematics": [
        "speed_mean",
        "speed_std",
        "accel_mean",
        "accel_std",
        "jitter",
        "click_interval_mean",
        "click_interval_std",
    ],
    "Keyboard patterns": [
        "hold_mean",
        "hold_std",
        "iki_mean",
        "iki_std",
        "burst_rate",
        "wasd_rhythm",
    ],
    "Session aggregates": [
        "event_rate",
        "mouse_key_ratio",
        "active_time_pct",
        "scroll_count",
        "scroll_direction_ratio",
    ],
}


@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_all_splits() -> pd.DataFrame:
    dfs = []
    for split in ("train", "val", "test"):
        p = SPLITS / f"{split}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["split"] = split
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def load_test() -> pd.DataFrame:
    p = SPLITS / "test.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BehaviorDNA",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 BehaviorDNA — Behavioral Biometrics Explorer")

artifact = load_artifact()
all_data = load_all_splits()
test_data = load_test()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "👤 Player Profiles",
        "🔮 Predict",
        "🕵️ Session Explorer",
        "📡 Live Session",
    ]
)

# ── Tab 1 — Overview ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Model artifact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", artifact.get("model_type", "—"))
    c2.metric("Task", artifact.get("task", "—"))
    c3.metric("Trained", "Yes" if artifact.get("trained") else "No")
    n_classes = len(artifact.get("classes") or [])
    c4.metric("Players", n_classes)

    st.subheader("Dataset")
    if not all_data.empty:
        ca, cb, cc = st.columns(3)
        ca.metric("Total windows", len(all_data))
        cb.metric("Unique players", all_data["player"].nunique())
        cc.metric("Features", len(FEATURE_COLS))

        counts = (
            all_data.groupby(["player", "split"]).size().reset_index(name="windows")
        )
        fig = go.Figure()
        for split, color in [
            ("train", "#4c78a8"),
            ("val", "#f58518"),
            ("test", "#54a24b"),
        ]:
            d = counts[counts["split"] == split]
            fig.add_trace(
                go.Bar(name=split, x=d["player"], y=d["windows"], marker_color=color)
            )
        fig.update_layout(
            barmode="stack",
            title="Windows per player by split",
            xaxis_title="Player",
            yaxis_title="Windows",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature groups")
    rows = [
        {"Group": group, "Feature": feat}
        for group, feats in FEATURE_GROUPS.items()
        for feat in feats
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab 2 — Player Profiles ───────────────────────────────────────────────────
with tab2:
    if all_data.empty:
        st.warning("No split data found — run `dvc repro` to generate splits.")
    else:
        players = sorted(all_data["player"].unique().tolist())
        player_means = all_data.groupby("player")[FEATURE_COLS].mean()

        selected = st.multiselect(
            "Compare players (up to 4)",
            players,
            default=players[:2] if len(players) >= 2 else players,
            max_selections=4,
        )

        if selected:
            mins = player_means[FEATURE_COLS].min()
            maxs = player_means[FEATURE_COLS].max()
            rng = (maxs - mins).replace(0, 1.0)
            categories = FEATURE_COLS + [FEATURE_COLS[0]]
            fig_radar = go.Figure()
            palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
            for i, p in enumerate(selected):
                norm = ((player_means.loc[p, FEATURE_COLS] - mins) / rng).tolist()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=norm + [norm[0]],
                        theta=categories,
                        fill="toself",
                        name=p,
                        line_color=palette[i % len(palette)],
                        opacity=0.65,
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Behavioral fingerprint — normalized per feature across players",
                height=520,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("All players — feature heatmap (z-score)")
        zscored = (player_means - player_means.mean()) / player_means.std().replace(
            0, 1.0
        )
        fig_heat = go.Figure(
            go.Heatmap(
                z=zscored.values,
                x=FEATURE_COLS,
                y=players,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="z-score"),
            )
        )
        fig_heat.update_layout(
            title="Player × feature z-score",
            xaxis_tickangle=-40,
            height=max(300, 70 * len(players)),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 3 — Predict ───────────────────────────────────────────────────────────
with tab3:
    model_type = artifact.get("model_type", "")
    trained = artifact.get("trained", False)

    if not trained:
        st.warning("Model not trained — run `dvc repro` first.")
    elif all_data.empty:
        st.warning("No split data found — run `dvc repro` to generate splits.")
    else:
        train_df = all_data[all_data["split"] == "train"]
        feat_means = train_df[FEATURE_COLS].mean()
        feat_mins = train_df[FEATURE_COLS].min()
        feat_maxs = train_df[FEATURE_COLS].max()

        st.subheader("Adjust behavioral features")
        st.caption(
            "Sliders default to the training-set mean. "
            "Drag to simulate a different behavioral pattern."
        )

        values: dict[str, float] = {}
        slider_cols = st.columns(3)
        for idx, feat in enumerate(FEATURE_COLS):
            lo = float(feat_mins[feat])
            hi = float(feat_maxs[feat])
            default = float(feat_means[feat])
            if lo >= hi:
                hi = lo + 1.0
            values[feat] = slider_cols[idx % 3].slider(
                feat,
                min_value=lo,
                max_value=hi,
                value=default,
                format="%.4f",
                key=f"slider_{feat}",
            )

        if st.button("🔮 Predict", type="primary"):
            scaler = artifact["scaler"]
            mdl = artifact["model"]
            x = np.array([[values[f] for f in FEATURE_COLS]])
            x_sc = scaler.transform(x)

            if model_type == "lightgbm":
                le = artifact["label_encoder"]
                proba = mdl.predict_proba(x_sc)[0]
                classes = le.classes_
                best = int(np.argmax(proba))
                st.metric(
                    "Predicted player",
                    classes[best],
                    f"{proba[best]:.1%} confidence",
                )
                fig_p = go.Figure(
                    go.Bar(
                        x=proba,
                        y=classes,
                        orientation="h",
                        marker_color=[
                            "#54a24b" if i == best else "#4c78a8"
                            for i in range(len(classes))
                        ],
                    )
                )
                fig_p.update_layout(
                    title="Identification probabilities",
                    xaxis_title="Probability",
                    xaxis_range=[0, 1],
                    height=max(250, 50 * len(classes)),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            elif model_type == "isolation_forest":
                score = float(mdl.score_samples(x_sc)[0])
                is_anomaly = mdl.predict(x_sc)[0] == -1
                st.metric(
                    "Anomaly score",
                    f"{score:.4f}",
                    "anomaly" if is_anomaly else "normal",
                )
            else:
                st.info(
                    f"Prediction display not implemented for"
                    f" model type '{model_type}'."
                )


# ── Tab 4 — Session Explorer ──────────────────────────────────────────────────
with tab4:
    if test_data.empty:
        st.warning("No test split found — run `dvc repro` to generate splits.")
    else:
        sessions = sorted(test_data["session_id"].unique().tolist())

        def _session_label(sid: str) -> str:
            player = test_data.loc[test_data["session_id"] == sid, "player"].iloc[0]
            return f"{sid}  [{player}]"

        selected_session = st.selectbox(
            "Select session", sessions, format_func=_session_label
        )
        feat_choice = st.selectbox("Feature to plot", FEATURE_COLS, key="feat_explorer")

        sess = (
            test_data[test_data["session_id"] == selected_session]
            .copy()
            .sort_values("window_idx")
            .reset_index(drop=True)
        )

        scaler = artifact["scaler"]
        mdl = artifact["model"]
        model_type = artifact.get("model_type", "")
        x_all = scaler.transform(sess[FEATURE_COLS].fillna(0).values)

        if model_type == "lightgbm":
            le = artifact["label_encoder"]
            preds = le.inverse_transform(mdl.predict(x_all))
            sess["predicted"] = preds
            actual_player = sess["player"].iloc[0]
            dot_colors = ["#54a24b" if p == actual_player else "#e45756" for p in preds]
        else:
            sess["predicted"] = None
            dot_colors = ["#4c78a8"] * len(sess)

        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=sess["window_idx"],
                y=sess[feat_choice],
                mode="lines",
                name=feat_choice,
                line=dict(color="#aec7e8", width=2),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=sess["window_idx"],
                y=sess[feat_choice],
                mode="markers",
                marker=dict(color=dot_colors, size=10),
                name="predicted (green=correct, red=wrong)",
            )
        )
        fig_line.update_layout(
            title=f"Session {selected_session} — {feat_choice}",
            xaxis_title="Window index",
            yaxis_title=feat_choice,
            height=370,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        display_cols = ["window_idx", "player"] + FEATURE_COLS[:5]
        if model_type == "lightgbm":
            display_cols.append("predicted")
        st.dataframe(sess[display_cols], use_container_width=True, hide_index=True)

        if model_type == "isolation_forest":
            scores = mdl.score_samples(x_all)
            fig_sc = go.Figure(
                go.Scatter(
                    x=sess["window_idx"],
                    y=scores,
                    mode="markers+lines",
                    marker=dict(color="#e45756"),
                    name="anomaly score",
                )
            )
            fig_sc.update_layout(
                title="Anomaly score per window (lower = more anomalous)",
                xaxis_title="Window index",
                yaxis_title="score_samples()",
                height=300,
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 5 — Live Session ──────────────────────────────────────────────────────
with tab5:
    import json
    import time

    st.subheader("📡 Live cheat-risk timeline")
    st.markdown("""
        Replays a recorded session through the same streaming pipeline an
        always-on anti-cheat would use (see [`docs/STREAMING.md`](../docs/STREAMING.md)).
        Optionally injects a synthetic cheat partway through so the risk score
        rises in real time — the "money shot" of the Phase 4 work.

        > **Mock-data caveat:** all current `data/raw/*.json` recordings are mouse-on-desktop
        > mock data, not real gameplay. The streaming pipeline runs correctly; the
        > *absolute* risk magnitudes will tighten once the GTA recordings land.
        """)

    RAW_DIR = ROOT / "data" / "raw"
    session_files = sorted(RAW_DIR.glob("*.json"))
    if not session_files:
        st.warning(f"No session JSONs in {RAW_DIR}. Record some sessions first.")
    else:
        with st.form("live_replay_form"):
            col_a, col_b = st.columns(2)
            session_path = col_a.selectbox(
                "Session to replay",
                options=[p.name for p in session_files],
                index=min(len(session_files) - 1, len(session_files) - 1),
            )
            cheat_type = col_b.selectbox(
                "Inject synthetic cheat",
                options=["(none)", "aimbot", "triggerbot", "macro"],
                index=1,
            )
            inject_at_s = col_b.number_input(
                "Inject at (seconds)", min_value=0.0, value=10.0, step=5.0
            )
            speed = col_a.slider(
                "Replay speed (events/wall-clock-sec)",
                min_value=1,
                max_value=2000,
                value=400,
                step=100,
                help=(
                    "Higher = faster. The streaming engine is event-driven so "
                    "this only controls how often the chart redraws."
                ),
            )
            submitted = st.form_submit_button("▶  Run live replay", type="primary")

        chart_placeholder = st.empty()
        status_placeholder = st.empty()

        if submitted:
            # Imports kept inside the handler so the dashboard loads without
            # building the streaming pipeline upfront.
            from pipeline.inference.streaming import build_stream_state
            from scripts.replay_session import _maybe_inject_cheat

            with st.spinner(
                "Loading streaming pipeline (LSTM-AE + detectors + aggregator)…"
            ):
                state = build_stream_state()

            with open(RAW_DIR / session_path, encoding="utf-8") as f:
                session = json.load(f)
            cheat = None if cheat_type == "(none)" else cheat_type
            session = _maybe_inject_cheat(
                session, cheat, inject_at_s if cheat else None
            )

            events = session.get("events", [])
            updates: list[dict] = []
            t_history: list[float] = []
            risk_history: list[float] = []

            redraw_every = max(1, len(events) // 60)  # ~60 chart frames
            last_redraw_wall = 0.0

            for i, ev in enumerate(events):
                update = state.push_event(ev)
                if update is None:
                    continue
                d = update.to_dict()
                updates.append(d)
                t_history.append(d["t"] / 1000.0)
                risk_history.append(d["session_risk"])

                # Throttle redraws: every redraw_every events AND no faster
                # than the user-selected wall-clock speed allows.
                if i % redraw_every == 0:
                    wall_now = time.time()
                    if wall_now - last_redraw_wall >= 1.0 / max(
                        speed / redraw_every, 1
                    ):
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=t_history,
                                y=risk_history,
                                mode="lines",
                                line=dict(color="#e94560", width=2.5),
                                name="combined risk",
                            )
                        )
                        fig.add_hline(
                            y=0.5,
                            line=dict(color="#8892a4", dash="dash"),
                            annotation_text="alert threshold",
                            annotation_position="bottom right",
                        )
                        if cheat:
                            fig.add_vline(
                                x=inject_at_s,
                                line=dict(color="black", dash="dot"),
                                annotation_text=f"{cheat} injected",
                                annotation_position="top right",
                            )
                        fig.update_layout(
                            title=f"Live session risk — {session_path}",
                            xaxis_title="time (s)",
                            yaxis_title="session risk",
                            yaxis=dict(range=[0, 1.05]),
                            height=380,
                            uirevision="live",  # keep zoom across redraws
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        status_placeholder.markdown(
                            f"**Events processed:** {i + 1:,}/{len(events):,}  ·  "
                            f"**windows:** {d['n_windows']}  ·  **chunks:** {d['n_chunks']}  ·  "
                            f"**risk now:** `{d['session_risk']:.3f}`"
                        )
                        last_redraw_wall = wall_now

            # Final snapshot
            final = state.finalize()
            if final is not None:
                final_d = final.to_dict()
                updates.append(final_d)
                t_history.append(final_d["t"] / 1000.0)
                risk_history.append(final_d["session_risk"])

                # Per-detector contribution panel
                fig_main = go.Figure()
                fig_main.add_trace(
                    go.Scatter(
                        x=t_history,
                        y=risk_history,
                        mode="lines",
                        line=dict(color="#e94560", width=2.5),
                        name="combined risk",
                    )
                )
                fig_main.add_hline(
                    y=0.5,
                    line=dict(color="#8892a4", dash="dash"),
                    annotation_text="alert threshold",
                    annotation_position="bottom right",
                )
                if cheat:
                    fig_main.add_vline(
                        x=inject_at_s,
                        line=dict(color="black", dash="dot"),
                        annotation_text=f"{cheat} injected",
                        annotation_position="top right",
                    )
                fig_main.update_layout(
                    title=f"Final replay — {session_path}",
                    xaxis_title="time (s)",
                    yaxis_title="session risk",
                    yaxis=dict(range=[0, 1.05]),
                    height=380,
                )
                chart_placeholder.plotly_chart(fig_main, use_container_width=True)

                # Per-detector logits panel
                detector_names = sorted(
                    {k for u in updates for k in u.get("detector_logits", {})}
                )
                if detector_names:
                    fig_det = go.Figure()
                    for name in detector_names:
                        ys = [u["detector_logits"].get(name, 0.0) for u in updates]
                        xs = [u["t"] / 1000.0 for u in updates]
                        fig_det.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode="lines",
                                name=name,
                                line=dict(width=1.5),
                            )
                        )
                    fig_det.update_layout(
                        title="Per-detector logit contribution",
                        xaxis_title="time (s)",
                        yaxis_title="logit",
                        height=280,
                    )
                    st.plotly_chart(fig_det, use_container_width=True)

                status_placeholder.markdown(
                    f"**Final session risk:** `{final_d['session_risk']:.3f}` over "
                    f"{final_d['n_events']:,} events  ·  {final_d['n_windows']} windows  ·  "
                    f"{final_d['n_chunks']} chunks"
                )

                with st.expander("Raw ScoreUpdate snapshots (JSON)"):
                    st.json(updates[-5:])
