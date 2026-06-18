import json
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Agent Observability",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from the root .env file (if running locally)
# Later when deployed to Streamlit Cloud, these will go into Streamlit Secrets.
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000/api/v1"))
ADMIN_API_KEY = st.secrets.get("ADMIN_API_KEY", os.getenv("ADMIN_API_KEY", "dev_secret_key"))

@st.cache_data(ttl=60)
def fetch_dashboard_data(limit=500):
    headers = {"X-API-Key": ADMIN_API_KEY}
    try:
        response = requests.get(f"{API_URL}/admin/logs?limit={limit}", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch data from API: {e}")
        return None

@st.dialog("Chat Details", width="large")
def show_chat_details(row):
    with st.chat_message("user"):
        st.write(row['query'])
        
    with st.chat_message("assistant"):
        st.write(row['answer'])
        st.caption(f"Latency: {row.get('latency_s', 0)}s | Tokens: {row.get('tokens', 0)} | Chunks: {row.get('retrieved', 0)} | Intent: {row['intent']} | Time: {row['created_at']}")
        st.caption(f"Faithfulness: {row.get('faithfulness', 'N/A')} | Empathy: {row.get('empathy', 'N/A')} | Reasoning: {row.get('reasoning', 'N/A')} | Lookup: {row.get('lookup', 'N/A')}")
        
        # Show retrieved context if available
        retrieved_context = row.get('retrieved_context', [])
        if isinstance(retrieved_context, list) and len(retrieved_context) > 0:
            with st.expander(f"View Retrieved Context ({len(retrieved_context)} chunks)"):
                for idx, chunk in enumerate(retrieved_context):
                    st.markdown(f"**[{idx+1}] {chunk.get('course_name') or chunk.get('title') or 'Unknown'}** (Score: `{chunk.get('score', 0):.4f}`)")
                    st.text(chunk.get('text', ''))
                    st.divider()
        
        # Judgment Label
        issues = []
        if pd.notna(row.get('faithfulness')) and row.get('faithfulness') is not None:
            if float(row['faithfulness']) < 0.8:
                issues.append("Faithfulness Rendah (Potensi Halusinasi)")
        if row.get('intent') == 'KNOWLEDGE' and row.get('retrieved', 0) == 0:
            issues.append("KNOWLEDGE tapi tidak ada chunk ditarik")
        
        if issues:
            st.error(f"Problematic Chat: {', '.join(issues)}")
        elif pd.notna(row.get('faithfulness')) and row.get('faithfulness') is not None:
            st.success("Healthy Chat (Faithful)")

# --- UI Layout ---

st.title("Agent Observability Dashboard")
st.markdown("Dashboard ini mengambil data via REST API FastAPI secara aman dan ringan.")

data = fetch_dashboard_data(limit=500)

if not data:
    st.stop()

kpis = data.get("kpis", {})
intents = data.get("intents", [])
trends = data.get("trends", [])
logs = data.get("logs", [])
users = data.get("users", [])

# Setup tabs
tab_overview, tab_explorer, tab_ltm, tab_gate = st.tabs([
    "Overview & Recent Logs",
    "Session Explorer",
    "User LTM Profiles",
    "Intent Gate Monitor",
])

with tab_overview:
    # KPI Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", f"{kpis.get('total_queries', 0):,}")
    col2.metric("Avg Latency", f"{kpis.get('avg_latency', 0.0)/1000:.2f} s")
    col3.metric("Cache Hit Rate", f"{kpis.get('hit_rate', 0.0):.1f}%")

    # Rolling-7d KPIs — p95/p99 expose the latency tail that Avg hides, and
    # faithfulness is the sampled judge quality score. "—" when no data.
    col4, col5, col6 = st.columns(3)
    col4.metric("P95 Latency (7d)", f"{kpis.get('p95_latency_7d', 0.0)/1000:.2f} s")
    col5.metric("P99 Latency (7d)", f"{kpis.get('p99_latency_7d', 0.0)/1000:.2f} s")
    _faith = kpis.get('faithfulness_avg_7d')
    _faith_n = kpis.get('faithfulness_n_7d', 0)
    _faith_fail = kpis.get('faithfulness_fail_7d', 0)
    col6.metric(
        "Faithfulness (7d)",
        f"{_faith:.3f}" if _faith is not None else "—",
        delta=f"-{_faith_fail} unfaithful" if _faith_fail else None,
        delta_color="inverse",
        help=f"Avg LLM-judge faithfulness over {_faith_n} evaluated turns (sampled). "
             f"{_faith_fail} scored below the {0.75} pass threshold.",
    )

    st.divider()

    # Charts Row
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Distribusi Intent")
        if intents:
            df_intents = pd.DataFrame(intents)
            fig_pie = px.pie(df_intents, values='count', names='intent', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Belum ada data intent.")

    with col_chart2:
        st.subheader("Tren Request Harian")
        if trends:
            df_trends = pd.DataFrame(trends)
            fig_line = px.line(df_trends, x='date', y='queries', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Belum ada data tren.")

    st.divider()

    # Log Viewer Section
    st.subheader("Recent Chat Logs Viewer")
    st.markdown("Pilih salah satu baris di bawah ini untuk melihat percakapan singkat.")

    if not logs:
        st.info("Belum ada data log.")
    else:
        df_logs = pd.DataFrame(logs)
        if 'created_at' in df_logs.columns:
            # DB stores UTC (DateTime(timezone=True)); show in WIB/GMT+7.
            df_logs['created_at'] = (
                pd.to_datetime(df_logs['created_at'], utc=True)
                .dt.tz_convert('Asia/Jakarta')
                .dt.strftime('%d/%m/%Y %H:%M:%S')
            )

        # Defensive programming: ensure new columns exist in case the backend API is outdated
        for col in ['faithfulness', 'empathy', 'reasoning', 'lookup', 'tokens', 'retrieved']:
            if col not in df_logs.columns:
                df_logs[col] = None
                
        # Calculate latency in seconds
        df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
        
        # Truncate text for the table view
        df_logs['query_short'] = df_logs['query'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        df_logs['answer_short'] = df_logs['answer'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        
        # Use dataframe selection
        event = st.dataframe(
            df_logs[['created_at', 'session_id', 'intent', 'latency_s', 'tokens', 'retrieved', 'cache_hit', 'faithfulness', 'empathy', 'reasoning', 'lookup', 'query_short', 'answer_short']],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        selected_rows = event.selection.rows
        if selected_rows:
            idx = selected_rows[0]
            selected_log = df_logs.iloc[idx]
            show_chat_details(selected_log)

with tab_explorer:
    st.subheader("Eksplorasi Riwayat Sesi")
    st.markdown("Pilih **Session ID** untuk melihat urutan percakapan secara kronologis.")
    
    if not logs:
        st.info("Belum ada data log.")
    else:
        df_logs = pd.DataFrame(logs)
        if 'created_at' in df_logs.columns:
            # DB stores UTC (DateTime(timezone=True)); show in WIB/GMT+7.
            df_logs['created_at'] = (
                pd.to_datetime(df_logs['created_at'], utc=True)
                .dt.tz_convert('Asia/Jakarta')
                .dt.strftime('%d/%m/%Y %H:%M:%S')
            )
        if 'latency_ms' in df_logs.columns:
            df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
        
        # Group by session_id to get summary
        if 'session_id' in df_logs.columns:
            session_summary = df_logs.groupby('session_id').agg(
                latest_activity=('created_at', 'max'),
                total_turns=('query', 'count')
            ).reset_index().sort_values('latest_activity', ascending=False)
            
            st.markdown("Pilih baris di tabel ini untuk melihat riwayat percakapannya:")
            event_session = st.dataframe(
                session_summary,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            selected_rows = event_session.selection.rows
            if selected_rows:
                idx = selected_rows[0]
                selected_session = session_summary.iloc[idx]['session_id']
                
                # Filter logs for selected session and sort chronologically (oldest first)
                session_logs = df_logs[df_logs['session_id'] == selected_session].sort_values('created_at', ascending=True)
                
                st.markdown(f"### Riwayat Chat: `{selected_session}`")
                st.markdown(f"**Total percakapan:** {len(session_logs)} giliran. Klik pada baris untuk melihat detailnya.")
                
                # Format text for table view
                session_logs['query_short'] = session_logs['query'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
                session_logs['answer_short'] = session_logs['answer'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
                
                # Use dataframe selection
                event_turn = st.dataframe(
                    session_logs[['created_at', 'intent', 'latency_s', 'tokens', 'retrieved', 'cache_hit', 'faithfulness', 'query_short', 'answer_short']],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"session_turns_table_{selected_session}"
                )
                
                selected_turn_rows = event_turn.selection.rows
                if selected_turn_rows:
                    idx_turn = selected_turn_rows[0]
                    selected_turn = session_logs.iloc[idx_turn]
                    show_chat_details(selected_turn)

with tab_ltm:
    st.subheader("User LTM Profiles")
    st.markdown("Preferensi dan informasi profil jangka panjang (Long-Term Memory) dari masing-masing pengguna.")
    
    if not users:
        st.info("Belum ada data User LTM.")
    else:
        df_users = pd.DataFrame(users)
        df_users = df_users.rename(columns={"user_id": "session_id"})
        st.dataframe(
            df_users,
            use_container_width=True,
            hide_index=True
        )

# ─── Intent Gate Monitor tab ────────────────────────────────────────────────
# Reads JSON snapshots written by scripts/auto_calibrate_intent_gate.py
# (cron 03:00 WIB daily). Shows the latest snapshot, the per-class
# recommended thresholds, and the drift trend across all snapshots.
# If no snapshots exist yet, shows a friendly "first run pending" state
# with a button to trigger one immediately.

CALIB_DIR = Path(__file__).parent.parent / "eval" / "results"


@st.cache_data(ttl=60)
def _load_calibration_snapshots():
    """Load every intent_gate_calibration_*.json in eval/results/. Returns
    an empty list when none exist (the typical pre-traffic state)."""
    if not CALIB_DIR.exists():
        return []
    files = sorted(CALIB_DIR.glob("intent_gate_calibration_*.json"))
    out = []
    for f in files:
        try:
            out.append((f.name, json.loads(f.read_text(encoding="utf-8"))))
        except Exception as e:
            st.warning(f"Skipping unreadable snapshot {f.name}: {e}")
    return out


with tab_gate:
    st.subheader("Intent Gate — Live Calibration")
    st.markdown(
        "Threshold & margin untuk **semantic intent gate** (Tier-0 embedding). "
        "Diset otomatis tiap hari 03:00 WIB dari `agent_logs.gate_*` oleh "
        "`scripts/auto_calibrate_intent_gate.py`. Drift > 0.05 dari setting "
        "aktif = butuh review manual."
    )

    snapshots = _load_calibration_snapshots()

    if not snapshots:
        st.info(
            "Belum ada snapshot. Jalankan: `python -m scripts.auto_calibrate_intent_gate` "
            "(atau tunggu cron harian)."
        )
        if st.button("Jalankan kalibrasi sekarang"):
            with st.spinner("Mengambil data agent_logs..."):
                try:
                    import subprocess
                    import sys
                    result = subprocess.run(
                        [sys.executable, "-m", "scripts.auto_calibrate_intent_gate"],
                        capture_output=True, text=True, timeout=240,
                        cwd=str(Path(__file__).parent.parent),
                    )
                    st.code(result.stdout[-2000:], language="bash")
                    if result.returncode != 0:
                        st.warning(f"Drift terdeteksi (exit={result.returncode}). Cek rekomendasi di bawah.")
                    _load_calibration_snapshots.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menjalankan kalibrasi: {e}")
    else:
        latest_name, latest = snapshots[-1]
        cur = latest.get("current_settings", {})
        drift = latest.get("drift_alert", False)

        # KPI strip
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows analyzed", f"{latest.get('rows_analyzed', 0):,}")
        c2.metric("Current threshold", f"{cur.get('threshold', '?')}")
        c3.metric("Current margin", f"{cur.get('margin', '?')}")
        c4.metric("Drift alert", "🚨 YES" if drift else "OK")

        st.caption(f"Snapshot: `{latest_name}` · generated {latest.get('generated_at', '?')}")

        # Decision distribution
        dec = latest.get("decision_distribution", {})
        if dec:
            st.markdown("##### Decision distribution")
            df_dec = pd.DataFrame(
                [{"decision": k, "count": v} for k, v in dec.items()]
            )
            st.bar_chart(df_dec, x="decision", y="count", height=200)

        # Per-class recommendations
        per_class = latest.get("per_class", {})
        if per_class:
            st.markdown("##### Per-class recommendations")
            rows = []
            for intent in sorted(per_class):
                info = per_class[intent]
                rows.append({
                    "intent": intent,
                    "n_samples": info.get("n_samples", 0),
                    "n_caught": info.get("n_caught", 0),
                    "lowest_caught_cosine": info.get("lowest_caught_cosine", "-"),
                    "recommended_threshold": info.get("recommended_threshold", "-"),
                    "TPR": info.get("TPR_at_recommended", "-"),
                    "FP": info.get("FP_at_recommended", "-"),
                    "note": info.get("note", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Drift trend across all snapshots
        if len(snapshots) >= 2:
            st.markdown("##### Threshold drift over time")
            trend_rows = []
            for name, snap in snapshots:
                ts = snap.get("generated_at", "")
                cs = snap.get("current_settings", {})
                cur_thr = cs.get("threshold")
                for intent, info in snap.get("per_class", {}).items():
                    rec = info.get("recommended_threshold")
                    if rec is None or cur_thr is None:
                        continue
                    trend_rows.append({
                        "generated_at": ts,
                        "intent": intent,
                        "current": cur_thr,
                        "recommended": rec,
                        "delta": round(rec - cur_thr, 3),
                    })
            if trend_rows:
                df_trend = pd.DataFrame(trend_rows)
                # Plot recommended per intent over time, current as horizontal ref.
                fig = px.line(
                    df_trend, x="generated_at", y="recommended",
                    color="intent", markers=True,
                    title="Recommended threshold per intent over time",
                )
                if cur_thr is not None:
                    fig.add_hline(
                        y=cur_thr, line_dash="dash", line_color="gray",
                        annotation_text=f"current={cur_thr}",
                    )
                st.plotly_chart(fig, use_container_width=True)
