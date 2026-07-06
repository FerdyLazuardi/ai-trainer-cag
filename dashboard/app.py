import json
import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="CAG Agent Observability",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from the root .env file (if running locally)
# Later when deployed to Streamlit Cloud, these will go into Streamlit Secrets.
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path, override=True)

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

try:
    ADMIN_API_KEY = st.secrets["ADMIN_API_KEY"]
except Exception:
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "dev_secret_key")

# Strip quotes from environment variables
if API_URL:
    API_URL = API_URL.strip('"').strip("'")
if ADMIN_API_KEY:
    ADMIN_API_KEY = ADMIN_API_KEY.strip('"').strip("'")

@st.cache_data(ttl=3600)
def fetch_usd_to_idr_rate():
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        response.raise_for_status()
        data = response.json()
        rate = data.get("rates", {}).get("IDR")
        if rate:
            return float(rate)
    except Exception as e:
        st.sidebar.warning(f"Gagal mengambil kurs realtime: {e}. Fallback: Rp 16.000")
    return 16000.0

USD_TO_IDR = fetch_usd_to_idr_rate()


def display_model(value):
    model = str(value or "").strip()
    if not model:
        return "-"
    model = model.split("/")[-1]
    return re.sub(r"-20\d{6}.*$", "", model) or "-"

@st.cache_data(ttl=15)
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
        if pd.notna(row.get('rewritten_query')) and row.get('rewritten_query') and row.get('rewritten_query') != row['query']:
            st.caption(f"**AI Query Rewrite (for DB search):** {row['rewritten_query']}")
        
    with st.chat_message("assistant"):
        st.write(row['answer'])
        st.caption(f"Latency: {row.get('latency_s', 0)}s | Tokens: {row.get('tokens', 0)} | Model: {row.get('model', display_model(row.get('or_provider')))} | Cost: {row.get('cost', 'Rp 0')} | Intent: {row['intent']} | Time: {row['created_at']}")
        st.caption(f"Faithfulness: {row.get('faithfulness', 'N/A')}")
        

        
        # Show retrieved context if available
        retrieved_context = row.get('retrieved_context', [])
        if isinstance(retrieved_context, list) and len(retrieved_context) > 0:
            with st.expander(f"View Retrieved Context ({len(retrieved_context)} chunks)"):
                for idx, chunk in enumerate(retrieved_context):
                    score = chunk.get('score', 0.0)
                    dense = chunk.get('dense_score', 0.0)
                    sparse = chunk.get('sparse_score', 0.0)
                    st.markdown(f"**[{idx+1}] {chunk.get('course_name') or chunk.get('title') or 'Unknown'}** (Score: `{score:.4f}` | Dense: `{dense:.4f}` | Sparse: `{sparse:.4f}`)")
                    st.text(chunk.get('text', ''))
                    st.divider()
        
        # Judgment Label
        issues = []
        if pd.notna(row.get('faithfulness')) and row.get('faithfulness') is not None:
            if float(row['faithfulness']) < 0.8:
                issues.append("Faithfulness Rendah (Potensi Halusinasi)")
        
        if issues:
            st.error(f"Problematic Chat: {', '.join(issues)}")
        elif pd.notna(row.get('faithfulness')) and row.get('faithfulness') is not None:
            st.success("Healthy Chat (Faithful)")

# --- UI Layout ---

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.title("CAG Agent Observability Dashboard")
with col_refresh:
    st.write("")  # vertical alignment hack
    if st.button("🔄 Refresh", help="Hapus cache & muat ulang data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

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
tab_overview, tab_explorer, tab_ltm = st.tabs([
    "Overview & Recent Logs",
    "Session Explorer",
    "User LTM Profiles",
])

with tab_overview:
    # KPI Row
    # KPI Row 1
    col1, col2, col3, col_faith = st.columns(4)
    col1.metric("Total Queries", f"{kpis.get('total_queries', 0):,}")
    col2.metric("Avg Latency", f"{kpis.get('avg_latency', 0.0)/1000:.2f} s")
    col3.metric("OR Cache Hit Rate", f"{kpis.get('hit_rate', 0.0):.1f}%")
    _faith = kpis.get('faithfulness_avg_7d')
    _faith_n = kpis.get('faithfulness_n_7d', 0)
    _faith_fail = kpis.get('faithfulness_fail_7d', 0)
    col_faith.metric(
        "Faithfulness (7d)",
        f"{_faith:.3f}" if _faith is not None else "—",
        delta=f"-{_faith_fail} unfaithful" if _faith_fail else None,
        delta_color="inverse",
        help=f"Avg LLM-judge faithfulness over {_faith_n} evaluated turns (sampled). "
             f"{_faith_fail} scored below the {0.75} pass threshold.",
    )

    # KPI Row 2 (Aligned to center columns of the 4-column layout)
    _, col_tokens, col_cost, _ = st.columns(4)
    col_tokens.metric("OR Cached/Total Tokens", f"{kpis.get('or_cached_tokens', 0):,} / {(kpis.get('or_prompt_tokens', 0) + kpis.get('or_completion_tokens', 0)):,}")
    
    total_cost_idr = kpis.get('total_cost', 0.0) * USD_TO_IDR
    col_cost.metric("Total Cost", f"Rp {total_cost_idr:,.0f}")

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
        for col in ['faithfulness', 'tokens', 'or_cached_tokens', 'or_provider', 'rewritten_query', 'or_generation_id', 'cost']:
            if col not in df_logs.columns:
                df_logs[col] = None
                
        df_logs['cost'] = df_logs['cost'].apply(lambda x: f"Rp {x * USD_TO_IDR:,.0f}" if pd.notna(x) else "Rp 0")
        df_logs['model'] = df_logs['or_provider'].apply(display_model)
        # Calculate latency in seconds
        df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
        
        df_logs['is_cache_hit'] = df_logs['or_cached_tokens'].apply(lambda x: True if pd.notna(x) and x > 0 else False)
        df_logs['tokens_saved'] = df_logs['or_cached_tokens']
        
        # Truncate text for the table view
        df_logs['query_short'] = df_logs['query'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        df_logs['answer_short'] = df_logs['answer'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        
        # Use dataframe selection
        event = st.dataframe(
            df_logs[['created_at', 'session_id', 'intent', 'latency_s', 'tokens', 'model', 'cost', 'is_cache_hit', 'tokens_saved', 'faithfulness', 'query_short', 'answer_short']],
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
        # Defensive programming: ensure new columns exist
        for col in ['faithfulness', 'tokens', 'or_cached_tokens', 'or_provider', 'rewritten_query', 'or_generation_id', 'cost']:
            if col not in df_logs.columns:
                df_logs[col] = None

        df_logs['cost'] = df_logs['cost'].apply(lambda x: f"Rp {x * USD_TO_IDR:,.0f}" if pd.notna(x) else "Rp 0")
        df_logs['model'] = df_logs['or_provider'].apply(display_model)

        if 'created_at' in df_logs.columns:
            # DB stores UTC (DateTime(timezone=True)); show in WIB/GMT+7.
            df_logs['created_at'] = (
                pd.to_datetime(df_logs['created_at'], utc=True)
                .dt.tz_convert('Asia/Jakarta')
                .dt.strftime('%d/%m/%Y %H:%M:%S')
            )
        if 'latency_ms' in df_logs.columns:
            df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
            
        if 'or_cached_tokens' in df_logs.columns:
            df_logs['is_cache_hit'] = df_logs['or_cached_tokens'].apply(lambda x: True if pd.notna(x) and x > 0 else False)
            df_logs['tokens_saved'] = df_logs['or_cached_tokens']
        else:
            df_logs['is_cache_hit'] = False
            df_logs['tokens_saved'] = None
        
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
                    session_logs[['created_at', 'intent', 'latency_s', 'tokens', 'model', 'cost', 'is_cache_hit', 'tokens_saved', 'faithfulness', 'query_short', 'answer_short']],
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


