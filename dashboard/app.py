import os
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Agent Observability",
    page_icon="🤖",
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

# --- UI Layout ---

st.title("🤖 Agent Observability Dashboard")
st.markdown("Dashboard ini mengambil data via REST API FastAPI secara aman dan ringan.")

data = fetch_dashboard_data(limit=500)

if not data:
    st.stop()

kpis = data.get("kpis", {})
intents = data.get("intents", [])
trends = data.get("trends", [])
logs = data.get("logs", [])

# Setup tabs
tab_overview, tab_explorer = st.tabs(["📊 Overview & Recent Logs", "🕵️ Session Explorer"])

with tab_overview:
    # KPI Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", f"{kpis.get('total_queries', 0):,}")
    col2.metric("Avg Latency", f"{kpis.get('avg_latency', 0.0)/1000:.2f} s")
    col3.metric("Cache Hit Rate", f"{kpis.get('hit_rate', 0.0):.1f}%")

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
    st.subheader("📝 Recent Chat Logs Viewer")
    st.markdown("Pilih salah satu baris di bawah ini untuk melihat percakapan singkat.")

    if not logs:
        st.info("Belum ada data log.")
    else:
        df_logs = pd.DataFrame(logs)
        # Calculate latency in seconds
        df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
        
        # Truncate text for the table view
        df_logs['query_short'] = df_logs['query'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        df_logs['answer_short'] = df_logs['answer'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
        
        # Use dataframe selection
        event = st.dataframe(
            df_logs[['created_at', 'session_id', 'intent', 'latency_s', 'tokens', 'retrieved', 'cache_hit', 'query_short', 'answer_short']],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        selected_rows = event.selection.rows
        if selected_rows:
            idx = selected_rows[0]
            selected_log = df_logs.iloc[idx]
            
            st.markdown("### 💬 Chat Preview")
            st.caption(f"Sesi: {selected_log.get('session_id', 'Unknown')} | Waktu: {selected_log['created_at']} | Intent: {selected_log['intent']} | Latency: {selected_log['latency_s']}s | Tokens: {selected_log.get('tokens', 0)}")
            
            with st.chat_message("user"):
                st.write(selected_log['query'])
                
            with st.chat_message("assistant"):
                st.write(selected_log['answer'])

with tab_explorer:
    st.subheader("🕵️ Eksplorasi Riwayat Sesi")
    st.markdown("Pilih **Session ID** untuk melihat urutan percakapan secara kronologis.")
    
    if not logs:
        st.info("Belum ada data log.")
    else:
        df_logs = pd.DataFrame(logs)
        if 'latency_ms' in df_logs.columns:
            df_logs['latency_s'] = df_logs['latency_ms'].apply(lambda x: round(x / 1000.0, 2))
        
        # Get unique sessions and their latest activity time
        if 'session_id' in df_logs.columns:
            session_info = df_logs.groupby('session_id')['created_at'].max().reset_index()
            session_info = session_info.sort_values('created_at', ascending=False)
            unique_sessions = session_info['session_id'].tolist()
            
            if unique_sessions:
                selected_session = st.selectbox("Pilih Session ID:", unique_sessions)
                
                # Filter logs for selected session and sort chronologically (oldest first)
                session_logs = df_logs[df_logs['session_id'] == selected_session].sort_values('created_at', ascending=True)
                
                st.markdown(f"**Total percakapan:** {len(session_logs)} giliran")
                st.divider()
                
                for _, row in session_logs.iterrows():
                    # Render User Message
                    with st.chat_message("user"):
                        st.write(row['query'])
                        
                    # Render Assistant Message
                    with st.chat_message("assistant"):
                        st.write(row['answer'])
                        st.caption(f"⏱️ {row.get('latency_s', 0)}s | 🪙 {row.get('tokens', 0)} tokens | 📚 {row.get('retrieved', 0)} docs | 🧠 Intent: {row['intent']} | 📅 {row['created_at']}")
