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
def fetch_dashboard_data(limit=100):
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

data = fetch_dashboard_data(limit=100)

if not data:
    st.stop()

kpis = data.get("kpis", {})
intents = data.get("intents", [])
trends = data.get("trends", [])
logs = data.get("logs", [])

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
st.markdown("Pilih salah satu baris di bawah ini untuk melihat percakapan lengkap.")

if not logs:
    st.info("Belum ada data log.")
else:
    df_logs = pd.DataFrame(logs)
    # Truncate text for the table view
    df_logs['query_short'] = df_logs['query'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
    df_logs['answer_short'] = df_logs['answer'].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x)
    
    # Use dataframe selection
    event = st.dataframe(
        df_logs[['created_at', 'intent', 'latency_ms', 'cache_hit', 'query_short', 'answer_short']],
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
        st.caption(f"Waktu: {selected_log['created_at']} | Intent: {selected_log['intent']} | Latency: {selected_log['latency_ms']}ms")
        
        with st.chat_message("user"):
            st.write(selected_log['query'])
            
        with st.chat_message("assistant"):
            st.write(selected_log['answer'])
