import os
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
tab_overview, tab_explorer, tab_ltm = st.tabs(["Overview & Recent Logs", "Session Explorer", "User LTM Profiles"])

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
    st.subheader("Recent Chat Logs Viewer")
    st.markdown("Pilih salah satu baris di bawah ini untuk melihat percakapan singkat.")

    if not logs:
        st.info("Belum ada data log.")
    else:
        df_logs = pd.DataFrame(logs)
        if 'created_at' in df_logs.columns:
            df_logs['created_at'] = pd.to_datetime(df_logs['created_at']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
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
            df_logs['created_at'] = pd.to_datetime(df_logs['created_at']).dt.strftime('%d/%m/%Y %H:%M:%S')
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
