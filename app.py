"""
Streamlit Web UI for Tabular Data Analysis Agent.
Features:
  - Universal CSV/Excel uploader
  - Interactive multi-turn chat
  - Transparent tool trace & intermediate computational outputs
  - Inline chart display
  - Persistent Memory Inspector & Management sidebar
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Add repo root to path
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_agent.core.agent import DataAnalysisAgent
from data_agent.core.memory import MemoryStore
from data_agent.core.llm import LLMClient
from data_agent.core.data_loader import load_dataset


st.set_page_config(
    page_title="Tabular Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize Session State
if "memory_store" not in st.session_state:
    st.session_state.memory_store = MemoryStore("data_agent_memory.sqlite")

if "backend" not in st.session_state:
    st.session_state.backend = os.getenv("LLM_BACKEND", "heuristic")

if "agent" not in st.session_state:
    llm = LLMClient(backend=st.session_state.backend)
    st.session_state.agent = DataAnalysisAgent(
        memory_store=st.session_state.memory_store,
        llm_client=llm
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_df" not in st.session_state:
    st.session_state.current_df = None
    st.session_state.filename = None


# ==============================================================================
# SIDEBAR: DATA UPLOAD & MEMORY MANAGER
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")

    # 1. Dataset Upload
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "tsv"])

    # Or pick a demo dataset
    demo_choice = st.selectbox(
        "Or choose a demo dataset:",
        ["None", "data/sales_transactions.csv", "data/student_performance.csv", "data/health_survey.csv"]
    )

    if uploaded_file is not None:
        if st.session_state.filename != uploaded_file.name:
            df, meta = load_dataset(uploaded_file)
            st.session_state.current_df = df
            st.session_state.filename = uploaded_file.name
            st.session_state.agent.set_dataset(df, uploaded_file.name)
            st.success(f"Loaded: `{uploaded_file.name}` ({len(df):,} rows)")
    elif demo_choice != "None" and os.path.exists(demo_choice):
        if st.session_state.filename != demo_choice:
            df, meta = load_dataset(demo_choice)
            st.session_state.current_df = df
            st.session_state.filename = demo_choice
            st.session_state.agent.set_dataset(df, demo_choice)
            st.success(f"Loaded: `{os.path.basename(demo_choice)}` ({len(df):,} rows)")

    if st.session_state.current_df is not None:
        df = st.session_state.current_df
        st.write(f"**Shape**: `{df.shape[0]:,} rows × {df.shape[1]} cols`")
        with st.expander("🔍 Column Details", expanded=False):
            st.write(list(df.columns))

    st.markdown("---")

    # 2. LLM Backend Selector
    st.subheader("🧠 LLM Configuration")
    backend_opt = st.selectbox(
        "Model Backend",
        ["heuristic", "gemini", "ollama"],
        index=["heuristic", "gemini", "ollama"].index(st.session_state.backend)
    )
    if backend_opt != st.session_state.backend:
        st.session_state.backend = backend_opt
        st.session_state.agent.llm = LLMClient(backend=backend_opt)
        st.info(f"Switched backend to: {backend_opt.upper()}")

    st.markdown("---")

    # 3. Persistent Memory Inspector
    st.subheader("💾 Persistent Memory Store")
    st.caption("Curated domain facts, metric definitions, and preferences stored in SQLite.")

    memories = st.session_state.memory_store.list_memories()
    if memories:
        for m in memories:
            with st.container():
                st.markdown(f"**#{m['id']} [{m['category']}] `{m['key']}`**")
                st.code(m['value'], language="sql")
                col_del, col_date = st.columns([1, 2])
                with col_del:
                    if st.button("Forget", key=f"del_{m['id']}", type="secondary"):
                        st.session_state.memory_store.delete_memory(m['id'])
                        st.rerun()
                with col_date:
                    st.caption(f"Updated: {m['updated_at']}")
                st.divider()
    else:
        st.write("No memories stored yet.")

    # Manual Add Form
    with st.expander("➕ Add Memory Manually", expanded=False):
        with st.form("manual_add_mem"):
            cat = st.selectbox("Category", ["metric_definition", "column_meaning", "filter", "preference", "user_correction"])
            k = st.text_input("Key (e.g. active customers)")
            v = st.text_input("Value (e.g. status == 'Active' and cancelled_flag == 0)")
            submit = st.form_submit_button("Save Memory")
            if submit and k and v:
                res = st.session_state.memory_store.save_memory(category=cat, key=k, value=v)
                st.success(res["message"])
                st.rerun()


# ==============================================================================
# MAIN VIEW: CHAT INTERFACE & TOOL EXECUTION TRACE
# ==============================================================================

st.title("🤖 Tabular Data Analysis Agent")
st.markdown("Ask natural-language questions about your data. The agent computes answers using deterministic tools and recalls your saved definitions.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display tool execution box if present
        if msg.get("tool_used"):
            with st.expander(f"🛠️ Tool Trace: `{msg['tool_used']}`", expanded=False):
                if msg.get("memories_used"):
                    st.info(f"🧠 **Memories Applied**: {', '.join(msg['memories_used'])}")
                st.caption(f"**Arguments**: `{json.dumps(msg.get('tool_arguments', {}), default=str)}`")
                inter = msg.get("intermediate_result", {})
                if isinstance(inter, dict) and "data" in inter and isinstance(inter["data"], list):
                    st.dataframe(pd.DataFrame(inter["data"]))
                else:
                    st.json(inter)

        # Display chart if generated
        if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
            st.image(msg["chart_path"], caption=msg.get("chart_title", "Generated Chart"))

# Chat Input
prompt = st.chat_input("Ask a question or provide a definition...")

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            res = st.session_state.agent.ask(prompt)

        st.markdown(res["answer"])

        # Check for chart
        chart_path = None
        chart_title = None
        inter = res.get("intermediate_result")
        if isinstance(inter, dict) and inter.get("saved_image_path"):
            chart_path = inter["saved_image_path"]
            chart_title = inter.get("title", "Generated Chart")
            if os.path.exists(chart_path):
                st.image(chart_path, caption=chart_title)

        # Show tool trace
        if res.get("tool_used"):
            with st.expander(f"🛠️ Tool Trace: `{res['tool_used']}`", expanded=True):
                if res.get("memories_used"):
                    st.info(f"🧠 **Memories Applied**: {', '.join(res['memories_used'])}")
                st.caption(f"**Arguments**: `{json.dumps(res.get('tool_arguments', {}), default=str)}`")
                if isinstance(inter, dict) and "data" in inter and isinstance(inter["data"], list):
                    st.dataframe(pd.DataFrame(inter["data"]))
                else:
                    st.json(inter)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "tool_used": res.get("tool_used"),
        "tool_arguments": res.get("tool_arguments"),
        "intermediate_result": res.get("intermediate_result"),
        "memories_used": res.get("memories_used"),
        "chart_path": chart_path,
        "chart_title": chart_title
    })
