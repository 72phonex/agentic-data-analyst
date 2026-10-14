import os, io, streamlit as st, pandas as pd
from tools import clean_dataframe, describe_dataset
from agent import ask_agent
from memory import load_memories, save_memory, delete_memory, find_relevant_memories
from llm import OLLAMA_MODEL

st.set_page_config(page_title="Data Analyst Chatbot", page_icon="📊", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Read uploaded file into a pandas DataFrame and clean numeric columns."""
    if file_name.endswith(".tsv"):
        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    return clean_dataframe(df)

# Initialize reload-safe session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_memory" not in st.session_state:
    st.session_state.pending_memory = None

# Sidebar: Configuration, upload, summary, and memory
with st.sidebar:
    st.title("⚙️ Controls")
    model_name = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    st.caption(f"Active Model: **{model_name}**")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "tsv", "xlsx", "xls"])
    
    # Optional sample data loader for quick student demos
    sample_choice = st.selectbox("Or try synthetic sample dataset:", ["None", "sales_transactions.csv", "student_performance.csv", "health_survey.csv"])
    
    df = None
    if uploaded_file:
        df = load_data(uploaded_file.getvalue(), uploaded_file.name)
        st.success(f"Loaded `{uploaded_file.name}`")
    elif sample_choice != "None":
        sample_path = os.path.join("sample_data", sample_choice)
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                df = load_data(f.read(), sample_choice)
            st.info(f"Loaded synthetic demo `{sample_choice}`")

    if df is not None:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
        with st.expander("View Column Names"):
            st.write(list(df.columns))

    st.divider()
    st.subheader("🧠 Business Rules Memory")
    saved_memories = load_memories()
    if saved_memories:
        for m in saved_memories:
            col_txt, col_del = st.columns([0.8, 0.2])
            col_txt.caption(f"• {m['text']}")
            if col_del.button("✕", key=f"del_{m['id']}", help="Delete rule"):
                delete_memory(m["id"])
                st.rerun()
    else:
        st.caption("No rules saved. Type `remember: <rule>` in chat to save one.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Area: Header and Chat
st.header("📊 Agentic Data Analyst")

if df is None:
    st.info("👋 Welcome! Please upload a CSV, TSV, or Excel file in the sidebar to begin analyzing your data.")
else:
    # Handle pending memory confirmation if user typed "remember: ..."
    if st.session_state.pending_memory:
        st.warning(f"Do you want to save this rule to memory?\n\n**'{st.session_state.pending_memory}'**")
        c1, c2 = st.columns([0.2, 0.8])
        if c1.button("💾 Save Rule"):
            save_memory(st.session_state.pending_memory)
            st.session_state.pending_memory = None
            st.success("Rule saved to memory!")
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.pending_memory = None
            st.rerun()

    # Display chat message history with tool traces and charts
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
                st.image(msg["chart_path"])
            if msg.get("traces"):
                with st.expander("🛠️ Tool Trace & Raw Results", expanded=False):
                    for trace in msg["traces"]:
                        st.markdown(f"**Tool:** `{trace['tool']}`")
                        st.caption("Arguments:")
                        st.json(trace.get("args", {}))
                        st.caption("Output:")
                        st.json(trace.get("result", {}))

    # Chat input box
    if prompt := st.chat_input("Ask a question about your data..."):
        if prompt.strip().lower().startswith("remember:"):
            # Set confirmation state for user memory addition
            st.session_state.pending_memory = prompt.strip()[9:].strip()
            st.rerun()
        else:
            # Display user question
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Retrieve only keywords matching saved memory
            relevant_rules = find_relevant_memories(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data with tools..."):
                    result = ask_agent(df, prompt, chat_history=st.session_state.messages[:-1], memories=relevant_rules)
                st.write(result["answer"])
                if result.get("chart_path") and os.path.exists(result["chart_path"]):
                    st.image(result["chart_path"])
                if result.get("traces"):
                    with st.expander("🛠️ Tool Trace & Raw Results", expanded=False):
                        for trace in result["traces"]:
                            st.markdown(f"**Tool:** `{trace['tool']}`")
                            st.caption("Arguments:")
                            st.json(trace.get("args", {}))
                            st.caption("Output:")
                            st.json(trace.get("result", {}))

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "traces": result.get("traces", []),
                "chart_path": result.get("chart_path")
            })
