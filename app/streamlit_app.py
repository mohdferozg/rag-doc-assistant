"""Streamlit UI for the RAG Doc Assistant."""
import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Doc Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Doc Assistant")
st.caption("Ask questions about your documents. Grounded answers with page citations.")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded is not None:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Extracting, chunking, and embedding..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_URL}/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Loaded **{data['filename']}** ({data['chunks']} chunks)")
                    st.session_state["loaded"] = True
                else:
                    st.error(f"Upload failed: {resp.text}")

    st.divider()
    st.header("2. Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    st.caption("Higher = more context, lower = more focused.")

# --- Main: Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: page(s) {', '.join(map(str, msg['sources']))}")

# Chat input
if question := st.chat_input("Ask a question about the document..."):
    if not st.session_state.get("loaded"):
        st.warning("Please upload a document first (see sidebar).")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating..."):
                t0 = time.time()
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": top_k},
                )
                elapsed = time.time() - t0

            if resp.status_code == 200:
                data = resp.json()
                st.markdown(data["answer"])
                st.caption(
                    f"Sources: page(s) {', '.join(map(str, data['sources']))} "
                    f"· {elapsed:.1f}s"
                )

                # Expandable retrieval details
                with st.expander("🔍 Retrieved chunks (debug)"):
                    for c in data["retrieved"]:
                        st.markdown(
                            f"**{c['chunk_id']}** · page {c['page_number']} "
                            f"· score `{c['score']:.3f}`"
                        )
                        st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                })
            else:
                st.error(f"Error: {resp.text}")

# --- Footer ---
st.divider()
st.caption(
    "Built with FastAPI + Streamlit + Gemini · "
    "In-memory vector store (see README for production notes)."
)