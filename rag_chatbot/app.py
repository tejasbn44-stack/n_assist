"""
Knowledge Assistant — Domain-Specific RAG Chatbot
--------------------------------------------------
A compliant chatbot that answers only from your uploaded documents.
Built with: LangChain · FAISS · HuggingFace Embeddings · OpenAI · Streamlit
"""

import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DOCS_DIR = Path("docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0          # deterministic — no creative hallucination
MAX_RETRIEVAL_DOCS = 4     # how many chunks to retrieve per query


# ─────────────────────────────────────────────
# SYSTEM PROMPT (Compliance + Few-Shot)
# ─────────────────────────────────────────────
SYSTEM_TEMPLATE = """You are a specialized Knowledge Assistant.
Your ONLY job is to answer questions using the document context provided below.

Rules you must NEVER break:
1. Use ONLY the provided context to answer. Do not use any outside knowledge.
2. If the answer is not in the context, respond EXACTLY with:
   "I am sorry, that information is not in my database."
3. Never speculate, guess, or extrapolate beyond the provided text.
4. Cite the relevant section or document when possible.

---
Few-Shot Examples:

User: What is the late submission policy?
Context: "Assignments submitted after the deadline will incur a 10% penalty per day."
Good Answer: According to the handbook, assignments submitted after the deadline
incur a 10% penalty per day.

User: What is the capital of France?
Context: [No relevant passage about France's capital]
Correct Refusal: I am sorry, that information is not in my database.

---
Context retrieved from your documents:
{context}
"""

HUMAN_TEMPLATE = "{question}"


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_vector_store(docs_path: str) -> FAISS:
    """
    Load documents → chunk → embed → store in FAISS index.
    Cached so it only rebuilds when the app restarts.
    """
    # 1. LOAD — supports .txt and .pdf files
    loaders = {
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
    }

    all_docs = []
    docs_dir = Path(docs_path)

    for pattern, LoaderClass in loaders.items():
        for file_path in docs_dir.glob(pattern):
            try:
                loader = LoaderClass(str(file_path))
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")

    if not all_docs:
        return None

    # 2. CHUNK — 500-char pieces with 60-char overlap so context isn't cut mid-sentence
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # 3. EMBED — HuggingFace model runs locally, no API key needed for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. VECTOR STORE — FAISS in-memory index for fast similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_chain(vector_store: FAISS, memory: ConversationBufferMemory):
    """
    Connect: Vector Store → Retriever → Compliance Prompt → LLM → Chain
    """
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=st.session_state.get("openai_key", ""),
    )

    # Compliance prompt assembly
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": MAX_RETRIEVAL_DOCS},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

def setup_page():
    st.set_page_config(
        page_title="Knowledge Assistant",
        page_icon="📚",
        layout="wide",
    )
    # Custom CSS
    st.markdown("""
    <style>
      .main-header { font-size: 1.6rem; font-weight: 600; color: #1a1a2e; }
      .sub-header  { font-size: 0.9rem; color: #666; margin-top: -8px; }
      .source-card {
        background: #f8f9ff;
        border-left: 3px solid #4361ee;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #444;
        margin-top: 4px;
      }
      .chunk-text { color: #333; font-style: italic; }
      .status-ok   { color: #2d6a4f; background: #d8f3dc; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
      .status-err  { color: #9d0208; background: #ffccd5; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(vector_store):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_key", ""),
            help="Your key is only stored in session memory — never saved to disk.",
        )
        if api_key:
            st.session_state["openai_key"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        st.divider()
        st.markdown("## 📂 Knowledge Base")

        # Document upload
        uploaded = st.file_uploader(
            "Upload new documents",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Supports .txt and .pdf files",
        )
        
        if uploaded and not st.session_state.get("uploaded_done"):
            for f in uploaded:
                dest = DOCS_DIR / f.name
                with open(dest, "wb") as out:
                    out.write(f.read())
            st.cache_resource.clear()
            st.session_state["uploaded_done"] = True
            st.rerun()

        if not uploaded:
            st.session_state["uploaded_done"] = False

        # Show current docs
        existing = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
        if existing:
            st.markdown(f"**{len(existing)} document(s) loaded:**")
            for doc in existing:
                st.markdown(f"- 📄 `{doc.name}`")
        else:
            st.warning("No documents found in `docs/` folder.")

        # Vector store status
        st.divider()
        if vector_store:
            n = vector_store.index.ntotal
            st.markdown(f'<span class="status-ok">✓ Index ready · {n} chunks</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ No index — add documents</span>', unsafe_allow_html=True)

        st.divider()
        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            st.rerun()

        st.markdown("---")
        st.caption("Knowledge Assistant v1.0\nBuilt with LangChain + FAISS")


def main():
    setup_page()

    # ── State init ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    # ── Build vector store ──
    DOCS_DIR.mkdir(exist_ok=True)
    with st.spinner("🔍 Indexing documents…"):
        vector_store = build_vector_store(str(DOCS_DIR))

    # ── Sidebar ──
    render_sidebar(vector_store)

    # ── Main area ──
    st.markdown('<div class="main-header">📚 Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Answers drawn exclusively from your documents</div>', unsafe_allow_html=True)
    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show source chunks for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Source passages used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        fname = Path(src.metadata.get("source", "unknown")).name
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>#{i} — {fname}</strong><br>'
                            f'<span class="chunk-text">"{src.page_content[:300]}…"</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Chat input ──
    if prompt := st.chat_input("Ask a question about your documents…"):
        if not st.session_state.get("openai_key"):
            st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
            st.stop()
        if not vector_store:
            st.error("⚠️ No documents indexed. Please upload documents in the sidebar.")
            st.stop()

        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run chain
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                chain = build_chain(vector_store, st.session_state.memory)
                result = chain({"question": prompt})

            answer = result["answer"]
            sources = result.get("source_documents", [])

            st.markdown(answer)

            if sources:
                with st.expander("📎 Source passages used", expanded=False):
                    for i, src in enumerate(sources, 1):
                        fname = Path(src.metadata.get("source", "unknown")).name
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>#{i} — {fname}</strong><br>'
                            f'<span class="chunk-text">"{src.page_content[:300]}…"</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
