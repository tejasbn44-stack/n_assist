# Knowledge Assistant — Domain-Specific RAG Chatbot

A compliant chatbot that answers questions **only** from your own documents.
Built with - 
  LangChain 
  FAISS 
  HuggingFace Embeddings 
  OpenAI 
  Streamlit

---

## Project Structure

```
rag_chatbot/
├── app.py               ← Main Streamlit app (all logic lives here)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── docs/                ← Drop your .txt or .pdf files here
    ├── university_handbook.txt
    └── tech_faq.txt
```

---

## How It Works (Architecture)

```
User Question
     │
     ▼
┌─────────────────┐
│  FAISS Vector   │  ← Searches for the 4 most relevant 500-char chunks
│  Store (Index)  │
└────────┬────────┘
         │  retrieved context
         ▼
┌─────────────────┐
│  Compliance     │  ← System prompt enforces "only use this context"
│  Prompt         │
└────────┬────────┘
         │  prompt + context + question
         ▼
┌─────────────────┐
│  OpenAI LLM     │  ← Generates the final answer
│  (gpt-3.5-turbo)│
└────────┬────────┘
         │
         ▼
    Answer + Source Passages shown in Streamlit UI
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your documents
Place `.txt` or `.pdf` files inside the `docs/` folder.
Two sample documents are already included for testing.

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Enter your OpenAI API key
Paste your key in the sidebar (it's never saved to disk).

---

## Configuration (in app.py)

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 60 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Local HuggingFace model |
| `LLM_MODEL` | gpt-3.5-turbo | OpenAI model |
| `MAX_RETRIEVAL_DOCS` | 4 | Chunks retrieved per query |
| `TEMPERATURE` | 0.0 | 0 = deterministic, 1 = creative |

---

## Compliance Behaviour

The bot is engineered to never hallucinate:

| Scenario | Bot Response |
|---|---|
| Answer exists in docs | Answers with citation |
| Answer not in docs | "I am sorry, that information is not in my database." |
| Off-topic question | Refuses and redirects |

This is enforced by the system prompt in `SYSTEM_TEMPLATE` inside `app.py`.

---

## Submission Checklist

- [x] `docs/` — source documents folder
- [x] `app.py` — LangChain pipeline connecting data to AI
- [x] `SYSTEM_TEMPLATE` — compliance system prompt (inside app.py)
- [x] Streamlit UI with `st.chat_input`, `st.chat_message`, sidebar controls
