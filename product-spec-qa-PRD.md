# Product Spec Q&A Assistant — PRD
> Eczacıbaşı Holding · Generative AI Internship Interview · June 2026

---

## 1. One-Liner

Upload product data sheet PDFs; ask in natural language — the system retrieves the relevant technical information in seconds, with the source paragraph attached.

---

## 2. Problem & Business Value

Eczacıbaşı Group's construction products and consumer goods lines span hundreds of SKUs, each backed by a technical data sheet in PDF format. The current reality:

- A sales rep must find the right PDF, open it, and ctrl+F — averaging 3–5 minutes per question.
- Technical teams waste time on routine queries that could be self-served.
- Search is keyword-based: typing "low temperature" won't surface a result that says "−10°C".
- Cross-document comparison is practically impossible.

**Solution:** A RAG (Retrieval-Augmented Generation) pipeline. Chunk PDFs, embed them with a multilingual model, build a FAISS vector index, then at query time retrieve the most relevant chunks and pass them as grounded context to an LLM.

| | Before | With This Tool |
|---|---|---|
| Time to answer | ~3–5 min (manual PDF search) | ~5 sec |
| Semantic miss | "low temperature" ≠ "−10°C" | Meaning-based retrieval — language agnostic |
| Expert dependency | Technical team required | Self-service; source paragraph always visible |
| Multi-doc compare | Not feasible | Upload N PDFs, compare in a single query |

**Real context:** Built and validated on Tan Tedarik Kimyasal supplier PDFs — the same operational environment where I manage marketplace and B2B operations on Trendyol and Hepsiburada.

---

## 3. Input Data

The pipeline supports any technical PDF. Recommended sources for the demo:

| Source | Content |
|---|---|
| VitrA Technical Specs | Bathroom products — assembly tolerances, material resistance |
| Eczacıbaşı Construction Chemicals | Paint/coating sheets — temperature, surface, application conditions |
| Tan Tedarik Supplier PDFs | Real production data — strengthens demo credibility |

Each PDF is loaded via LangChain `PyPDFLoader` and chunked with `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64). The overlap ensures answers are never cut off at a chunk boundary.

---

## 4. Architecture

Two-phase pipeline — **Indexing** (once) and **Query** (per question):

```
── INDEXING PIPELINE (offline) ────────────────────────────────
PDF Upload
    │
    ▼
PyPDFLoader  →  per-page Documents with metadata (source, page)
    │
    ▼
RecursiveCharacterTextSplitter  (chunk_size=512, overlap=64)
    │
    ▼
HuggingFace Embeddings
    paraphrase-multilingual-MiniLM-L12-v2  (local, no API cost)
    │
    ▼
FAISS Index  →  persisted to  faiss_index/

── QUERY PIPELINE (real-time) ─────────────────────────────────
User Question
    │
    ▼
Same embedding model  →  query vector
    │
    ▼
FAISS similarity_search  →  top-k chunks (k=4)
    │
    ▼
LangChain RetrievalQA chain
    │
    ▼
GPT-4o-mini (temperature=0)  →  grounded answer + source citation
```

---

## 5. Tech Stack

| Layer | Tool | Version / Model | Why |
|---|---|---|---|
| UI | Streamlit | ≥ 1.35 | Fast prototype; native file uploader |
| Doc Loading | LangChain PyPDFLoader | langchain ≥ 0.2 | Page metadata (source, page no.) auto-attached |
| Chunking | RecursiveCharacterTextSplitter | LangChain core | Hierarchical split; sentence integrity preserved |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers | TR + EN; fully local — zero API cost |
| Vector Store | FAISS | faiss-cpu 1.8 | Local persist; zero infra required |
| LLM Chain | RetrievalQA | LangChain | Ready-made chain; source docs returned automatically |
| LLM | GPT-4o-mini | openai ≥ 1.0 | Grounded answer sufficient; temperature=0 |
| Reranking (v2) | cross-encoder/ms-marco-MiniLM-L-6-v2 | HuggingFace Hub | Post-retrieval precision boost |

**Model selection rationale:** `paraphrase-multilingual-MiniLM-L12-v2` handles both Turkish and English product sheets in the same index, runs entirely locally, and costs nothing per embedding. 100 PDFs × 200 chunks = 20,000 embeddings — the cost difference vs. OpenAI's embedding API is real and material at scale.

---

## 6. Prompt Design

The `RetrievalQA` system prompt enforces two hard rules:

1. Answer only from retrieved chunks — hallucination is explicitly prohibited.
2. If the answer is not in the context, say so exactly.

```
SYSTEM PROMPT
─────────────
You are a technical product assistant for an industrial manufacturing company.
Answer questions ONLY using the provided context chunks from product data sheets.
Always cite the source document and page number.
If the answer is not in the provided context, respond exactly:
"This information is not available in the uploaded documents."
Do not infer, extrapolate, or use external knowledge.
```

**Why this constraint matters:** In an industrial context, a wrong answer to "Can this paint be applied at −10°C?" can cause a real production failure. The constraint is both a safety measure and a trust signal for the sales or technical team using the tool.

---

## 7. Full Application Code

### app.py

```python
import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH  = "faiss_index"

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a technical product assistant for an industrial manufacturing company.
Answer questions ONLY using the provided context chunks from product data sheets.
Always cite the source document and page number.
If the answer is not in the provided context, respond exactly:
"This information is not available in the uploaded documents."
Do not infer, extrapolate, or use external knowledge.

Context:
{context}

Question: {question}
Answer:"""
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_index(pdf_paths: list[str]) -> FAISS:
    docs = []
    for path in pdf_paths:
        docs.extend(PyPDFLoader(path).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(INDEX_PATH)
    return db

def load_index() -> FAISS | None:
    if Path(INDEX_PATH).exists():
        return FAISS.load_local(
            INDEX_PATH, get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return None

def make_qa_chain(db: FAISS) -> RetrievalQA:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT}
    )

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spec Q&A", page_icon="🔍", layout="wide")
st.title("🔍 Product Spec Q&A Assistant")
st.caption("Upload technical PDFs · Ask in natural language · Grounded answer + source")

with st.sidebar:
    st.header("📄 Upload & Index PDFs")
    uploaded = st.file_uploader(
        "Data sheet PDFs", type="pdf", accept_multiple_files=True
    )
    if uploaded and st.button("🔨 Build Index", type="primary"):
        paths = []
        for f in uploaded:
            tmp = f"/tmp/{f.name}"
            with open(tmp, "wb") as out:
                out.write(f.getbuffer())
            paths.append(tmp)
        with st.spinner(f"Processing {len(paths)} PDF(s)..."):
            db = build_index(paths)
            st.session_state["db"] = db
        st.success(f"✅ Index ready — {len(paths)} document(s)")

    if Path(INDEX_PATH).exists():
        if st.button("💾 Load Existing Index"):
            st.session_state["db"] = load_index()
            st.success("Index loaded")

db = st.session_state.get("db")

if db:
    qa = make_qa_chain(db)
    query = st.text_input(
        "Ask your question",
        placeholder="Can this paint be applied at −10°C?"
    )
    if query:
        with st.spinner("Searching..."):
            result = qa.invoke({"query": query})
        st.subheader("Answer")
        st.markdown(result["result"])
        with st.expander("📚 Source Chunks"):
            for doc in result["source_documents"]:
                st.caption(
                    f"📄 {doc.metadata.get('source', '?')} · "
                    f"Page {doc.metadata.get('page', '?')}"
                )
                st.text(doc.page_content[:400] + "...")
else:
    st.info("Upload PDFs or load an existing index from the sidebar to get started.")
```

### requirements.txt

```
openai>=1.0.0
streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-huggingface>=0.0.3
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pypdf>=4.0.0
torch>=2.0.0
```

---

## 8. Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Run
streamlit run app.py

# 4. Demo flow
# Sidebar → upload PDFs → "Build Index"
# Ask: "Can this product be applied at −10°C?"
# → Answer + source page number
```

---

## 9. Interview Talking Points

**"Why did you build this?"**
> "In my first project (Trendyol Feedback Intelligence) I built an unstructured text → structured insight pipeline. RAG is a different problem class: semantic search over domain-specific technical documentation. Eczacıbaşı's construction and consumer goods portfolio has hundreds of data sheets — this directly solves a real operational pain."

**"Why LangChain?"**
> "RetrievalQA, document loaders, and text splitters are composable and swappable. I can replace FAISS with ChromaDB or swap GPT-4o-mini for a local Llama without touching application logic. LangChain's abstraction layer is what makes that flexibility cheap."

**"Why HuggingFace embeddings instead of OpenAI?"**
> "The multilingual MiniLM runs entirely locally — zero API cost per embedding. 100 PDFs × 200 chunks = 20,000 embedding calls. With OpenAI's text-embedding-3-small that's a real cost at scale. Local model is production-grade quality; the speed-cost tradeoff clearly favors local here."

**"How did you handle hallucination risk?"**
> "Two controls: a hard prompt constraint ('Answer ONLY from context; if not found, say so') and temperature=0. Source chunks are always shown to the user — they can verify the answer against the document themselves. In an industrial context, a wrong answer to a technical spec question can cause a production failure. That tradeoff was a conscious design decision, not an afterthought."

**"How does this fit Eczacıbaşı?"**
> "50+ companies, 120+ international markets, product lines across construction, consumer, and health. Every line has technical data sheets. Sales, export, and customer service teams could use this self-service. Straightforward extension: all documents in one index with a company filter for scope control."

---

## 10. Extension Ideas

*If asked about next steps:*

- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` for a precision boost after top-k retrieval — one extra line from HuggingFace Hub.
- **Multi-doc comparison:** "What's the application temperature difference between Product A and Product B?" — the retriever pulls chunks from both sources naturally.
- **Conversation memory:** `ConversationalRetrievalChain` to maintain context across follow-up questions.
- **Incremental indexing CLI:** New data sheet arrives → add to existing index without a full rebuild.
- **Hepsiburada / B2B portal adapter:** Same RAG core, different data source — a natural next step for Tan Tedarik's B2B operations.
- **OpenAI Batch API:** For large initial indexing jobs, batch embedding requests cut cost ~50%.

---

*Built by Güngör Efe Çetin · github.com/gungorefecetin · linkedin.com/in/gungorefecetin*
*Validated on real supplier data — Tan Tedarik Kimyasal · June 2026*
