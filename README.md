# Product Spec Q&A Assistant

Upload product data sheet PDFs. Ask questions in natural language — Turkish or English. Get a grounded answer with the exact source paragraph, in seconds.

---

## The problem

A sales rep answering "Can this product be applied at −10°C?" opens the right PDF, ctrl-F's the temperature, finds nothing because the spec says "−15°C minimum" rather than "cold temperature", and gives up or escalates to a technical expert. That's 3–5 minutes per query, repeated across hundreds of SKUs.

**This tool makes it 5 seconds and self-service — with the source paragraph always visible so the answer can be verified.**

---

## Architecture

```
── INDEXING (once per document set) ──────────────────────────────
PDF Upload
    │
    ▼
PyPDFLoader  →  per-page Documents  →  filter image-only pages
    │
    ▼
Header enrichment  →  prepend section header to each chunk
    │
    ▼
RecursiveCharacterTextSplitter  (chunk_size=512, overlap=128)
    │
    ▼
paraphrase-multilingual-MiniLM-L12-v2  (local, no API cost)
    │
    ▼
FAISS index  →  persisted to  faiss_index/<timestamp>/

── QUERY (per question) ──────────────────────────────────────────
User question
    │
    ▼
Same embedding model  →  query vector
    │
    ▼
FAISS similarity_search  (k=12 candidates)
    │
    ▼
CrossEncoder reranker  (ms-marco-MiniLM-L-6-v2)  →  top-5 chunks
    │
    ▼
GPT-4o-mini  (temperature=0)  →  grounded answer + source citation
```

### Two-stage retrieval

FAISS retrieves 12 candidates by embedding similarity — high recall, imprecise ranking. The cross-encoder reranker scores each `(query, chunk)` pair directly and filters to the top 5 — high precision, runs only on the 12 candidates so it stays fast. The LLM sees only the 5 best chunks.

### Multilingual by design

`paraphrase-multilingual-MiniLM-L12-v2` maps Turkish and English into the same vector space. A query in English retrieves the right paragraph from a Turkish document without any translation step.

### Hallucination control

The system prompt prohibits the LLM from answering outside the retrieved context. If the answer is not in the documents, it responds with an exact sentinel phrase. `temperature=0` makes output deterministic.

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF loading | LangChain `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Vector store | FAISS (local, disk-persisted) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | GPT-4o-mini via OpenAI API |

---

## Quickstart

```bash
# 1. Clone and create environment
git clone https://github.com/gungorefecetin/product-qa.git
cd product-qa
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# edit .env and add your key: OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

Then:
- Sidebar → upload one or more product data sheet PDFs → **Build Index**
- Ask a question in the main panel
- The answer cites the source document and page number
- Expand **Source Chunks** to verify the answer against the raw text

---

## Key design decisions

**Why local embeddings?** 100 PDFs × 200 chunks = 20,000 embeddings per index build. The local multilingual MiniLM runs at zero API cost and is production-grade quality for retrieval. The only API call is the single LLM inference per query.

**Why FAISS and not a cloud vector DB?** Zero infrastructure for a pilot. The index is persisted to a timestamped slot on disk — rollback is a symlink change. Swapping to ChromaDB or Pinecone is one constructor change in LangChain.

**Why the reranker?** Embedding similarity measures geometric distance in vector space, not query intent. The cross-encoder reads the full `(query, chunk)` pair and scores relevance directly. In testing, it moved the correct chunk from rank 3 to rank 1 on product comparison queries — the difference between a right answer and a wrong one.

**Why GPT-4o-mini?** The hard work is done by retrieval and reranking. The LLM synthesises 5 pre-filtered chunks into a cited answer. gpt-4o-mini does this reliably at ~1/20th the cost of gpt-4o.

---

## Extension ideas

- **Incremental indexing** — `db.add_documents()` on new data sheets without a full rebuild
- **Conversation memory** — `ConversationalRetrievalChain` for follow-up questions
- **Company/product-line filter** — metadata field on chunks for scoped retrieval
- **OCR pre-processing** — Textract or Tesseract for scanned (image-only) PDFs
- **OpenAI Batch API** — ~50% cost reduction on large initial indexing jobs

---

*Built by Güngör Efe Çetin · [github.com/gungorefecetin](https://github.com/gungorefecetin) · [linkedin.com/in/gungorefecetin](https://linkedin.com/in/gungorefecetin)*
