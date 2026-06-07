from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_ROOT  = Path("faiss_index")
LATEST_LINK = INDEX_ROOT / "latest"

NOT_FOUND_SENTINEL = "The requested information is not available in the uploaded documents."

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a technical product assistant for an industrial manufacturing company.
Answer questions ONLY using the provided context chunks from product data sheets.
Read ALL context chunks carefully — the product name and its properties may appear in different chunks.
Rules:
- Always state the exact product name before describing it.
- If multiple products match the question, list all of them with their names and key properties.
- Always cite the source page number in your answer.
- Reply in the same language the user asked in.
- If the answer is genuinely not present anywhere in the context, respond with this exact sentence: The requested information is not available in the uploaded documents.
- Do not infer, extrapolate, or use external knowledge.

Context:
{context}

Question: {question}
Answer:""",
)


RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N  = 5   # chunks passed to LLM after reranking
FAISS_FETCH_K = 12  # wider FAISS recall; reranker filters down to RERANK_TOP_N


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return docs
    pairs  = [(query, d.page_content) for d in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]


def build_index(pdf_paths: list[str]) -> tuple[FAISS, int, list[dict]]:
    docs = []
    doc_meta: list[dict] = []

    for path in pdf_paths:
        try:
            loaded = PyPDFLoader(path).load()
            # Drop image-only pages that yield no text — they add noise, not signal
            text_pages = [p for p in loaded if p.page_content.strip()]
            if not text_pages:
                st.warning(f"Skipped {Path(path).name}: no extractable text (scanned image PDF?).")
                continue
            docs.extend(text_pages)
            doc_meta.append({
                "name": Path(path).name,
                "pages": len(text_pages),
            })
        except Exception as exc:
            st.warning(f"Skipped {Path(path).name}: {exc}")

    if not docs:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs. "
            "Are they scanned image-only files?"
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    raw_chunks = splitter.split_documents(docs)

    if not raw_chunks:
        raise ValueError("Chunking produced zero segments. Check PDF content.")

    # Prepend the first non-empty line of each page (usually the section/product header)
    # to every chunk from that page so retrieval doesn't confuse same-page products
    # that share similar vocabulary but belong to different categories.
    page_headers: dict[tuple, str] = {}
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in page_headers:
            first_line = next(
                (ln.strip() for ln in doc.page_content.splitlines() if ln.strip()), ""
            )
            page_headers[key] = first_line

    chunks = []
    for chunk in raw_chunks:
        key    = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        header = page_headers.get(key, "")
        # Only prepend if the header isn't already in the chunk (avoids duplication)
        if header and header not in chunk.page_content:
            enriched = f"[{header}]\n{chunk.page_content}"
        else:
            enriched = chunk.page_content
        chunks.append(Document(page_content=enriched, metadata=chunk.metadata))

    db = FAISS.from_documents(chunks, get_embeddings())

    slot = INDEX_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    slot.mkdir(parents=True, exist_ok=True)
    db.save_local(str(slot))

    if LATEST_LINK.is_symlink() or LATEST_LINK.exists():
        LATEST_LINK.unlink()
    LATEST_LINK.symlink_to(slot.name)

    return db, len(chunks), doc_meta


def load_index() -> FAISS | None:
    # is_symlink() is True even for broken links; exists() is False for broken ones
    if not LATEST_LINK.exists():
        if LATEST_LINK.is_symlink():
            st.error("Index symlink is broken — the slot it points to was deleted. "
                     "Please rebuild the index.")
        return None
    return FAISS.load_local(
        str(LATEST_LINK), get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def infer_doc_meta_from_index() -> list[dict]:
    """Best-effort: derive indexed doc names from the FAISS docstore."""
    db: FAISS | None = st.session_state.get("db")
    if db is None:
        return []
    seen: dict[str, int] = {}
    for doc in db.docstore._dict.values():
        name = Path(doc.metadata.get("source", "unknown")).name
        seen[name] = seen.get(name, 0) + 1
    return [{"name": n, "pages": "?"} for n in seen]


@st.cache_resource
def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def run_query(query: str, db: FAISS) -> dict:
    """FAISS → cross-encoder rerank → LLM. Returns {result, source_documents}."""
    # Step 1: broad FAISS retrieval
    candidates = db.similarity_search(query, k=FAISS_FETCH_K)

    # Step 2: cross-encoder reranking — precision over recall
    top_docs = rerank(query, candidates, top_n=RERANK_TOP_N)

    # Step 3: format context and call LLM
    context = "\n\n".join(d.page_content for d in top_docs)
    prompt  = SYSTEM_PROMPT.format(context=context, question=query)
    response = get_llm().invoke(prompt)

    return {
        "result":           response.content,
        "source_documents": top_docs,
    }


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Spec Q&A", page_icon="🔍", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Upload & Index PDFs")

    uploaded = st.file_uploader(
        "Data sheet PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded and st.button("🔨 Build Index", type="primary"):
        paths = []
        # Unique prefix prevents two files with the same name colliding in /tmp
        session_tmp = Path(f"/tmp/specqa_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
        session_tmp.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            tmp = session_tmp / f.name
            tmp.write_bytes(f.getbuffer())
            paths.append(str(tmp))
        with st.spinner(f"Processing {len(paths)} PDF(s)…"):
            try:
                db, n_chunks, doc_meta = build_index(paths)
                st.session_state["db"]       = db
                st.session_state["n_chunks"] = n_chunks
                st.session_state["doc_meta"] = doc_meta
                st.success(f"✅ Index ready — {len(paths)} doc(s), {n_chunks} chunks")
            except ValueError as exc:
                st.error(str(exc))

    index_exists = LATEST_LINK.exists()

    if index_exists:
        col_load, col_clear = st.columns(2)

        with col_load:
            if st.button("💾 Load Index", use_container_width=True):
                with st.spinner("Loading…"):
                    db = load_index()
                    if db:
                        st.session_state["db"]       = db
                        st.session_state["doc_meta"] = infer_doc_meta_from_index()
                        st.success("Loaded")
                    else:
                        st.error("Could not load index.")

        with col_clear:
            if st.button("🗑 Clear", use_container_width=True, type="secondary"):
                shutil.rmtree(INDEX_ROOT, ignore_errors=True)
                for key in ("db", "n_chunks", "doc_meta", "history"):
                    st.session_state.pop(key, None)
                st.rerun()

    st.divider()

    # Index status badge
    db_live = st.session_state.get("db") is not None
    if db_live:
        st.success("● Index active", icon=None)

        doc_meta: list[dict] = st.session_state.get("doc_meta", [])
        n_chunks: int        = st.session_state.get("n_chunks", 0)

        if doc_meta:
            st.markdown("**Indexed documents**")
            for d in doc_meta:
                pages_label = f"{d['pages']} pages" if d["pages"] != "?" else ""
                st.markdown(f"- `{d['name']}`" + (f"  ·  {pages_label}" if pages_label else ""))

        if n_chunks:
            st.caption(f"{n_chunks} chunks · {EMBED_MODEL.split('/')[-1]}")
        else:
            st.caption(f"Embed: {EMBED_MODEL.split('/')[-1]}")
        st.caption(f"Reranker: {RERANK_MODEL.split('/')[-1]}")
    else:
        st.warning("● No index loaded")
        st.caption(f"Embed: {EMBED_MODEL.split('/')[-1]}")
        st.caption(f"Reranker: {RERANK_MODEL.split('/')[-1]}")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 Product Spec Q&A Assistant")
st.caption("Upload technical PDFs · Ask in natural language · Grounded answer + source")

db = st.session_state.get("db")

if not db:
    st.info("Upload PDFs or load an existing index from the sidebar to get started.")
    st.stop()

try:
    get_llm()   # validates API key early; raises EnvironmentError if missing
except EnvironmentError as exc:
    st.error(str(exc))
    st.stop()

# Query history — persist last 5 in session
history: list[str] = st.session_state.setdefault("history", [])

query = st.text_input(
    "Ask your question",
    placeholder="Can this paint be applied at −10°C?",
    key="query_input",
)

if history:
    with st.expander("Recent questions", expanded=False):
        for i, past in enumerate(reversed(history[-5:])):
            if st.button(past, key=f"hist_{i}", use_container_width=True):
                query = past

if not query.strip():
    st.stop()

# Deduplicate: don't re-run the same query twice in a row
if history and history[-1] == query.strip():
    st.stop()

with st.spinner("Searching…"):
    try:
        result = run_query(query.strip(), db)
    except Exception as exc:
        exc_str = str(exc)
        if "AuthenticationError" in type(exc).__name__ or "Incorrect API key" in exc_str:
            st.error("OpenAI authentication failed. Check your OPENAI_API_KEY in .env.")
        elif "RateLimitError" in type(exc).__name__ or "rate limit" in exc_str.lower():
            st.error("OpenAI rate limit reached. Wait a moment and try again.")
        else:
            st.error(f"Query failed: {exc}")
        st.stop()

# Persist to history
if not history or history[-1] != query.strip():
    history.append(query.strip())

answer_text    = result.get("result", "").strip()
source_docs    = result.get("source_documents", [])
not_found      = answer_text == NOT_FOUND_SENTINEL

st.divider()

# Answer
col_ans, col_meta = st.columns([3, 1])
with col_ans:
    st.subheader("Answer")
    st.markdown(answer_text)

with col_meta:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    if not not_found:
        unique_sources = {
            (Path(d.metadata.get("source", "?")).name, d.metadata.get("page", "?"))
            for d in source_docs
        }
        st.caption(f"📎 {len(unique_sources)} source chunk(s)")

st.divider()

# Source chunks — hidden when LLM says not found (showing them would be misleading)
if not not_found:
    with st.expander("📚 Source Chunks", expanded=True):
        seen: set[tuple] = set()
        for doc in source_docs:
            source   = Path(doc.metadata.get("source", "?")).name
            raw_page = doc.metadata.get("page", "?")
            # PyPDFLoader is 0-indexed; display as 1-indexed to match PDF viewer
            page = (raw_page + 1) if isinstance(raw_page, int) else raw_page
            key  = (source, page)
            if key in seen:
                continue
            seen.add(key)
            st.caption(f"📄 **{source}** · Page {page}")
            st.code(doc.page_content, language="")
            st.write("")
else:
    st.info("No matching content found in the indexed documents. "
            "Try rephrasing or upload additional PDFs.")
