import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

st.set_page_config(
    page_title="Financial Insight RAG",
    page_icon="📊",
    layout="wide"
)

SAMPLE_QUESTIONS = [
    "What are the key revenue growth drivers?",
    "What are the main risks facing the company?",
    "How did the company perform this quarter?",
    "What is the guidance for next quarter?",
    "What cost reduction initiatives are being implemented?",
]


@st.cache_resource(show_spinner="Loading earnings call dataset and building index...")
def load_engine():
    try:
        ds = load_dataset("lamini/earnings-calls-qa", split="train[:500]")
    except Exception:
        ds = load_dataset("lamini/earnings-calls-qa", split="train")
        ds = ds.select(range(min(500, len(ds))))

    cols = ds.column_names
    q_col = "question" if "question" in cols else cols[0]
    a_col = "answer" if "answer" in cols else (cols[1] if len(cols) > 1 else cols[0])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    questions = [row[q_col] for row in ds]
    embeddings = model.encode(questions, batch_size=64, show_progress_bar=False)

    return ds, model, np.array(embeddings), questions, q_col, a_col


def search(query, ds, model, embeddings, questions, q_col, a_col, top_k):
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "question": questions[i],
            "answer": ds[int(i)][a_col],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Financial Insight RAG")
st.markdown(
    "Semantic search over **earnings call Q&A** pairs · "
    "Dataset: [lamini/earnings-calls-qa](https://huggingface.co/datasets/lamini/earnings-calls-qa) · "
    "Model: `all-MiniLM-L6-v2`"
)
st.divider()

# ── Load engine ───────────────────────────────────────────────────────────────
try:
    ds, model, embeddings, questions, q_col, a_col = load_engine()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.success(f"✅ {len(ds)} Q&A pairs loaded and indexed from earnings call transcripts")
st.divider()

# ── Sample questions ──────────────────────────────────────────────────────────
st.markdown("**Try a sample question:**")
sample_cols = st.columns(len(SAMPLE_QUESTIONS))
selected_sample = None
for i, (col, q) in enumerate(zip(sample_cols, SAMPLE_QUESTIONS)):
    if col.button(q, key=f"sample_{i}", use_container_width=True):
        selected_sample = q

# ── Search input ──────────────────────────────────────────────────────────────
query = st.text_input(
    "Or type your own financial question:",
    value=selected_sample or "",
    placeholder="e.g. What were the company's earnings per share this quarter?",
)

col_left, col_right = st.columns([3, 1])
with col_right:
    top_k = st.slider("Results to show", min_value=1, max_value=10, value=5)

# ── Results ───────────────────────────────────────────────────────────────────
if query.strip():
    with st.spinner("Searching..."):
        results = search(query, ds, model, embeddings, questions, q_col, a_col, top_k)

    st.subheader(f"Top {len(results)} Results")

    for i, r in enumerate(results, 1):
        score_pct = round(r["score"] * 100, 1)
        badge = "🟢" if r["score"] > 0.65 else "🟡" if r["score"] > 0.35 else "🔴"
        with st.expander(f"{badge} #{i} — Relevance: {score_pct}%  |  {r['question'][:90]}..."):
            st.markdown(f"**Question:**")
            st.info(r["question"])
            st.markdown(f"**Answer:**")
            st.success(r["answer"])
            st.progress(min(r["score"], 1.0), text=f"Similarity score: {score_pct}%")
else:
    st.info("Enter a question above to search through earnings call transcripts.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · "
    "[GitHub](https://github.com/koushik174/financial-insight-rag) · "
    "Dataset licensed under CC-BY"
)
