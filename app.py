import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import CrossEncoder

from cache.scrl_cache import SCRLCache
from core.chatbot import chatbot
from core.cold_storage import ColdStorage
from core.config import MYSQL_CONFIG
from core.embedding_model import embedding_model

CACHE_SIZE = 15
HIT_THRESHOLD = 7       # matches needed for a hit; also how many chunks the LLM gets
INSERT_THRESHOLD = 0.5  # a cold-search chunk must be at least this similar to be cached
INSERT_COUNT = 5        # most chunks cached after one miss
BM25_TOP_K = 10

ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def relevance_scores(query: str, results) -> dict:
    """Cross-encoder relevance of each chunk to the query, from 0 to 1."""
    if not results:
        return {}
    raw = ce_model.predict([(query, r.text) for r in results])
    scores = 1 / (1 + np.exp(-np.array(raw)))
    return {r.id: float(s) for r, s in zip(results, scores)}


@st.cache_resource
def init_cold_storage():
    return ColdStorage(**MYSQL_CONFIG)


@st.cache_resource
def init_hot_storage():
    return SCRLCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD)


cold_storage = init_cold_storage()
hot_storage = init_hot_storage()

if "history" not in st.session_state:
    st.session_state.history = []
if "hits" not in st.session_state:
    st.session_state.hits = 0
if "misses" not in st.session_state:
    st.session_state.misses = 0

st.title("CacheMind RAG System")

query = st.chat_input("Enter your query:")

if query:
    details = {
        "hot_matches": 0,
        "bm25_candidates": 0,
        "passed": [],
        "failed": [],
        "pool_size": 0,
        "sent_to_llm": 0,
        "inserted": 0,
        "evicted": [],
    }

    doc_texts = {}

    def get_doc_text(doc_id):
        doc_id = int(doc_id)
        if doc_id not in doc_texts:
            doc_texts[doc_id] = cold_storage.get_document(doc_id)
        return doc_texts[doc_id]

    query_embedding = normalize(np.array(embedding_model.embed_query(query), dtype=np.float32))

    hot_results = hot_storage.request(query_embedding)
    details["hot_matches"] = len(hot_results)
    is_hit = len(hot_results) >= HIT_THRESHOLD

    cold_results = []
    inserted = []

    if is_hit:
        results = hot_results
    else:
        # Miss: search the cold store, keep the chunks that are close in
        # meaning, and mix them with whatever the hot cache already found.
        cold_results = cold_storage.search_bm25(query, top_k=BM25_TOP_K)
        details["bm25_candidates"] = len(cold_results)

        pool = {str(doc_id): {"sim": sim, "embedding": None, "ce_score": None, "bm25": None}
                for doc_id, sim in hot_results}

        ce_scores = relevance_scores(query, cold_results)
        for result in cold_results:
            doc_embedding = normalize(np.array(embedding_model.embed_documents([result.text])[0],
                                               dtype=np.float32))
            sim = float(np.dot(doc_embedding, query_embedding))
            if sim < INSERT_THRESHOLD:
                details["failed"].append({"id": result.id, "sim": sim})
                continue

            details["passed"].append({"id": result.id, "sim": sim})
            doc_id = str(result.id)
            if doc_id not in pool or sim > pool[doc_id]["sim"]:
                pool[doc_id] = {
                    "sim": sim,
                    "embedding": doc_embedding,
                    "ce_score": ce_scores.get(result.id, 1.0),
                    "bm25": result.score,
                }

        ranked = sorted(pool.items(), key=lambda kv: kv[1]["sim"], reverse=True)[:HIT_THRESHOLD]
        results = [(doc_id, info["sim"]) for doc_id, info in ranked]
        details["pool_size"] = len(pool)
        details["sent_to_llm"] = len(ranked)

        # Only chunks that came from the cold search need to be added; the
        # rest are already in the hot cache.
        new_chunks = [(doc_id, info) for doc_id, info in ranked if info["embedding"] is not None]
        for doc_id, info in new_chunks[:INSERT_COUNT]:
            hot_storage.insert(doc_id, info["embedding"], ce_score=info["ce_score"])
            inserted.append((doc_id, info["ce_score"]))
        details["inserted"] = len(inserted)

        if inserted:
            hot_storage.update_retrieved([pool[doc_id]["embedding"] for doc_id, _ in inserted])

        details["evicted"] = hot_storage.pop_evictions()
        for doc_id in details["evicted"]:
            st.toast(f"🗑️ Evicted doc ID: {doc_id}", icon="⚠️")

    if is_hit:
        st.session_state.hits += 1
    else:
        st.session_state.misses += 1

    context = [text for text in (get_doc_text(doc_id) for doc_id, _ in results) if text]
    if context:
        response = chatbot.invoke({"context": "\n".join(context), "question": query})
    else:
        response = "No relevant documents found."

    if is_hit:
        st.markdown("<span style='color:green'><b>CACHE HIT ✅</b></span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red'><b>CACHE MISS ❌</b></span>", unsafe_allow_html=True)

    st.subheader("Response:")
    st.write(response)

    st.subheader("📄 Documents Used for Answer")
    for doc_id, sim in results:
        with st.expander(f"Doc ID: {doc_id} | Similarity: {sim:.4f}"):
            if sim > 0.7:
                dot = "🟢"
            elif sim > 0.4:
                dot = "🟡"
            else:
                dot = "🔴"
            st.progress(sim, text=f"Similarity: {sim:.4f} {dot}")
            doc_text = get_doc_text(doc_id)
            st.write(doc_text[:300] if doc_text else "Text not found")

    if not is_hit:
        st.subheader("🧊 Documents Fetched from Cold Storage (BM25)")
        for i, r in enumerate(cold_results):
            with st.expander(f"Rank {i+1} | Doc ID: {r.id} | BM25 Score: {r.score:.4f}"):
                st.write(r.text[:200])

        if inserted:
            st.subheader(f"🔥 Top {INSERT_COUNT} Docs Inserted into Hot Store")
            cols = st.columns(min(5, len(inserted)))
            for i, (doc_id, ce_score) in enumerate(inserted):
                with cols[i % len(cols)]:
                    st.metric(label=f"Doc {doc_id}", value=f"{ce_score*100:.1f}% CE")
        else:
            st.caption("No cold-storage candidates qualified to be inserted this run.")

    with st.expander("🔍 What happened for this query"):
        st.write(f"**Hit threshold:** {HIT_THRESHOLD} · **Insert threshold:** {INSERT_THRESHOLD}")
        st.write(f"**Hot cache matches:** {details['hot_matches']}, so it is a {'HIT' if is_hit else 'MISS'}")
        if not is_hit:
            st.write(f"**BM25 candidates from cold storage:** {details['bm25_candidates']}")
            if details["passed"]:
                st.write("**Similar enough (added to the pool):**")
                st.table(pd.DataFrame(details["passed"]))
            if details["failed"]:
                st.write("**Not similar enough (dropped):**")
                st.table(pd.DataFrame(details["failed"]))
            st.write(f"**Pool (hot matches + cold chunks):** {details['pool_size']} doc(s)")
            st.write(f"**Sent to the LLM (best {HIT_THRESHOLD} of the pool):** {details['sent_to_llm']} doc(s)")
            st.write(f"**Inserted into the hot cache:** {details['inserted']} of at most {INSERT_COUNT}")
            if details["evicted"]:
                st.write(f"**Evicted to make room:** {details['evicted']}")

    st.session_state.history.append({"query": query, "response": response, "cache_hit": is_hit})
else:
    st.warning("Please enter a query.")

if st.session_state.history:
    st.subheader("Query History")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query {len(st.session_state.history)-i}: {item['query'][:50]}..."):
            st.write(f"**Query:** {item['query']}")
            st.write(f"**Response:** {item['response']}")
            st.write(f"**Cache:** {'Hit' if item['cache_hit'] else 'Miss'}")

# Drawn last so it shows the cache after this query.
with st.sidebar:
    st.title("📊 Cache Statistics")

    stats = hot_storage.get_stats()
    st.metric("Cache Size", f"{stats['size']} / {CACHE_SIZE}")

    total = st.session_state.hits + st.session_state.misses
    hit_rate = st.session_state.hits / total if total else 0.0
    st.metric("Total Hits", st.session_state.hits)
    st.metric("Total Misses", st.session_state.misses)
    st.metric("Hit Rate", f"{hit_rate*100:.1f}%")

    st.divider()

    st.subheader("🧠 Expert Weights (Hedge)")
    weights = stats["weights"]
    st.bar_chart(pd.DataFrame({"Expert": list(weights), "Weight": list(weights.values())})
                 .set_index("Expert"))

    st.divider()

    st.subheader("🎯 Expert Evictions")
    evictions = stats["expert_evictions"]
    st.table({"Expert": list(evictions), "Evictions": list(evictions.values())})

    st.divider()

    st.metric("Hedge Weight Updates", stats["weight_updates"])
    st.metric("History Hits", stats["history_hits"])
