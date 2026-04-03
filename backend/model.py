# backend/model.py

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Global variables (DO NOT LOAD ANYTHING HERE)
bm25 = None
dense_model = None
corpus_embeddings = None
doc_ids = None
corpus_texts = None
initialized = False


# 🔥 Load model lazily
def get_model():
    global dense_model
    if dense_model is None:
        print("🔄 Loading MiniLM model...")

        from sentence_transformers import SentenceTransformer  # 🔥 lazy import

        dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    return dense_model


# 🔥 Normalize safely
def normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


# 🔥 Initialize EVERYTHING lazily (only when needed)
def initialize():
    global bm25, corpus_embeddings, doc_ids, corpus_texts, initialized

    if initialized:
        return

    print("⚡ Initializing data on first request...")

    BASE_DIR = os.path.dirname(__file__)

    try:
        # Load embeddings
        corpus_embeddings = np.load(
            os.path.join(BASE_DIR, "corpus_embeddings.npy")
        )

        doc_ids = np.load(
            os.path.join(BASE_DIR, "doc_ids.npy"),
            allow_pickle=True
        ).tolist()

        with open(
            os.path.join(BASE_DIR, "corpus_texts.txt"),
            "r",
            encoding="utf-8"
        ) as f:
            corpus_texts = [line.strip() for line in f]

        # Sanity check
        assert len(doc_ids) == len(corpus_texts) == len(corpus_embeddings)

        # Build BM25
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        initialized = True
        print("✅ Initialization complete")

    except Exception as e:
        print("❌ Initialization failed:", e)
        raise


# 🔥 Core retrieval
def hybrid_retrieve(query, query_embedding, alpha=0.5, top_k=10):
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)
    dense_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]

    # Normalize
    bm25_scores = normalize(bm25_scores)
    dense_scores = normalize(dense_scores)

    # Combine
    final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [
        {
            "doc_id": doc_ids[i],
            "text": corpus_texts[i][:300]  # truncate for speed
        }
        for i in top_indices
    ]


# 🔥 MAIN SEARCH FUNCTION (lazy everything)
def search(query: str, alpha: float = 0.2):
    global initialized

    print("🔍 Query:", query)

    # Initialize only on first request
    if not initialized:
        initialize()

    model = get_model()

    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    bm25_results = hybrid_retrieve(query, query_embedding, alpha=1.0)
    dense_results = hybrid_retrieve(query, query_embedding, alpha=0.0)
    hybrid_results = hybrid_retrieve(query, query_embedding, alpha=alpha)

    return {
        "bm25": bm25_results,
        "dense": dense_results,
        "hybrid": hybrid_results
    }