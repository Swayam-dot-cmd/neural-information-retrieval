# backend/model.py

import os
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Globals
bm25 = None
vectorizer = None
tfidf_matrix = None
doc_ids = None
corpus_texts = None
initialized = False


# 🔥 Normalize safely
def normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


# 🔥 Initialize (lazy loading)
def initialize():
    global bm25, vectorizer, tfidf_matrix, doc_ids, corpus_texts, initialized

    if initialized:
        return

    print("⚡ Initializing lightweight IR system...")

    BASE_DIR = os.path.dirname(__file__)

    try:
        # Load data
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
        assert len(doc_ids) == len(corpus_texts)

        # 🔹 BM25
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # 🔹 TF-IDF (dense replacement)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)

        initialized = True
        print("✅ Initialization complete")

    except Exception as e:
        print("❌ Initialization failed:", e)
        raise


# 🔥 Core retrieval
def hybrid_retrieve(query, alpha=0.5, top_k=10):
    tokenized_query = query.lower().split()

    # BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    # TF-IDF similarity (acts like dense)
    query_vec = vectorizer.transform([query])
    dense_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()

    # Normalize
    bm25_scores = normalize(bm25_scores)
    dense_scores = normalize(dense_scores)

    # Combine
    final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [
        {
            "doc_id": doc_ids[i],
            "text": corpus_texts[i][:300]
        }
        for i in top_indices
    ]


# 🔥 Main search
def search(query: str, alpha: float = 0.2):
    global initialized

    print("🔍 Query:", query)

    if not initialized:
        initialize()

    return {
        "bm25": hybrid_retrieve(query, alpha=1.0),
        "dense": hybrid_retrieve(query, alpha=0.0),
        "hybrid": hybrid_retrieve(query, alpha=alpha)
    }