import os
import numpy as np
import requests
from rank_bm25 import BM25Okapi

bm25 = None
corpus_embeddings = None
doc_ids = None
corpus_texts = None
initialized = False

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"


def get_embedding(text):
    import requests
    import numpy as np
    import os

    token = os.getenv("HF_TOKEN")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
        headers=headers,
        json={
            "inputs": text,
            "options": {"wait_for_model": True}
        }
    )

    data = response.json()
    print("HF response:", data)

    # ❌ error handling
    if isinstance(data, dict):
        raise Exception(f"HF API Error: {data}")

    embedding = np.array(data)

    # flatten if needed
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    return embedding

def normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def initialize():
    global bm25, corpus_embeddings, doc_ids, corpus_texts, initialized

    if initialized:
        return

    BASE_DIR = os.path.dirname(__file__)

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

    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    initialized = True
    print("✅ Initialized")


def hybrid_retrieve(query, alpha=0.5, top_k=10):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    query_embedding = get_embedding(query)
    dense_scores = corpus_embeddings @ query_embedding

    bm25_scores = normalize(bm25_scores)
    dense_scores = normalize(dense_scores)

    final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [
        {
            "doc_id": doc_ids[i],
            "text": corpus_texts[i][:300]
        }
        for i in top_indices
    ]


def search(query: str, alpha: float = 0.2):
    if not initialized:
        initialize()

    return {
        "bm25": hybrid_retrieve(query, alpha=1.0),
        "dense": hybrid_retrieve(query, alpha=0.0),
        "hybrid": hybrid_retrieve(query, alpha=alpha)
    }