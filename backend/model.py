# backend/model.py
bm25 = None
dense_model = None
corpus_embeddings = None
doc_ids = None
corpus_texts = None
initialized = False

from beir.datasets.data_loader import GenericDataLoader
from beir import util

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 🔥 STEP 1: Load everything ONCE
def initialize():
    global bm25, dense_model, corpus_embeddings, doc_ids, corpus_texts, initialized

    if initialized:
        return

    print("Loading dataset and models...")

    # Download dataset
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    out_dir = "./datasets"
    data_path = util.download_and_unzip(url, out_dir)

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

    # BM25
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Dense model
    dense_model = SentenceTransformer('BAAI/bge-small-en')

    # Embeddings
  #  corpus_embeddings = dense_model.encode(
   #     corpus_texts,
    #    convert_to_numpy=True,
     #   show_progress_bar=True
    #)

    initialized = True
    print("✅ Pipeline loaded successfully")

# Load once globally
# bm25, dense_model, corpus_embeddings, doc_ids, corpus_texts = load_pipeline()


# 🔥 STEP 2: Retrieval function
ddef hybrid_retrieve(query, alpha=0.5, top_k=10):
    global corpus_embeddings

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Query embedding
    query_embedding = dense_model.encode([query], convert_to_numpy=True)[0]

    # 🔥 Lazy compute corpus embeddings
    if corpus_embeddings is None:
        print("Encoding corpus embeddings (first time)...")
        corpus_embeddings = dense_model.encode(
            corpus_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    dense_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]

    # Normalize
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)

    # Combine
    final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [
        {
            "doc_id": doc_ids[i],
            "text": corpus_texts[i]
        }
        for i in top_indices
    ]

# 🔥 STEP 3: API-facing function
def search(query: str, alpha: float = 0.2):
    initialize()   # 👈 THIS IS THE KEY FIX

    bm25_results = hybrid_retrieve(query, alpha=1.0, top_k=10)
    dense_results = hybrid_retrieve(query, alpha=0.0, top_k=10)
    hybrid_results = hybrid_retrieve(query, alpha=alpha, top_k=10)

    return {
        "bm25": bm25_results,
        "dense": dense_results,
        "hybrid": hybrid_results
    }