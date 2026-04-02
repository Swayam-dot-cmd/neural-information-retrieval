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
#from sentence_transformers import SentenceTransformer

def get_model():
    global dense_model
    if dense_model is None:
        print("Loading MiniLM model...")
        dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    return dense_model


# 🔥 STEP 1: Load everything ONCE
def initialize():
    global bm25, corpus_embeddings, doc_ids, corpus_texts, initialized

    if initialized:
        return

    print("Loading precomputed data...")

    import numpy as np

    corpus_embeddings = np.load("corpus_embeddings.npy")
    doc_ids = np.load("doc_ids.npy").tolist()

    with open("corpus_texts.txt", "r", encoding="utf-8") as f:
        corpus_texts = [line.strip() for line in f]

    # BM25
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    initialized = True

    print("✅ Fast load complete")

# Load once globally
# bm25, dense_model, corpus_embeddings, doc_ids, corpus_texts = load_pipeline()


# 🔥 STEP 2: Retrieval function
def hybrid_retrieve(query, alpha=0.5, top_k=10):
    global corpus_embeddings

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Query embedding
    #query_embedding = dense_model.encode([query], convert_to_numpy=True)[0]
    model = get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    # 🔥 Lazy compute corpus embeddings
#    if corpus_embeddings is None:
 #       print("Encoding corpus embeddings...")
  #      model = get_model()
   #     corpus_embeddings = model.encode(
    #        corpus_texts,
     #       convert_to_numpy=True,
      #      show_progress_bar=True
       # )

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