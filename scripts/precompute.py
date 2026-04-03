from beir.datasets.data_loader import GenericDataLoader
from beir import util
from sentence_transformers import SentenceTransformer
import numpy as np
import os

print("⬇️ Downloading dataset...")

dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

out_dir = "./datasets"
data_path = util.download_and_unzip(url, out_dir)

print("📂 Loading dataset...")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

print("🤖 Loading MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")   # ✅ FIXED

print("⚡ Encoding corpus...")
corpus_embeddings = model.encode(
    corpus_texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=64   # ✅ faster + memory safe
)

# 🔥 Save directly into backend folder
BASE_DIR = os.path.dirname(__file__)

print("💾 Saving files to backend/...")

np.save(os.path.join(BASE_DIR, "corpus_embeddings.npy"), corpus_embeddings)
np.save(os.path.join(BASE_DIR, "doc_ids.npy"), doc_ids)

with open(os.path.join(BASE_DIR, "corpus_texts.txt"), "w", encoding="utf-8") as f:
    for text in corpus_texts:
        f.write(text.replace("\n", " ") + "\n")

print("✅ Precomputation DONE!")