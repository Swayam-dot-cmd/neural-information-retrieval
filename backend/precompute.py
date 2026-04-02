from beir.datasets.data_loader import GenericDataLoader
from beir import util
from sentence_transformers import SentenceTransformer
import numpy as np

print("Downloading dataset...")

dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

out_dir = "./datasets"
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-small-en")

print("Encoding corpus...")
corpus_embeddings = model.encode(
    corpus_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Saving files...")

np.save("corpus_embeddings.npy", corpus_embeddings)
np.save("doc_ids.npy", doc_ids)

with open("corpus_texts.txt", "w", encoding="utf-8") as f:
    for text in corpus_texts:
        f.write(text.replace("\n", " ") + "\n")

print("✅ Done!")