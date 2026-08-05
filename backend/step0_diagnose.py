
import os, numpy as np, requests

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- Q1: do the three artifacts even line up? -------------------------
emb  = np.load(os.path.join(BASE, "corpus_embeddings.npy"))
ids  = np.load(os.path.join(BASE, "doc_ids.npy"), allow_pickle=True).tolist()
with open(os.path.join(BASE, "corpus_texts.txt"), encoding="utf-8") as f:
    texts = [l.strip() for l in f]

print(f"embeddings : {emb.shape}")
print(f"doc_ids    : {len(ids)}")
print(f"corpus_texts: {len(texts)}")
if not (emb.shape[0] == len(ids) == len(texts)):
    print("!! MISALIGNED -- corpus_texts.txt lost rows to embedded newlines.")
    print("   Your API returns correct doc_ids with the WRONG text attached.")
else:
    print("ok: all three aligned")

# ---- Q2: are the stored vectors normalized? ---------------------------
norms = np.linalg.norm(emb, axis=1)
print(f"\nvector norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")
if abs(norms.mean() - 1.0) > 0.01:
    print("!! NOT normalized -- `corpus_embeddings @ q` is dot product, not cosine.")
    print("   Long documents are being systematically favoured.")

# ---- Q3: does the HF API embedding match the stored index? ------------
# This is the one that matters most. If the .npy was built with a different
# checkpoint than the API serves, query and doc vectors are in different
# spaces and dense retrieval is quietly broken.
tok = os.environ["HF_TOKEN"]
URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

probe = texts[0][:500]          # embed a document the index already contains
r = requests.post(URL, headers={"Authorization": f"Bearer {tok}"},
                  json={"inputs": probe, "options": {"wait_for_model": True}},
                  timeout=(3, 30))
v = np.array(r.json())
print(f"\nAPI returned shape: {v.shape}")
if v.ndim > 1:
    print("!! API returned TOKEN-level embeddings, not a pooled sentence vector.")
    print("   Your code does data[0] -- that is the FIRST TOKEN, not the document.")
    v = v.mean(axis=0)          # mean-pool as a stopgap to continue the test

v = v / np.linalg.norm(v)
stored = emb[0] / np.linalg.norm(emb[0])
sim = float(v @ stored)
print(f"cosine(API embedding of doc0, stored vector for doc0) = {sim:.4f}")
if sim < 0.99:
    print("!! MISMATCH -- the .npy was built with a different model than the API serves.")
    print("   Rebuild the index (Step 1) with the exact checkpoint the API uses.")
else:
    print("ok: index and API agree")