# Hybrid Neural Information Retrieval

Semantic search over the BEIR SciFact corpus, combining BM25 lexical retrieval
with dense embeddings, served as a FastAPI REST service.

**Status:** ongoing undergraduate project (UGP) — semester 1 of 2.

---

## Architecture

```
query
  ├─► BM25 (rank_bm25, in-memory index)          ──► scores
  └─► bge-small-en-v1.5 (HF Inference API)       ──► query vector
                                                      │
                              corpus_embeddings.npy ──┤
                                                      ▼
                              min–max normalize both channels
                                                      ▼
                    final = α · BM25 + (1 − α) · dense      (α weights BM25)
                                                      ▼
                                                   top-k
```

Corpus embeddings are precomputed offline and loaded as an L2-normalized
`float32` matrix (5183 × 384, ~8 MB) at startup, so document scoring is a
single matrix–vector product. Only the query is embedded per request, which
keeps the service inside a 512 MB memory budget — the model itself never has
to be loaded into the web process.

**Stack:** FastAPI · numpy · rank_bm25 · HuggingFace Inference API · Render
(backend) · Vercel (frontend)

---

## Results — SciFact test split (n = 300), Recall@10

Measured during semester 1 with `bge-small-en`. Re-evaluation with the
corrected BM25 tokenizer and the deployed `v1.5` encoder is in progress; the
BM25 figure below is expected to rise.

| Method | Recall@10 |
|---|---|
| BM25 | 0.628 |
| Dense (bge-small-en) | 0.799 |
| HyDE (flan-t5-small generator) | 0.804 |
| Hybrid (α = 0.2) | 0.819 |

**Observations.** Dense retrieval outperformed lexical matching by a wide
margin on this corpus, but part of that gap was an artifact of a broken
baseline (see below). HyDE contributed a gain of roughly half a percentage
point — approximately two queries out of 300, which is inside the noise floor.
The most likely cause is generator quality: `flan-t5-small` (80M parameters)
produces short, generic pseudo-documents on scientific claims, and the
implementation interpolates the query and pseudo-document embeddings at
0.7/0.3 rather than searching on the pseudo-document alone as the original
paper does.

---

## Known issues and semester-2 work

Found by reviewing the semester-1 implementation before extending it.

| Issue | Impact | Status |
|---|---|---|
| Precomputed index built with a different checkpoint than the query encoder | Query and document vectors occupied unrelated spaces — cosine between the API embedding and the stored vector for the *same document* was **0.079**. Dense retrieval was returning effectively random documents, and hybrid was averaging good lexical signal with noise. | **Fixed** — index rebuilt with the matching checkpoint; cosine now 0.969 |
| BM25 tokenized with `str.split()` | Case-sensitive and punctuation-sensitive: `"Cancer"` never matched `"cancer"`, `"cells,"` never matched `"cells"`. The lexical baseline was understated, inflating the reported dense-vs-lexical gap. | **Fixed** — regex tokenizer, applied to both index and query paths |
| Three embedding API calls per search | One round-trip per α branch, though the α = 1.0 (BM25-only) branch discards the embedding it fetches. 3× latency and quota, and a provider failure took down lexical retrieval that needs no network at all. | **Fixed** — score once, fuse three ways |
| No timeout on the embedding call | `requests` blocks indefinitely by default; a hung upstream call pins a worker thread until the process dies. | **Fixed** — bounded connect/read timeout |
| Fusion α tuned on the test split | Test-set leakage in the reported hybrid figure. | Planned — tune on the train split, report test once |
| Recall@10 as the only metric | Ignores rank position entirely: a relevant document at rank 10 scores the same as one at rank 1. | Planned — add nDCG@10 |
| Titles excluded from indexed text | Only `corpus[id]["text"]` is indexed, dropping the highest-signal terms in each document. Penalizes BM25 disproportionately. | Planned |
| HyDE deviates from the published method | Interpolates query and pseudo-document embeddings (0.7/0.3) instead of retrieving on the pseudo-document alone. | Planned — evaluate both, with a larger generator |
| Flat exhaustive search | Correct and optimal at 5,183 documents; does not survive a corpus 100× larger. | Planned — FAISS / HNSW, measuring the recall–latency tradeoff |
| Errors returned as HTTP 200 | Failures are invisible to clients and to monitoring; internal exception text leaks to the response body. | Planned — proper status codes |

---

## Repository layout

```
backend/
  main.py                  FastAPI app and routes
  model.py                 retrieval logic, BM25 + dense fusion
  corpus_embeddings.npy    precomputed L2-normalized document vectors
  doc_ids.npy              positional index → BEIR document id
  corpus_texts.txt         document text for result snippets
  step0_diagnose.py        index/encoder consistency check
frontend/
  index.html               search UI
research/notebooks/        evaluation: BM25, dense, HyDE, hybrid, comparison
scripts/precompute.py      offline index builder
```

---

## Running locally

```bash
pip install -r requirements.txt
export HF_TOKEN="hf_..."          # PowerShell: $env:HF_TOKEN="hf_..."
cd backend
uvicorn main:app --reload
```

```
GET /search?query=covid vaccine effectiveness&alpha=0.2
```

Returns `bm25`, `dense`, and `hybrid` result lists so the three retrieval
strategies can be compared directly on the same query.

### Verifying the index

`step0_diagnose.py` checks that the precomputed index and the query encoder
agree. It embeds a document already present in the index and compares against
the stored vector — cosine should be > 0.95. A low value means the index was
built with a different model than the API serves, which fails silently: the
service returns ranked results, they are simply meaningless.

```bash
python step0_diagnose.py
```

---

## Dataset

BEIR SciFact — 5,183 scientific abstracts, 300 test queries, expert-annotated
relevance judgments. Chosen because it is small enough to iterate on quickly
while remaining a genuine benchmark, and because it is unusually favourable to
lexical retrieval, which makes it a demanding setting for dense methods.