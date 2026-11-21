import json
import os
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

RRF_K = 60
RETRIEVAL_DEPTH = 100


# --------------------------------------------------
# 1. LOAD SCIFACT CORPUS
# --------------------------------------------------

corpus = []

with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        corpus.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": " ".join(doc["abstract"])
        })

print(f"Loaded {len(corpus)} documents.")


# --------------------------------------------------
# 2. LOAD SAME 188 CLAIMS
# --------------------------------------------------

claims = []

with open("data/claims_dev.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        claim = json.loads(line)

        if claim.get("evidence"):
            claims.append(claim)

print(f"Using {len(claims)} claims.")


# --------------------------------------------------
# 3. BM25
# --------------------------------------------------

tokenized_corpus = [
    doc["text"].lower().split()
    for doc in corpus
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")


# --------------------------------------------------
# 4. DENSE MODEL + SAVED EMBEDDINGS
# --------------------------------------------------

print("Loading BGE-small model...")

dense_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embedding_file = "data/corpus_embeddings.npy"

if not os.path.exists(embedding_file):
    raise FileNotFoundError(
        "Saved corpus embeddings not found."
    )

corpus_embeddings = np.load(embedding_file)

print("Dense retrieval ready.")


# --------------------------------------------------
# 5. HYBRID RRF EVALUATION
# --------------------------------------------------

hits = 0
results = []

for number, claim in enumerate(claims, start=1):

    claim_text = claim["claim"]

    print(f"Processing claim {number}/{len(claims)}")


    # ---------------- BM25 RANKING ----------------

    bm25_scores = bm25.get_scores(
        claim_text.lower().split()
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:RETRIEVAL_DEPTH]


    # ---------------- DENSE RANKING ----------------

    claim_embedding = dense_model.encode(
        claim_text,
        normalize_embeddings=True
    )

    dense_scores = corpus_embeddings @ claim_embedding

    dense_indices = np.argsort(
        dense_scores
    )[::-1][:RETRIEVAL_DEPTH]


    # ---------------- RRF ----------------

    rrf_scores = {}

    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    for rank, index in enumerate(dense_indices, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    hybrid_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:5]


    # ---------------- GOLD DOCUMENTS ----------------

    gold_doc_ids = {
        int(doc_id)
        for doc_id in claim["evidence"].keys()
    }

    retrieved_doc_ids = {
        corpus[i]["doc_id"]
        for i in hybrid_indices
    }


    # ---------------- GOLD LABEL ----------------

    labels = []

    for document_evidence in claim["evidence"].values():
        for item in document_evidence:
            labels.append(item["label"])

    gold_label = labels[0]


    # ---------------- RECALL@5 ----------------

    hit = bool(
        gold_doc_ids.intersection(retrieved_doc_ids)
    )

    if hit:
        hits += 1


    # ---------------- SAVE ----------------

    results.append({
        "claim_id": claim["id"],
        "claim": claim_text,
        "gold_label": gold_label,
        "gold_doc_ids": list(gold_doc_ids),
        "hybrid_top5": [
            {
                "doc_id": corpus[i]["doc_id"],
                "title": corpus[i]["title"],
                "text": corpus[i]["text"]
            }
            for i in hybrid_indices
        ],
        "hit_at_5": hit
    })


# --------------------------------------------------
# 6. FINAL RESULT
# --------------------------------------------------

recall_at_5 = hits / len(claims)

print()
print("--------------------------------")
print("HYBRID RRF RESULTS")
print("--------------------------------")
print(f"Hits: {hits}/{len(claims)}")
print(f"Recall@5: {recall_at_5:.2%}")


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

with open(
    "hybrid_rrf_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Saved to: hybrid_rrf_results.json")
