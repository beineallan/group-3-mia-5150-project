import json
import os
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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
# 2. LOAD ALL CLAIMS WITH GOLD EVIDENCE
# --------------------------------------------------

claims = []

with open("data/claims_dev.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        claim = json.loads(line)

        if claim.get("evidence"):
            claims.append(claim)

print(f"Using {len(claims)} claims for final experiment.")


# --------------------------------------------------
# 3. CREATE BM25 INDEX
# --------------------------------------------------

tokenized_corpus = [
    doc["text"].lower().split()
    for doc in corpus
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")


# --------------------------------------------------
# 4. LOAD DENSE MODEL
# --------------------------------------------------

print("Loading BGE-small model...")

dense_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# 5. LOAD SAVED CORPUS EMBEDDINGS
# --------------------------------------------------

embedding_file = "data/corpus_embeddings.npy"

if not os.path.exists(embedding_file):
    raise FileNotFoundError(
        "Saved corpus embeddings not found at "
        "data/corpus_embeddings.npy"
    )

print("Loading saved corpus embeddings...")

corpus_embeddings = np.load(embedding_file)

print("Saved embeddings loaded.")


# --------------------------------------------------
# 6. RETRIEVE TOP 5 FOR ALL 188 CLAIMS
# --------------------------------------------------

final_results = []

for number, claim in enumerate(claims, start=1):

    claim_text = claim["claim"]

    print(f"Processing claim {number}/{len(claims)}")


    # ---------------- BM25 ----------------

    bm25_scores = bm25.get_scores(
        claim_text.lower().split()
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:5]

    bm25_top5 = [
        {
            "doc_id": corpus[i]["doc_id"],
            "title": corpus[i]["title"],
            "text": corpus[i]["text"]
        }
        for i in bm25_indices
    ]


    # ---------------- DENSE ----------------

    claim_embedding = dense_model.encode(
        claim_text,
        normalize_embeddings=True
    )

    dense_scores = corpus_embeddings @ claim_embedding

    dense_indices = np.argsort(
        dense_scores
    )[::-1][:5]

    dense_top5 = [
        {
            "doc_id": corpus[i]["doc_id"],
            "title": corpus[i]["title"],
            "text": corpus[i]["text"]
        }
        for i in dense_indices
    ]


    # ---------------- GOLD LABEL ----------------

    labels = []

    for document_evidence in claim["evidence"].values():
        for item in document_evidence:
            labels.append(item["label"])

    gold_label = labels[0]


    # ---------------- SAVE RESULT ----------------

    final_results.append({
        "claim_id": claim["id"],
        "claim": claim_text,
        "gold_label": gold_label,
        "bm25_top5": bm25_top5,
        "dense_top5": dense_top5
    })


# --------------------------------------------------
# 7. SAVE FINAL RETRIEVAL DATA
# --------------------------------------------------

output_file = "final_retrieval_results.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        final_results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("--------------------------------")
print("FINAL RETRIEVAL DATA READY")
print("--------------------------------")
print(f"Claims: {len(final_results)}")
print(f"Saved to: {output_file}")
