import json
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# 1. Load the SciFact document corpus
# ---------------------------------------------------------

corpus = []

with open("data/corpus.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        corpus.append(json.loads(line))

print(f"Loaded {len(corpus)} documents from the SciFact corpus.")


# ---------------------------------------------------------
# 2. Prepare corpus and build BM25 index
# ---------------------------------------------------------

tokenized_corpus = []

for document in corpus:
    text = document["title"] + " " + " ".join(document["abstract"])
    tokenized_corpus.append(text.lower().split())

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index created successfully.")


# ---------------------------------------------------------
# 3. Load SciFact development claims with gold evidence
# ---------------------------------------------------------

claims = []

with open("data/claims_dev.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        claim = json.loads(line)

        if claim.get("evidence"):
            claims.append(claim)

print(f"Loaded {len(claims)} claims with gold evidence.")


# ---------------------------------------------------------
# 4. Evaluate retrieval using Recall@5
# ---------------------------------------------------------

correct_at_5 = 0

for claim in claims:
    query = claim["claim"].lower().split()

    scores = bm25.get_scores(query)
    top_5_indices = scores.argsort()[-5:][::-1]

    retrieved_doc_ids = {
        str(corpus[index]["doc_id"])
        for index in top_5_indices
    }

    gold_doc_ids = set(claim["evidence"].keys())

    if retrieved_doc_ids & gold_doc_ids:
        correct_at_5 += 1


# ---------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------

total_claims = len(claims)
recall_at_5 = correct_at_5 / total_claims

print("\n--- BM25 Retrieval Evaluation ---")
print(f"Evaluated claims: {total_claims}")
print(f"Correct evidence found in Top 5: {correct_at_5}")
print(f"Recall@5: {recall_at_5:.2%}")
