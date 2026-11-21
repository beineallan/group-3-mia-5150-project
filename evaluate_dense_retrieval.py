import json
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------
# 1. Load the SciFact document corpus
# ---------------------------------------------------------

corpus = []

with open("data/corpus.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        corpus.append(json.loads(line))

print(f"Loaded {len(corpus)} documents from the SciFact corpus.")


# ---------------------------------------------------------
# 2. Load dense retrieval model
# ---------------------------------------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Dense retrieval model loaded successfully.")


# ---------------------------------------------------------
# 3. Create corpus embeddings
# ---------------------------------------------------------

corpus_texts = [
    document["title"] + " " + " ".join(document["abstract"])
    for document in corpus
]

print("Creating corpus embeddings...")

corpus_embeddings = model.encode(
    corpus_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Corpus embeddings created successfully.")


# ---------------------------------------------------------
# 4. Load SciFact development claims with gold evidence
# ---------------------------------------------------------

claims = []

with open("data/claims_dev.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        claim = json.loads(line)

        if claim.get("evidence"):
            claims.append(claim)

print(f"Loaded {len(claims)} claims with gold evidence.")


# ---------------------------------------------------------
# 5. Create embeddings for all claims
# ---------------------------------------------------------

claim_texts = [claim["claim"] for claim in claims]

print("Creating claim embeddings...")

claim_embeddings = model.encode(
    claim_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)


# ---------------------------------------------------------
# 6. Evaluate dense retrieval using Recall@5
# ---------------------------------------------------------

correct_at_5 = 0

for i, claim in enumerate(claims):

    scores = util.cos_sim(
        claim_embeddings[i],
        corpus_embeddings
    )[0]

    top_5_indices = scores.argsort(descending=True)[:5]

    retrieved_doc_ids = {
        str(corpus[index.item()]["doc_id"])
        for index in top_5_indices
    }

    gold_doc_ids = set(claim["evidence"].keys())

    if retrieved_doc_ids & gold_doc_ids:
        correct_at_5 += 1


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

total_claims = len(claims)
recall_at_5 = correct_at_5 / total_claims

print("\n--- Dense Retrieval Evaluation ---")
print(f"Evaluated claims: {total_claims}")
print(f"Correct evidence found in Top 5: {correct_at_5}")
print(f"Recall@5: {recall_at_5:.2%}")
