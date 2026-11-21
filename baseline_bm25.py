import json
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# 1. Load the SciFact document corpus
# ---------------------------------------------------------

corpus = []

with open("data/corpus.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)
        corpus.append(document)

print(f"Loaded {len(corpus)} documents from the SciFact corpus.")


# ---------------------------------------------------------
# 2. Prepare documents for BM25
# ---------------------------------------------------------

tokenized_corpus = []

for document in corpus:
    text = document["title"] + " " + " ".join(document["abstract"])
    tokenized_corpus.append(text.lower().split())


# ---------------------------------------------------------
# 3. Build the BM25 search index
# ---------------------------------------------------------

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index created successfully.")

# ---------------------------------------------------------
# 4. Test BM25 retrieval with one example claim
# ---------------------------------------------------------

test_claim = "ALDH1 expression is associated with poorer prognosis in breast cancer."

tokenized_claim = test_claim.lower().split()

scores = bm25.get_scores(tokenized_claim)

top_5_indices = scores.argsort()[-5:][::-1]

print("\nTest claim:")
print(test_claim)

print("\nTop 5 retrieved documents:")

for rank, index in enumerate(top_5_indices, start=1):
    document = corpus[index]

    print(
        f"{rank}. Document ID: {document['doc_id']} | "
        f"Title: {document['title']}"
    )
    