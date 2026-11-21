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
# 2. Load the dense retrieval model
# ---------------------------------------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Dense retrieval model loaded successfully.")


# ---------------------------------------------------------
# 3. Convert corpus documents into embeddings
# ---------------------------------------------------------

corpus_texts = [
    document["title"] + " " + " ".join(document["abstract"])
    for document in corpus
]

print("Creating embeddings for the SciFact corpus...")

corpus_embeddings = model.encode(
    corpus_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Corpus embeddings created successfully.")


# ---------------------------------------------------------
# 4. Test dense retrieval with the same example claim
# ---------------------------------------------------------

test_claim = (
    "ALDH1 expression is associated with poorer prognosis "
    "in breast cancer."
)

claim_embedding = model.encode(
    test_claim,
    convert_to_tensor=True
)

scores = util.cos_sim(
    claim_embedding,
    corpus_embeddings
)[0]

top_5_indices = scores.argsort(descending=True)[:5]


# ---------------------------------------------------------
# 5. Display the Top 5 retrieved documents
# ---------------------------------------------------------

print("\nTest claim:")
print(test_claim)

print("\nTop 5 retrieved documents:")

for rank, index in enumerate(top_5_indices, start=1):
    document = corpus[index.item()]

    print(
        f"{rank}. Document ID: {document['doc_id']} | "
        f"Title: {document['title']}"
    )
    