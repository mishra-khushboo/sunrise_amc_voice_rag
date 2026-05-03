import chromadb
from chromadb.config import Settings

CHROMA_PATH = "data/chroma_db"
COLLECTION = "sunrise_faq"


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(COLLECTION)


def retrieve(query: str, n_results: int = 3) -> dict:
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return {
        "query": query,
        "results": [
            {
                "text": doc,
                "source": meta.get("source"),
                "q_number": meta.get("q_number")
            }
            for doc, meta in zip(documents, metadatas)
        ]
    }


if __name__ == "__main__":
    query = "How is mutual fund redemption taxed?"
    output = retrieve(query)

    print("\nRetrieved Results:\n")
    for r in output["results"]:
        print(f"[Q{r['q_number']}] {r['text'][:150]}...\n")