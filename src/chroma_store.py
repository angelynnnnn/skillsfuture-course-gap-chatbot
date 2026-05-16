from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

def get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_or_create_collection(name: str):
    client = get_client()
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

def reset_collection(name: str):
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

def chroma_results_to_records(results):
    records = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for item_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
        similarity = max(0.0, 1.0 - float(distance))
        record = dict(metadata)
        record["id"] = item_id
        record["text"] = doc
        record["retrieval_distance"] = round(float(distance), 4)
        record["retrieval_similarity"] = round(float(similarity), 4)
        records.append(record)
    return records
