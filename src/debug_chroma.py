import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chroma_store import get_or_create_collection

COLLECTIONS = [
    "job_documents",
    "course_documents",
    "course_skill_inventory",
]


def inspect_collection(name, limit=5):
    collection = get_or_create_collection(name)
    print("\n" + "=" * 80)
    print(f"Collection: {name}")
    print(f"Count: {collection.count()}")

    results = collection.get(
        limit=limit,
        include=["documents", "metadatas"],
    )

    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    for i in range(len(ids)):
        print("\n--- Record", i + 1, "---")
        print("ID:", ids[i])
        print("Metadata:", metas[i])
        print("Document preview:", docs[i][:700])


if __name__ == "__main__":
    for name in COLLECTIONS:
        inspect_collection(name)
