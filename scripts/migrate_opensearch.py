"""Copy legacy vectors to a new OpenSearch index; never change the source."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.opensearch_store import OpenSearchStore
from core.vector_store import VectorStoreManager
from core.business_migration import remap_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["chroma", "faiss"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--source-upload-root")
    options = parser.parse_args()
    if settings.VECTOR_STORE_TYPE != "opensearch":
        parser.error("Load the enterprise environment before migration")
    from core.embedding import get_embedding_model
    from opensearchpy.helpers import bulk
    from core.database import init_db

    init_db()

    settings.OPENSEARCH_INDEX = options.index
    store = OpenSearchStore(get_embedding_model())
    if store.count():
        raise ValueError("Migration requires a new empty destination index")
    assigner = object.__new__(VectorStoreManager)
    if options.source == "chroma":
        import chromadb

        client = chromadb.PersistentClient(path=str(options.source_dir))
        collection = client.get_collection("langchain")
        total = 0
        for offset in range(0, collection.count(), 500):
            payload = collection.get(limit=500, offset=offset, include=["documents", "metadatas", "embeddings"])
            original = [dict(metadata or {}) for metadata in payload["metadatas"]]
            for metadata in original:
                if metadata.get("source"):
                    metadata["source"] = remap_file(metadata["source"], options.source_upload_root, str(settings.UPLOAD_DIR))
            metadatas = assigner._assign_knowledge_metadata(original)
            actions = [{"_index": store.index, "_id": chunk_id,
                "_source": {"content": content, "metadata": metadata, "embedding": list(map(float, vector))}}
                for chunk_id, content, metadata, vector in zip(payload["ids"], payload["documents"], metadatas, payload["embeddings"])]
            if any(len(action["_source"]["embedding"]) != settings.OPENSEARCH_DIMENSION for action in actions):
                raise ValueError("Legacy vector dimensions do not match destination")
            bulk(store.client, actions, refresh="wait_for", raise_on_error=True)
            total += len(actions)
    else:
        from langchain_community.vectorstores import FAISS

        legacy = FAISS.load_local(str(options.source_dir), store.embeddings, allow_dangerous_deserialization=True)
        actions = []
        for position, chunk_id in legacy.index_to_docstore_id.items():
            doc = legacy.docstore.search(chunk_id)
            metadata = dict(doc.metadata)
            if metadata.get("source"):
                metadata["source"] = remap_file(metadata["source"], options.source_upload_root, str(settings.UPLOAD_DIR))
            metadata = assigner._assign_knowledge_metadata([metadata])[0]
            vector = list(map(float, legacy.index.reconstruct(position)))
            if len(vector) != settings.OPENSEARCH_DIMENSION:
                raise ValueError("Legacy vector dimensions do not match destination")
            actions.append({"_index": store.index, "_id": chunk_id, "_source": {"content": doc.page_content, "metadata": metadata, "embedding": vector}})
        bulk(store.client, actions, refresh="wait_for", raise_on_error=True)
        total = len(actions)
    if store.count() != total:
        raise RuntimeError("Migration document count mismatch")
    print(f"Migrated {total} chunks to {store.index}; existing embeddings preserved")


if __name__ == "__main__":
    main()
