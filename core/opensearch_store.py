"""OpenSearch BM25 and permission-filtered Lucene vector retrieval."""
import hashlib
import json
import math
from urllib.parse import urlsplit

from langchain_core.documents import Document

from config.settings import settings


def search_filter(filters: dict | None) -> dict:
    clauses = []
    for key, value in (filters or {}).items():
        if key in {"$and", "$or"}:
            parts = [search_filter(part) for part in value]
            clauses.append({"bool": {"filter": parts}} if key == "$and" else {"bool": {"should": parts, "minimum_should_match": 1}})
            continue
        if not key.replace("_", "").isalnum():
            raise ValueError("Invalid metadata filter field")
        field = "metadata." + key
        if not isinstance(value, dict):
            clauses.append({"term": {field: value}})
            continue
        for operation, operand in value.items():
            if operation in {"$eq", "$in"}:
                clauses.append({"term" if operation == "$eq" else "terms": {field: operand}})
            elif operation in {"$ne", "$nin"}:
                clauses.append({"bool": {"must_not": [{"term" if operation == "$ne" else "terms": {field: operand}}]}})
            elif operation in {"$gt", "$gte", "$lt", "$lte"}:
                clauses.append({"range": {field: {operation[1:]: operand}}})
            else:
                raise ValueError("Unsupported metadata filter: " + operation)
    return {"bool": {"filter": clauses}} if clauses else {"match_all": {}}


def cosine_similarity(left, right) -> float:
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else 0.0


def authorized_filter(filters):
    for key, value in (filters or {}).items():
        if key == "knowledge_base_id":
            if isinstance(value, str) and value:
                return True
            if isinstance(value, dict) and (value.get("$in") or value.get("$eq")):
                return True
        if key == "$and" and any(authorized_filter(part) for part in value):
            return True
        if key == "$or" and value and all(authorized_filter(part) for part in value):
            return True
    return False


class OpenSearchStore:
    def __init__(self, embeddings, client=None, initialize=True, index_name=None):
        if client is None:
            from opensearchpy import OpenSearch

            url = urlsplit(settings.OPENSEARCH_URL)
            if url.username or url.password:
                raise ValueError("Use separate OpenSearch credential settings")
            client = OpenSearch(hosts=[settings.OPENSEARCH_URL],
                http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
                verify_certs=settings.OPENSEARCH_VERIFY_CERTS, ca_certs=settings.OPENSEARCH_CA_CERTS,
                timeout=settings.OPENSEARCH_TIMEOUT)
        self.client = client
        self.embeddings = embeddings
        self.index = index_name or settings.OPENSEARCH_INDEX
        if initialize:
            self._ensure_index()

    def _ensure_index(self):
        if self.client.indices.exists(index=self.index):
            mapping = self.client.indices.get_mapping(index=self.index)
            dimensions = {entry["mappings"]["properties"]["embedding"]["dimension"] for entry in mapping.values()}
            if dimensions != {settings.OPENSEARCH_DIMENSION}:
                raise ValueError("OpenSearch embedding dimension differs from configured model")
            return
        self.client.indices.create(index=self.index, body={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic_templates": [{"strings": {"match_mapping_type": "string", "mapping": {"type": "keyword"}}}],
                "properties": {
                    "content": {"type": "text", "analyzer": "cjk", "similarity": "BM25"},
                    "embedding": {"type": "knn_vector", "dimension": settings.OPENSEARCH_DIMENSION,
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}},
                    "metadata": {"type": "object"},
                },
            },
        })

    def add_documents(self, documents):
        from opensearchpy.helpers import bulk

        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents):
            raise ValueError("Embedding count mismatch")
        if any(len(vector) != settings.OPENSEARCH_DIMENSION for vector in vectors):
            raise ValueError("Embedding dimension mismatch")
        actions = []
        for number, (doc, vector) in enumerate(zip(documents, vectors)):
            signature = json.dumps([doc.page_content, doc.metadata, number], sort_keys=True, ensure_ascii=True)
            actions.append({"_index": self.index, "_id": hashlib.sha256(signature.encode()).hexdigest(),
                "_source": {"content": doc.page_content, "metadata": doc.metadata, "embedding": vector}})
        bulk(self.client, actions, refresh="wait_for", raise_on_error=True)
        return [action["_id"] for action in actions]

    def similarity_search_with_score(self, query, k=5, filter=None):
        if not authorized_filter(filter):
            raise ValueError("An authorized knowledge filter is required")
        vector = self.embeddings.embed_query(query)
        if len(vector) != settings.OPENSEARCH_DIMENSION:
            raise ValueError("Embedding dimension mismatch")
        response = self.client.search(index=self.index, body={"size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k, "filter": search_filter(filter)}}}})
        return [(Document(page_content=hit["_source"]["content"], metadata=hit["_source"]["metadata"]),
            1.0 - cosine_similarity(vector, hit["_source"]["embedding"])) for hit in response["hits"]["hits"]]

    def similarity_search(self, query, k=5, filter=None):
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter)]

    def keyword_search(self, query, k, filter):
        if not authorized_filter(filter):
            raise ValueError("An authorized knowledge filter is required")
        response = self.client.search(index=self.index, body={"size": k, "_source": ["content", "metadata"],
            "min_score": settings.SEARCH_MIN_BM25_SCORE,
            "query": {"bool": {"must": [{"match": {"content": {"query": query, "operator": "or"}}}],
                "filter": [search_filter(filter)]}}})
        return [(Document(page_content=hit["_source"]["content"], metadata=hit["_source"]["metadata"]), hit["_score"])
            for hit in response["hits"]["hits"]]

    def get(self, where=None, include=None):
        from opensearchpy.helpers import scan

        hits = list(scan(self.client, index=self.index, query={"query": search_filter(where), "_source": ["content", "metadata"]}))
        return {"ids": [hit["_id"] for hit in hits], "documents": [hit["_source"]["content"] for hit in hits],
            "metadatas": [hit["_source"]["metadata"] for hit in hits]}

    def delete_by_source(self, source):
        response = self.client.delete_by_query(index=self.index, body={"query": {"term": {"metadata.source": source}}}, refresh=True)
        if response.get("failures"):
            raise RuntimeError("OpenSearch deletion failed")
        return response.get("deleted", 0)

    def count(self):
        return self.client.count(index=self.index)["count"]

    def clear(self):
        self.client.indices.delete(index=self.index)
        self._ensure_index()
