import uuid
import asyncio
import logging
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

log = logging.getLogger("RAGService")


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all metadata values to Pinecone-compatible types.
    """
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


class VectorStoreManager:
    """
    Async Pinecone manager for fast upserts and searches with session-based filtering.
    """

    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_cache: Dict[str, Any] = {}

    def create_or_verify_collection(self, collection_name: str, vector_dim: int):
        indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if collection_name not in indexes:
            log.info(f"Creating Pinecone index: {collection_name}")
            self.pc.create_index(
                name=collection_name,
                dimension=vector_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        if collection_name not in self.index_cache:
            self.index_cache[collection_name] = self.pc.Index(collection_name)

    async def upsert_documents(
        self,
        collection_name: str,
        session_id: str,
        documents: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 50
    ):
        if not documents:
            log.warning("No documents provided for upserting.")
            return

        if collection_name not in self.index_cache:
            self.index_cache[collection_name] = self.pc.Index(collection_name)
        index = self.index_cache[collection_name]

        start_time = time.time()
        items: List[Dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {
                    "text": doc["text"],
                    "session_id": session_id,
                    **sanitize_metadata(doc.get("metadata", {}))
                }
            }
            for doc, vec in zip(documents, vectors)
        ]

        # Upsert in batches concurrently
        async def upsert_batch(batch):
            await asyncio.to_thread(index.upsert, vectors=batch)

        tasks = [upsert_batch(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]
        await asyncio.gather(*tasks)

        log.info(f"Upsert completed in {round(time.time() - start_time, 2)}s ({len(items)} vectors)")

    async def search(
        self,
        collection_name: str,
        session_id: str,
        query_vector: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        if collection_name not in self.index_cache:
            self.index_cache[collection_name] = self.pc.Index(collection_name)
        index = self.index_cache[collection_name]

        start_time = time.time()
        results = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"session_id": {"$eq": session_id}}
        )
        log.info(f"Vector search completed in {round(time.time() - start_time, 2)}s ({len(results['matches'])} hits)")

        return [
            {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"].get("text"),
                "metadata": {k: v for k, v in match["metadata"].items() if k != "text"}
            }
            for match in results["matches"]
        ]