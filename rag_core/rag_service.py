import asyncio
import logging
import time
from typing import List, Dict, Any
from rag_core.config import settings
from rag_core.document_processor import DocumentProcessor
from rag_core.embedding_manager import EmbeddingManager
from rag_core.vector_store_manager import VectorStoreManager
from rag_core.reranker import Reranker

log = logging.getLogger("RAGService")


class RAGService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.doc_processor = DocumentProcessor(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                ocr_provider=settings.OCR_PROVIDER
            )
            cls._instance.embedding_manager = EmbeddingManager()
            cls._instance.vector_store = VectorStoreManager(api_key=settings.PINECONE_API_KEY)
            cls._instance.reranker = Reranker()
            cls._instance.collection_name = "rag-collection-1"

            # Ensure collection exists
            vector_dim = cls._instance.embedding_manager.get_embedding_dimension()
            cls._instance.vector_store.create_or_verify_collection(
                cls._instance.collection_name, vector_dim
            )
        return cls._instance

    async def upload_and_index_document(self, file_path: str, session_id: str):
        """Async version that properly handles async document processing"""
        total_start_time = time.time()

        # 1️⃣ Process document (async)
        chunks = await self.doc_processor.process_file(file_path)
        chunk_texts = [chunk["text"] for chunk in chunks]

        # 2️⃣ Embed chunks (async)
        embeddings = await self.embedding_manager.embed_documents(chunk_texts)

        # 3️⃣ Upsert into vector store (async)
        await self.vector_store.upsert_documents(
            collection_name=self.collection_name,
            session_id=session_id,
            documents=chunks,
            vectors=embeddings
        )

        log.info(
            "Document uploaded & indexed: %s | duration=%.2fs | chunks=%d",
            file_path, time.time() - total_start_time, len(chunks)
        )

    def upload_and_index_document_sync(self, file_path: str, session_id: str):
        """Synchronous wrapper for compatibility with sync code"""
        return asyncio.run(self.upload_and_index_document(file_path, session_id))

    async def query(self, query_text: str, session_id: str) -> List[Dict[str, Any]]:
        total_start_time = time.time()

        # 1️⃣ Embed query
        query_embedding = (await self.embedding_manager.embed_documents([query_text]))[0]

        # 2️⃣ Retrieve from vector store
        retrieved_docs = await self.vector_store.search(
            collection_name=self.collection_name,
            session_id=session_id,
            query_vector=query_embedding,
            top_k=settings.TOP_K_RETRIEVAL
        )

        # 3️⃣ Rerank
        reranked_docs = await asyncio.to_thread(
            self.reranker.rerank,
            query=query_text,
            documents=retrieved_docs,
            top_k=settings.TOP_K_RERANK
        )

        log.info(
            "Query completed | duration=%.2fs | results=%d",
            time.time() - total_start_time, len(reranked_docs)
        )

        return reranked_docs