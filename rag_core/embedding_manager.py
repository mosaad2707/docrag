import time
import os
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from rag_core.config import settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("RAGService")


class EmbeddingManager:
    """
    Async embedding manager for OpenAI or SentenceTransformer models.
    Supports batch embedding and non-blocking operation.
    """

    def __init__(self):
        self.model_type = settings.EMBEDDING_MODEL_TYPE
        self.model = "text-embedding-3-small"
        self.client = None
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        start_time = time.time()
        if self.model_type == "openai":
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = settings.OPENAI_EMBEDDING_MODEL
            log.info("Initialized OpenAI embedding client: %s", self.model_name)
        elif self.model_type == "custom":
            self.model_name = settings.CUSTOM_EMBEDDING_MODEL_NAME
            self.model = SentenceTransformer(self.model_name, device=settings.EMBEDDING_DEVICE)
            log.info("Loaded custom SentenceTransformer model: %s on device %s", self.model_name, settings.EMBEDDING_DEVICE)
        else:
            raise ValueError(f"Unsupported embedding model type: {self.model_type}")
        log.info("AsyncEmbeddingManager initialized in %.2f seconds", time.time() - start_time)

    async def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Async embedding generation for a list of texts.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        loop = asyncio.get_event_loop()

        if self.model_type == "openai":
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                success = False
                retries = 3
                while not success and retries > 0:
                    try:
                        # Run the blocking OpenAI call in thread pool
                        response = await loop.run_in_executor(
                            self.executor,
                            lambda: self.client.embeddings.create(input=batch, model=self.model_name)
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        embeddings.extend(batch_embeddings)
                        success = True
                    except Exception as e:
                        pass
                        retries -= 1
                        log.warning("OpenAI embedding failed, retries left=%d: %s", retries, str(e))
                        await asyncio.sleep(2)
                if not success:
                    raise RuntimeError("Failed to get embeddings from OpenAI after retries")
        else:
            # SentenceTransformer encoding in thread pool
            batch_embeddings = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(texts, convert_to_tensor=False).tolist()
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    def get_embedding_dimension(self) -> int:
        if self.model_type == "openai":
            known_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-ada-002": 1536,
            }
            # fallback: use embed_documents for unknown models
            if self.model_name in known_dims:
                return known_dims[self.model_name]
            else:
                return len(self.embed_documents(["test"])[0])
        elif self.model_type == "custom":
            return self.model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unsupported embedding model type: {self.model_type}")