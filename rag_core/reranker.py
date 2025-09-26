import time

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rag_core.config import settings

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("RAGService")

class Reranker:
    """
    Uses a Cross-Encoder model to rerank documents based on their semantic
    relevance to a given query, improving accuracy over pure vector search.
    """

    def __init__(self):
        model_name = settings.RERANKER_MODEL_NAME
        # log.info("Initializing Reranker", model=model_name)
        start_time = time.time()
        try:
            self.model = CrossEncoder(model_name)
            end_time = time.time()
            # log.info(
            #     "Cross-encoder model loaded successfully",
            #     model=model_name,
            #     duration_seconds=round(end_time - start_time, 2)
            # )
        except Exception as e:
            log.error("Failed to load cross-encoder model", model=model_name, error=str(e))
            raise

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents against a query.

        Args:
            query (str): The user's query.
            documents (List[Dict[str, Any]]): The list of documents from the initial retrieval.
            top_k (int): The number of reranked documents to return.

        Returns:
            List[Dict[str, Any]]: The top_k documents, sorted by reranked relevance.
        """
        if not documents:
            log.warning("No documents provided to rerank.")
            return []
            
        # log.info("Reranking documents", query=query, num_docs=len(documents), top_k=top_k)
        start_time = time.time()

        # Create pairs of [query, document_text] for the model
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            
        # Sort documents by the new rerank_score in descending order
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        end_time = time.time()
        # log.info(
        #     "Reranking completed",
        #     duration_seconds=round(end_time - start_time, 2)
        # )
        
        return reranked_docs[:top_k]
