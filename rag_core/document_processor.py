import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("RAGService")


class DocumentProcessor:
    """
    Async document processor for blazing-fast ingestion.
    Supports text, PDF, Word, Excel, images, and OCR.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100, ocr_provider: str = "easyocr"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_provider = ocr_provider.lower()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        log.info(
            "AsyncDocumentProcessor initialized with chunk_size=%s, overlap=%s, ocr_provider=%s",
            self.chunk_size, self.chunk_overlap, self.ocr_provider
        )

    async def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        loop = asyncio.get_event_loop()

        # 1️⃣ Partition the file (I/O heavy)
        elements = await loop.run_in_executor(
            self.executor,
            lambda: partition(
                filename=file_path,
                skip_infer_table_types=[],
                strategy="auto",
                ocr_kwargs={"provider": self.ocr_provider},
            )
        )

        # 2️⃣ Chunking (CPU heavy)
        chunks = await loop.run_in_executor(
            self.executor,
            lambda: chunk_by_title(
                elements,
                max_characters=self.chunk_size,
                new_after_n_chars=self.chunk_size,
                combine_text_under_n_chars=self.chunk_size // 2,
                overlap=self.chunk_overlap
            )
        )

        # 3️⃣ Build structured chunks
        structured_chunks = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": os.path.basename(file_path),
                "chunk_id": i,
                "category": chunk.category,
            }
            if hasattr(chunk, "metadata"):
                metadata.update(chunk.metadata.to_dict())

            structured_chunks.append({
                "text": chunk.text,
                "metadata": metadata,
            })

        log.info("Processed %d chunks from %s", len(structured_chunks), file_path)
        return structured_chunks