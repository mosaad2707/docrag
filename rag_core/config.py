import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

class Settings(BaseSettings):
    """
    Manages application configuration using environment variables.
    """
    # --- Qdrant Configuration ---
    # Prioritizes QDRANT_URL if set, otherwise falls back to host/port.
    
    PINECONE_API_KEY: Optional[str] = None
    
    QDRANT_URL: Optional[str] = None
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # --- OpenAI Configuration ---
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-4o"
    
    # --- Embedding Model Configuration ---
    EMBEDDING_MODEL_TYPE: Literal["openai", "custom"] = "openai"
    CUSTOM_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    
    # --- Reranker Configuration ---
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # --- Document Processing Configuration ---
    OCR_PROVIDER: Literal["tesseract", "easyocr"] = "tesseract"
    
    # --- RAG Configuration ---
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

try:
    settings = Settings()
except Exception as e:
    logging.error(f"Failed to load settings: {e}")
    # Provide defaults or exit if critical settings are missing
    raise ValueError("Critical configuration is missing. Ensure .env file is set up correctly.") from e

