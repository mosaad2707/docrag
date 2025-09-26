Here is a fixed and improved version of the project's README.

# Doc RAG Service

This project implements a comprehensive **Retrieval-Augmented Generation (RAG) service** with a modular backend built on **FastAPI** and a user-friendly frontend using **Streamlit**. It's designed for high performance, accuracy, and scalability when handling a variety of document types.

-----

##  Features

  * **Vector Database**: Integrates with **Pinceone/Qdrant** for efficient vector storage and retrieval.
  * **Configurable Embeddings**: Supports both **OpenAI embeddings** and custom, local **sentence-transformer** models.
  * **Advanced Retrieval**: Implements a **reranking** step using a **cross-encoder** to improve the relevance and quality of retrieved documents.
  * **High-Performance Document Processing**: Uses the `unstructured` library for fast and accurate extraction of text, tables, and images from various file formats.
  * **Integrated OCR**: Supports **Tesseract** and **EasyOCR** for text extraction from images, configurable via environment variables.
  * **Session Management**: Associates all data in Qdrant with a `session_id` to provide user-specific context and data isolation.
  * **REST API**: A robust FastAPI backend exposes all core functionalities.
  * **Comprehensive Logging**: Detailed logging with timestamps for performance monitoring and debugging.

-----

## 🛠️ Setup Instructions

### 1\. Prerequisites

  * **Python 3.9+**
  * **Tesseract OCR Engine** (required if `OCR_PROVIDER` is set to `tesseract`):
      * **macOS**: `brew install tesseract`
      * **Ubuntu**: `sudo apt-get install tesseract-ocr`
      * **Windows**: Download from the official [Tesseract repository](https://github.com/tesseract-ocr/tessdoc).

### 2\. Clone the Repository

Clone the project to your local machine using the following commands:

```bash
git clone <repository_url>
cd <repository_name>
```

### 3\. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*Note: If you plan to use EasyOCR, it may have additional system-level dependencies. But if it comes preinstalled with unstructured

### 4\. Configure Environment Variables

Create your environment file by copying the example and then edit it with your specific details.

```bash
cp .env.example .env
```

Open the `.env` file and configure the following key variables:

  * `OPENAI_API_KEY`: Your OpenAI API key.
  * `EMBEDDING_MODEL_TYPE`: Set to `openai` or `custom`.
  * `CUSTOM_EMBEDDING_MODEL_NAME`: The name of your Hugging Face model (e.g., `all-MiniLM-L6-v2`), only needed if `EMBEDDING_MODEL_TYPE` is `custom`.
  * `OCR_PROVIDER`: Set to `tesseract` or `easyocr`.

### 5\. Run the Services

You need to run the backend and frontend in separate terminal windows.

**Terminal 1: Run the FastAPI Backend**

```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Run the Streamlit Frontend**

```bash
streamlit run streamlit_app.py
```

Once both services are running, you can access the **Streamlit application** in your web browser at `http://localhost:8501`.
