# Local Embedding Client

This module provides functionality to generate 1536-dimension embeddings for PDF content using a local model, replacing the need for external API calls to embedding services.

## Features

- Uses SentenceTransformers locally installed models
- Automatically resizes embeddings to 1536 dimensions (OpenAI Ada-002 compatibility)
- Supports batch processing for efficient embedding generation
- Uses GPU acceleration when available

## Setup

1. Install the required dependencies:

   ```
   pip install -r requirements-embedding.txt
   ```

2. The system will use the default model `all-MiniLM-L6-v2` which provides good quality embeddings for semantic search.

3. For improved embeddings, you can specify a different model when initializing the client:

   ```python
   from utils.local_embedding_client import LocalEmbeddingClient

   # Some recommended alternative models:
   # - "sentence-transformers/all-mpnet-base-v2" (higher quality but slower)
   # - "sentence-transformers/multi-qa-mpnet-base-dot-v1" (optimized for retrieval)
   # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (multilingual support)

   client = LocalEmbeddingClient(model_name="sentence-transformers/all-mpnet-base-v2")
   ```

## Usage

```python
# Single embedding
embedding = client.get_embedding("Sample text to embed")

# Batch embeddings
texts = ["First document", "Second document", "Third document"]
embeddings = client.get_embeddings_batch(texts)
```

## Performance Notes

1. The first run will download the model files from HuggingFace.
2. Using GPU acceleration significantly improves performance.
3. The client resizes embeddings to 1536 dimensions by:
   - Truncating if original dimension > 1536
   - Repeating values if original dimension < 1536
4. For most use cases, the default model provides a good balance of quality and speed.

## Integration with Vector DB

The system automatically uses this client for PDF processing and vector search. No code changes are needed in the application as the interface matches the previous embedding client.
