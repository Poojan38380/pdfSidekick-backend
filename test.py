from utils.huggingface_client import AsyncHuggingFaceClient
from utils.colorLogger import print_info


# Example usage
async def example_usage():
    client = AsyncHuggingFaceClient()

    # Get embedding for a single text
    text = "Hello, world!"
    embedding = await client.get_embedding(text)
    print_info(f"Embedding dimension: {len(embedding)}")

    # Get embeddings for multiple texts
    texts = ["Hello, world!", "How are you?", "Embedding models are useful."]
    embeddings = await client.get_embeddings_batch(texts)
    print_info(f"Number of embeddings: {len(embeddings)}")
    print_info(f"Embedding dimensions: {[len(emb) for emb in embeddings]}")

    # Normalize embeddings
    normalized = await client.normalize_embeddings(embeddings)
    print_info("Embeddings normalized")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
