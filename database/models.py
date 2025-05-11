from typing import Dict, Any, Optional, List
import json


CREATE_PDFS_TABLE = """
CREATE TABLE IF NOT EXISTS public.Pdf (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    description TEXT,
    document_link TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    processing_status TEXT DEFAULT 'pending',
    processing_progress FLOAT DEFAULT 0,
    total_pages INTEGER,
    extracted_content TEXT,
    error_message TEXT,
    indexing_step TEXT,
    chunks_processed INTEGER,
    embeddings_created INTEGER
);
"""

CREATE_PDF_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS public.PdfChunk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    pdf_id UUID NOT NULL REFERENCES public.Pdf(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    metadata JSONB
);
"""

CREATE_PDF_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS public.PdfEmbedding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    pdf_id UUID NOT NULL REFERENCES public.Pdf(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES public.PdfChunk(id) ON DELETE CASCADE,
    embedding vector(384),
    embedding_model TEXT NOT NULL
);
"""

CREATE_PDF_EMBEDDINGS_INDEX = """
CREATE INDEX IF NOT EXISTS pdf_embeddings_hnsw_idx 
ON public.PdfEmbedding USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
"""


# Function to initialize the database
async def initialize_database(connection):
    """Create database tables if they don't exist"""

    # Create tables
    await connection.execute(CREATE_PDFS_TABLE)
    await connection.execute(CREATE_PDF_CHUNKS_TABLE)
    await connection.execute(CREATE_PDF_EMBEDDINGS_TABLE)

    # Create trigger for updating the updated_at columns
    update_timestamp_trigger = """
    CREATE OR REPLACE FUNCTION update_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    create_pdf_trigger = """
    DROP TRIGGER IF EXISTS update_pdfs_timestamp ON public.Pdf;
    CREATE TRIGGER update_pdfs_timestamp
    BEFORE UPDATE ON public.Pdf
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();
    """

    await connection.execute(update_timestamp_trigger)
    await connection.execute(create_pdf_trigger)

    # Enable pgvector extension if not already enabled
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create vector index
        await connection.execute(CREATE_PDF_EMBEDDINGS_INDEX)
    except Exception as e:
        print(f"Error creating vector extension or index (in initialize_database): {e}")


async def get_user_by_id(pool, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by ID"""
    try:
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                """
            SELECT * FROM public."User" WHERE id = $1
            """,
                user_id,
            )

        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by ID (in get_user_by_id): {e}")
        raise


# PDF model operations
async def create_pdf(
    pool, title: str, description: str, document_link: str, user_id: str
) -> Dict[str, Any]:
    """Create a new PDF document entry in the database"""

    try:
        async with pool.acquire() as conn:
            pdf = await conn.fetchrow(
                """
            INSERT INTO public.Pdf (title, description, document_link, user_id, processing_status, processing_progress)
            VALUES ($1, $2, $3, $4, 'pending', 0)
            RETURNING id, created_at, updated_at, title, description, document_link, user_id, processing_status, processing_progress
            """,
                title,
                description,
                document_link,
                user_id,
            )

        # Convert the record to a dict and ensure UUID is converted to string
        pdf_dict = dict(pdf)
        pdf_dict["id"] = str(pdf_dict["id"])
        return pdf_dict
    except Exception as e:
        print(f"Error creating PDF (in create_pdf): {e}")
        raise


async def get_pdfs_by_user_id(pool, user_id: str) -> List[Dict[str, Any]]:
    """Get all PDFs associated with a user"""

    try:
        async with pool.acquire() as conn:
            pdfs = await conn.fetch(
                """
            SELECT * FROM public.Pdf WHERE user_id = $1 ORDER BY created_at DESC
            """,
                user_id,
            )

        # Convert records to dicts and ensure UUIDs are converted to strings
        return [dict(pdf, id=str(pdf["id"])) for pdf in pdfs]
    except Exception as e:
        print(f"Error getting PDFs by user ID (in get_pdfs_by_user_id): {e}")
        raise


async def get_pdf_by_id(pool, pdf_id: str) -> Optional[Dict[str, Any]]:
    """Get a PDF by ID"""

    try:
        async with pool.acquire() as conn:
            pdf = await conn.fetchrow(
                """
            SELECT * FROM public.Pdf WHERE id = $1
            """,
                pdf_id,
            )

        if pdf:
            pdf_dict = dict(pdf)
            pdf_dict["id"] = str(pdf_dict["id"])
            return pdf_dict
        return None
    except Exception as e:
        print(f"Error getting PDF by ID (in get_pdf_by_id): {e}")
        raise


async def update_pdf_processing_status(
    pool,
    pdf_id: str,
    status: str,
    progress: float = None,
    total_pages: int = None,
    error_message: str = None,
    indexing_step: str = None,
    chunks_processed: int = None,
    embeddings_created: int = None,
) -> Dict[str, Any]:
    """Update the processing status of a PDF"""

    try:
        query_parts = ["UPDATE public.Pdf SET processing_status = $1"]
        params = [status]
        param_count = 2

        if progress is not None:
            query_parts.append(f", processing_progress = ${param_count}")
            params.append(progress)
            param_count += 1

        if total_pages is not None:
            query_parts.append(f", total_pages = ${param_count}")
            params.append(total_pages)
            param_count += 1

        if error_message is not None:
            query_parts.append(f", error_message = ${param_count}")
            params.append(error_message)
            param_count += 1

        if indexing_step is not None:
            query_parts.append(f", indexing_step = ${param_count}")
            params.append(indexing_step)
            param_count += 1

        if chunks_processed is not None:
            query_parts.append(f", chunks_processed = ${param_count}")
            params.append(chunks_processed)
            param_count += 1

        if embeddings_created is not None:
            query_parts.append(f", embeddings_created = ${param_count}")
            params.append(embeddings_created)
            param_count += 1

        query_parts.append(f" WHERE id = ${param_count}")
        params.append(pdf_id)

        query = (
            "".join(query_parts)
            + " RETURNING id, processing_status, processing_progress, total_pages, error_message, indexing_step, chunks_processed, embeddings_created"
        )

        async with pool.acquire() as conn:
            pdf = await conn.fetchrow(query, *params)

            if pdf:
                pdf_dict = dict(pdf)
                pdf_dict["id"] = str(pdf_dict["id"])
                return pdf_dict
            return None
    except Exception as e:
        print(
            f"Error updating PDF processing status (in update_pdf_processing_status): {e}"
        )
        raise


async def update_pdf_extracted_content(
    pool, pdf_id: str, extracted_content: str
) -> Dict[str, Any]:
    """Update the extracted content of a PDF"""

    async with pool.acquire() as conn:
        pdf = await conn.fetchrow(
            """
            UPDATE public.Pdf SET extracted_content = $1
            WHERE id = $2
            RETURNING id, extracted_content
            """,
            extracted_content,
            pdf_id,
        )

        if pdf:
            pdf_dict = dict(pdf)
            pdf_dict["id"] = str(pdf_dict["id"])
            return pdf_dict
        return None


async def create_pdf_chunk(
    pool,
    pdf_id: str,
    chunk_index: int,
    content: str,
    page_number: int = None,
    metadata: dict = None,
) -> Dict[str, Any]:
    """Create a new PDF chunk entry in the database"""

    # Convert metadata dict to JSON string for PostgreSQL JSONB
    if metadata is not None:
        metadata_json = json.dumps(metadata)
    else:
        metadata_json = None

    async with pool.acquire() as conn:
        chunk = await conn.fetchrow(
            """
            INSERT INTO public.PdfChunk (pdf_id, chunk_index, page_number, content, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id, created_at, pdf_id, chunk_index, page_number, content, metadata
            """,
            pdf_id,
            chunk_index,
            page_number,
            content,
            metadata_json,
        )

        chunk_dict = dict(chunk)
        chunk_dict["id"] = str(chunk_dict["id"])
        chunk_dict["pdf_id"] = str(chunk_dict["pdf_id"])
        return chunk_dict


async def get_pdf_chunks(pool, pdf_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a specific PDF"""

    async with pool.acquire() as conn:
        chunks = await conn.fetch(
            """
            SELECT * FROM public.PdfChunk WHERE pdf_id = $1 ORDER BY chunk_index
            """,
            pdf_id,
        )

        return [
            dict(chunk, id=str(chunk["id"]), pdf_id=str(chunk["pdf_id"]))
            for chunk in chunks
        ]


async def create_pdf_embedding(
    pool,
    pdf_id: str,
    chunk_id: str,
    embedding: List[float],
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, Any]:
    """Create a new PDF embedding entry in the database"""

    # Convert embedding list to PostgreSQL vector format
    embedding_str = f"[{','.join(map(str, embedding))}]"

    async with pool.acquire() as conn:
        embedding_record = await conn.fetchrow(
            """
            INSERT INTO public.PdfEmbedding (pdf_id, chunk_id, embedding, embedding_model)
            VALUES ($1, $2, $3::vector, $4)
            RETURNING id, created_at, pdf_id, chunk_id, embedding_model
            """,
            pdf_id,
            chunk_id,
            embedding_str,
            embedding_model,
        )

        embedding_dict = dict(embedding_record)
        embedding_dict["id"] = str(embedding_dict["id"])
        embedding_dict["pdf_id"] = str(embedding_dict["pdf_id"])
        embedding_dict["chunk_id"] = str(embedding_dict["chunk_id"])
        return embedding_dict


async def get_pdf_embeddings(pool, pdf_id: str) -> List[Dict[str, Any]]:
    """Get all embeddings for a specific PDF"""

    async with pool.acquire() as conn:
        embeddings = await conn.fetch(
            """
            SELECT id, created_at, pdf_id, chunk_id, embedding_model
            FROM public.PdfEmbedding 
            WHERE pdf_id = $1
            """,
            pdf_id,
        )

        return [
            dict(
                embedding,
                id=str(embedding["id"]),
                pdf_id=str(embedding["pdf_id"]),
                chunk_id=str(embedding["chunk_id"]),
            )
            for embedding in embeddings
        ]


async def search_similar_chunks(
    pool, query_embedding: List[float], limit: int = 5, threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks using vector similarity search

    Args:
        pool: Database connection pool
        query_embedding: The query embedding vector
        limit: Maximum number of results to return
        threshold: Minimum similarity threshold (0-1)

    Returns:
        List of similar chunks with similarity scores
    """
    # Convert embedding list to PostgreSQL vector format
    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT 
                e.id as embedding_id,
                e.pdf_id,
                e.chunk_id,
                c.content,
                c.page_number,
                c.metadata,
                p.title,
                1 - (e.embedding <=> $1::vector) as similarity
            FROM 
                public.PdfEmbedding e
                JOIN public.PdfChunk c ON e.chunk_id = c.id
                JOIN public.Pdf p ON e.pdf_id = p.id
            WHERE 
                1 - (e.embedding <=> $1::vector) > $2
            ORDER BY 
                similarity DESC
            LIMIT $3
            """,
            embedding_str,
            threshold,
            limit,
        )

        return [dict(result) for result in results]
