from typing import Dict, Any, Optional, List



CREATE_PDFS_TABLE = """
CREATE TABLE IF NOT EXISTS public.Pdf (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    description TEXT,
    document_link TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE
);
"""

# Function to initialize the database
async def initialize_database(connection):
    """Create database tables if they don't exist"""
    
    # Create tables
    await connection.execute(CREATE_PDFS_TABLE)
    
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




async def get_user_by_id(pool, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by ID"""
    
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            """
            SELECT * FROM public."User" WHERE id = $1
            """,
            user_id
        )
        
        return dict(user) if user else None

# PDF model operations
async def create_pdf(pool, title: str, description: str,
                  document_link: str, user_id: str) -> Dict[str, Any]:
    """Create a new PDF document entry in the database"""
    
    async with pool.acquire() as conn:
        pdf = await conn.fetchrow(
            """
            INSERT INTO public.Pdf (title, description, document_link, user_id)
            VALUES ($1, $2, $3, $4)
            RETURNING id, created_at, updated_at, title, description, document_link, user_id
            """,
            title, description, document_link, user_id
        )
        
        return dict(pdf)

async def get_pdfs_by_user_id(pool, user_id: str) -> List[Dict[str, Any]]:
    """Get all PDFs associated with a user"""
    
    async with pool.acquire() as conn:
        pdfs = await conn.fetch(
            """
            SELECT * FROM public.Pdf WHERE user_id = $1 ORDER BY created_at DESC
            """,
            user_id
        )
        
        return [dict(pdf) for pdf in pdfs]

async def get_pdf_by_id(pool, pdf_id: str) -> Optional[Dict[str, Any]]:
    """Get a PDF by ID"""
    
    async with pool.acquire() as conn:
        pdf = await conn.fetchrow(
            """
            SELECT * FROM public.Pdf WHERE id = $1
            """,
            pdf_id
        )
        
        return dict(pdf) if pdf else None 