async def add_indexing_status_columns():
    """Add indexing status columns to the Pdf table"""
    conn = await get_connection()
    try:
        # Check if columns already exist
        columns_exist = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'Pdf'
                AND column_name = 'indexing_step'
            );
        """
        )

        if columns_exist:
            print_info("Indexing status columns already exist in Pdf table")
            return

        # Add new columns
        await conn.execute(
            """
            ALTER TABLE public.Pdf
            ADD COLUMN IF NOT EXISTS indexing_step TEXT,
            ADD COLUMN IF NOT EXISTS chunks_processed INTEGER,
            ADD COLUMN IF NOT EXISTS embeddings_created INTEGER;
        """
        )

        print_info("Successfully added indexing status columns to Pdf table")
    except Exception as e:
        print_error(f"Error adding indexing status columns: {e}")
        raise
    finally:
        await close_connection(conn)


async def run_migrations():
    """Run all migrations"""
    try:
        print_info("Starting database migrations...")

        # Initialize pgvector extension
        await initialize_pgvector()

        # Migrate vector dimension to 384 for BGE model
        await migrate_vector_dimension(384)

        # Add indexing status columns
        await add_indexing_status_columns()

        print_info("Database migrations completed successfully")
    except Exception as e:
        print_error(f"Error during migrations: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_migrations())
