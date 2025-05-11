import os
import asyncpg
from typing import Optional
from dotenv import load_dotenv
from utils.colorLogger import print_info, print_error
import asyncio

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
POOL_CONFIG = {
    "min_size": 5,
    "max_size": 20,
    "command_timeout": 60,
    "max_inactive_connection_lifetime": 300.0,
}


async def get_connection() -> Optional[asyncpg.Pool]:
    """
    Create and return a database connection pool with retry logic.

    Returns:
        Optional[asyncpg.Pool]: Database connection pool or None if connection fails
    """
    for attempt in range(MAX_RETRIES):
        try:
            pool = await asyncpg.create_pool(DATABASE_URL, **POOL_CONFIG)
            print_info(
                f"Successfully connected to database (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            return pool
        except Exception as e:
            print_error(
                f"Error connecting to the database (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_DELAY)
    return None


async def close_connection(pool: Optional[asyncpg.Pool]) -> None:
    """
    Close the database connection pool safely.

    Args:
        pool (Optional[asyncpg.Pool]): The connection pool to close
    """
    if pool:
        try:
            await pool.close()
            print_info("Database connection pool closed successfully")
        except Exception as e:
            print_error(f"Error closing database connection pool: {e}")
            raise


async def create_db_pool() -> Optional[asyncpg.Pool]:
    """
    Create and return a database connection pool.
    This is an alias for get_connection for compatibility.

    Returns:
        Optional[asyncpg.Pool]: Database connection pool or None if connection fails
    """
    return await get_connection()
