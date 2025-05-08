import asyncio
from database import get_connection, close_connection

async def main():
   
    pool = await get_connection()
    await close_connection(pool)

if __name__ == "__main__":
    asyncio.run(main())
