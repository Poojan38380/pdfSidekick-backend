import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from datetime import datetime

from database import (
    get_connection,
    close_connection,
    initialize_database,
)
from api import api_router
from utils.colorLogger import (
    print_info,
    print_error,
    delete_logs,
    get_user_input,
    print_header,
)
from websocket.chat_server import chat_websocket_endpoint

start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print_info("Starting application")
        db_pool = await get_connection()

        await initialize_database(db_pool)

        app.state.db_pool = db_pool
        print_info("Database initialized")
        yield
    except Exception as e:
        print_error(f"Error during startup: {e}")
        raise
    finally:
        print_info("Shutting down application")
        if hasattr(app.state, "db_pool"):
            await close_connection(app.state.db_pool)


app = FastAPI(
    title="PDFSideKick API",
    description="API for PDFSideKick backend",
    version="0.1.0",
    lifespan=lifespan,
)


origins = [
    "http://localhost:3000",  # Frontend development server
    "http://localhost:8000",  # Backend development server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db_from_request(request: Request):
    return request.app.state.db_pool


app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Welcome to PDFSideKick API"}


@app.get("/health")
async def health_check(request: Request):
    current_time = time.time()
    uptime_seconds = current_time - start_time

    db_status = "healthy"
    try:
        db_pool = request.app.state.db_pool
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        print_error(f"Error during health check (in health_check): {e}")
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "uptime_seconds": uptime_seconds,
        "database": db_status,
    }


@app.websocket("/ws/chat/{pdf_id}")
async def websocket_endpoint(websocket: WebSocket, pdf_id: str):
    await chat_websocket_endpoint(websocket, pdf_id)


if __name__ == "__main__":
    print_header("Welcome to the PDFSideKick API")
    print_info("Do you want to clear logs? (y/n)")
    choice = get_user_input("Enter your choice: ")
    if choice == "y":
        delete_logs()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
