import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from database import get_connection, close_connection, initialize_database
from api import api_router
from utils.colorLogger import (
    print_info,
    print_error,
    delete_logs,
    get_user_input,
    print_header,
)


# Define lifespan event handlers for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    try:
        print_info("Starting application")
        db_pool = await get_connection()

        # Initialize database with tables
        await initialize_database(db_pool)

        app.state.db_pool = db_pool
        print_info("Database initialized")

        yield
    except Exception as e:
        print_error(f"Error during startup: {e}")
        # Re-raise the exception to halt the startup process
        raise
    finally:
        # Shutdown: Close DB connections
        print_info("Shutting down application")
        if hasattr(app.state, "db_pool"):
            await close_connection(app.state.db_pool)


# Create FastAPI application
app = FastAPI(
    title="PDFSideKick API",
    description="API for PDFSideKick backend",
    version="0.1.0",
    lifespan=lifespan,
)


# Configure CORS
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


# Dependency to get database connection from request
async def get_db_from_request(request: Request):
    return request.app.state.db_pool


# Include API routes
app.include_router(api_router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to PDFSideKick API"}


if __name__ == "__main__":
    print_header("Welcome to the PDFSideKick API")
    print_info("Do you want to clear logs? (y/n)")
    choice = get_user_input("Enter your choice: ")
    if choice == "y":
        delete_logs()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
