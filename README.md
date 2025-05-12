# PDFSidekick Backend

PDFSidekick is a full-stack application that allows users to upload PDF documents and ask questions regarding their content. The backend processes these documents and utilizes natural language processing to provide accurate answers to users' questions.

![PDFSidekick Banner](public/logo-1500x300.png)

## Features

- PDF document uploading and storage
- PDF text extraction with OCR support
- Document chunking and embedding generation
- Semantic search capabilities
- Real-time chat interface via WebSockets
- Background processing for large documents
- API endpoints for document management

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (with asyncpg)
- **NLP**: Transformers, Hugging Face models
- **Vector Database**: Custom implementation for semantic search
- **File Storage**: Cloudinary
- **Text Processing**: PyPDF2, pdf2image, pytesseract
- **Machine Learning**: Sentence Transformers, PyTorch

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Hugging Face account (for API token)
- Cloudinary account (for PDF storage)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pdfsidekick-backend.git
   cd pdfsidekick-backend
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/pdfsidekick
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   MODELS_CACHE_DIR=./.model_cache
   ```

### Running the Application

Start the application with:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## Architecture

### High-Level Design

PDFSidekick follows a modern, service-oriented architecture that separates concerns and promotes maintainability:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Client App  │ ◄─► │   FastAPI     │ ◄─► │   Database    │
│   (Frontend)  │     │   Backend     │     │  (PostgreSQL) │
└───────────────┘     └───────┬───────┘     └───────────────┘
                              │
                              │
                  ┌───────────┴───────────┐
                  │                       │
         ┌────────▼──────┐       ┌────────▼──────┐
         │   File Storage│       │  NLP Pipeline │
         │  (Cloudinary) │       │ (Transformers)│
         └───────────────┘       └───────────────┘
```

The architecture is designed around these core flows:

1. **Document Processing Flow**: PDF → Text Extraction → Chunking → Embedding Generation → Storage
2. **Query Processing Flow**: User Question → Context Retrieval → LLM Processing → Answer Generation
3. **Real-time Communication Flow**: WebSocket Connection → Question Processing → Streaming Response

### Low-Level Design

The codebase is organized into several key components that work together:

#### Core Components and Interactions

1. **API Layer** (`api/`)

   - `pdfs.py`: Handles all PDF-related endpoints including upload, retrieval, and search
   - Manages request validation, error handling, and response formatting
   - Delegates business logic to utility modules

2. **Database Layer** (`database/`)

   - Provides async database connection pool
   - Implements data access methods for PDFs, chunks, users, and embeddings
   - Handles transaction management and connection lifecycle

3. **PDF Processing Pipeline** (`utils/pdf_processor.py`)

   - Extracts text from PDFs using PyPDF2
   - Applies OCR for image-based content using pytesseract
   - Chunks extracted text into manageable segments
   - Updates processing status in real-time

4. **Vector Database** (`utils/vector_db.py`)

   - Manages creation and storage of document embeddings
   - Performs semantic search using similarity metrics
   - Optimizes search results based on relevance

5. **LLM Client** (`utils/llm_client.py`)

   - Interfaces with Hugging Face transformer models
   - Manages context window and token limits
   - Generates coherent answers based on retrieved context
   - Handles model loading and optimization

6. **Background Processing** (`utils/background_jobs.py`)

   - Manages asynchronous PDF processing tasks
   - Handles embedding generation
   - Updates processing status and progress

7. **WebSocket Server** (`websocket/chat_server.py`)

   - Manages real-time chat connections
   - Processes incoming questions
   - Streams answers back to clients

8. **File Storage** (`utils/cloudinary_utils.py`)
   - Handles PDF uploads to Cloudinary
   - Manages file metadata and access URLs

#### Data Flow

The sequence of operations for a typical user interaction:

1. **PDF Upload**:

   ```
   Client → API (pdfs.py) → Cloudinary Storage → Database → Background Processing
   ```

2. **Document Processing**:

   ```
   Background Job → PDF Processor → Text Extraction → Chunking → Vector DB → Database
   ```

3. **Question Answering**:
   ```
   Client WebSocket → Chat Server → Vector DB (Context Retrieval) → LLM Client (Answer Generation) → Client
   ```

#### Class Relationships

Key class interactions:

- `LLMClient`: Interfaces with transformer models to generate answers based on context
- `PDFProcessor`: Extracts and processes text from uploaded documents
- `VectorDB`: Manages embeddings and semantic search functionality

#### Code Organization

```
pdfsidekick-backend/
├── api/                  # API endpoints
├── database/             # Database access and models
├── schemas/              # Pydantic models for validation
├── utils/                # Utility modules
│   ├── llm_client.py     # LLM interaction
│   ├── pdf_processor.py  # PDF processing pipeline
│   ├── vector_db.py      # Vector database operations
│   └── ...               # Other utilities
├── websocket/            # WebSocket handlers
├── main.py               # Application entry point
└── requirements.txt      # Dependencies
```

This architecture allows for:

- Clear separation of concerns
- Independent scaling of components
- Easy addition of new features
- Maintainable and testable code structure

## API Endpoints

### PDF Management

- `POST /api/pdfs/upload` - Upload a PDF document
- `GET /api/pdfs/user/{user_id}` - Get all PDFs for a specific user
- `GET /api/pdfs/{pdf_id}` - Get details of a specific PDF
- `GET /api/pdfs/{pdf_id}/chunks` - Get all text chunks for a specific PDF
- `GET /api/pdfs/{pdf_id}/processing-status` - Get processing status of a PDF
- `POST /api/pdfs/{pdf_id}/reprocess` - Reprocess a PDF document
- `POST /api/pdfs/{pdf_id}/generate-embeddings` - Generate embeddings for a PDF
- `GET /api/pdfs/search` - Search across PDFs using semantic search
- `GET /api/pdfs/{pdf_id}/indexing-status` - Get indexing status of a PDF

### WebSocket

- `WS /ws/chat/{pdf_id}` - Real-time chat interface for asking questions about a PDF

## Core Components

1. **PDF Processor**: Extracts text from PDFs, applies OCR when needed, and chunks the text into manageable pieces.

2. **LLM Client**: Interfaces with Hugging Face Transformers models to generate answers based on context.

3. **Vector Database**: Stores and searches through vector embeddings for semantic search capabilities.

4. **Background Jobs**: Handles time-consuming tasks like PDF processing and embedding generation.

5. **WebSocket Server**: Provides real-time interaction for the chat interface.

## Local Development

For local development with custom embeddings, refer to the `utils/README_LOCAL_EMBEDDINGS.md` file.

## License

[MIT License](LICENSE)

## Contact

For questions or support, please create an issue in the repository.

## Related Projects

- [PDFSidekick Frontend](https://github.com/Poojan38380/pdfSidekick-frontend) - Frontend
