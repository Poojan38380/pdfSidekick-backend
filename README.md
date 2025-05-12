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

## Architecture

### Core Components

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
