# PDFSideKick Backend

Backend API for PDFSideKick application

## Setup

1. Clone the repository
2. Create a virtual environment:

```
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:

```
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory with the following variables:

```
DATABASE_URL=your_neondb_connection_string
```

## Running the Application

Start the server:

```
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Database Models

### User (Already exists in NeonDB)

- `id`: UUID (primary key)
- `created_at`: Timestamp
- `updated_at`: Timestamp
- `username`: String
- `first_name`: String
- `last_name`: String
- `email`: String
- `profile_pic`: String (URL)

### PDF

- `id`: UUID (primary key)
- `created_at`: Timestamp
- `updated_at`: Timestamp
- `title`: String
- `description`: String
- `document_link`: String
- `user_id`: UUID (foreign key to User)

## API Endpoints

### PDFs

- `POST /api/pdfs` - Create a new PDF document entry
- `GET /api/pdfs/user/{user_id}` - Get all PDFs for a specific user
- `GET /api/pdfs/{pdf_id}` - Get details of a specific PDF
