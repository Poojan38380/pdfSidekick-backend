import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from utils.colorLogger import print_error
import uuid
from typing import Dict, Any

load_dotenv()


cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True
)


async def upload_pdf_to_cloudinary(
    pdf_content: bytes, filename: str = None
) -> Dict[str, Any]:
    """
    Upload a PDF file to Cloudinary

    Args:
        pdf_content: PDF file content as bytes
        filename: Original filename (optional)

    Returns:
        Cloudinary response with upload details
    """
    try:
        # Generate a unique filename if none provided
        if not filename:
            filename = f"pdf_{uuid.uuid4()}.pdf"
        elif not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        # Validate file content
        if len(pdf_content) == 0:
            raise ValueError("Empty file content")

        # Upload to Cloudinary
        response = cloudinary.uploader.upload(
            pdf_content,
            resource_type="raw",
            folder="pdfs",
            public_id=os.path.splitext(filename)[0],
            format="pdf",
        )

        return response
    except Exception as e:
        print_error(f"Error uploading to Cloudinary: {e}")
        raise


async def delete_pdf_from_cloudinary(public_id):
    """
    Delete a PDF file from Cloudinary

    Args:
        public_id: The public ID of the file to delete

    Returns:
        dict: Cloudinary deletion response
    """
    try:
        # Delete from Cloudinary
        deletion_result = cloudinary.uploader.destroy(public_id, resource_type="raw")
        return deletion_result
    except Exception as e:
        print_error(e)
        raise Exception(f"Cloudinary deletion failed: {str(e)}")
