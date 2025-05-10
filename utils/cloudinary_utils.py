import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()


cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True
)


async def upload_pdf_to_cloudinary(file_content, public_id=None):
    """
    Upload a PDF file to Cloudinary

    Args:
        file_content: The PDF file content
        public_id: Optional custom public ID for the file

    Returns:
        dict: Cloudinary upload response containing URL and other details
    """
    try:
        upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET")

        if not cloud_name:
            raise ValueError("Cloudinary cloud name is not configured")

        if not file_content:
            raise ValueError("File content is empty")

        upload_result = cloudinary.uploader.unsigned_upload(file_content, upload_preset)

        return upload_result
    except cloudinary.exceptions.Error as ce:
        raise Exception(f"Cloudinary upload failed: {str(ce)}")
    except ValueError as ve:
        raise Exception(f"Invalid input: {str(ve)}")
    except Exception as e:
        raise Exception(f"Cloudinary upload failed: {str(e)}")


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
        raise Exception(f"Cloudinary deletion failed: {str(e)}")
