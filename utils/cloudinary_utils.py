import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from utils.colorLogger import print_error

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

        # Create a viewable PDF URL by modifying the secure_url
        if "secure_url" in upload_result:
            # Transform the URL to make it viewable instead of downloadable
            # Format: /image/upload/ -> /image/upload/fl_attachment:false/
            viewable_url = upload_result["secure_url"].replace(
                "/upload/", "/upload/fl_attachment:false/"
            )
            upload_result["viewable_url"] = viewable_url

        return upload_result
    except cloudinary.exceptions.Error as ce:
        print_error(ce)
        raise Exception(f"Cloudinary upload failed: {str(ce)}")
    except ValueError as ve:
        print_error(ve)
        raise Exception(f"Invalid input: {str(ve)}")

    except Exception as e:
        print_error(e)
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
        print_error(e)
        raise Exception(f"Cloudinary deletion failed: {str(e)}")
