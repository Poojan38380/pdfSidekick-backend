import os
import tempfile
from typing import List, Dict, Any
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from utils.colorLogger import print_info, print_error


async def perform_ocr_on_pdf(pdf_content: bytes) -> Dict[int, str]:
    """
    Extract text from a PDF using OCR when regular text extraction fails

    Args:
        pdf_content: The PDF file content as bytes

    Returns:
        Dictionary mapping page numbers to extracted text
    """
    try:
        # Create a temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF to images
            print_info("Converting PDF to images for OCR processing")
            images = convert_from_bytes(
                pdf_content,
                dpi=300,  # Higher DPI for better OCR quality
                output_folder=temp_dir,
                fmt="jpeg",
                thread_count=os.cpu_count() or 1,
            )

            # Process each page with OCR
            page_texts = {}
            for i, image in enumerate(images):
                page_num = i + 1  # 1-indexed page numbers
                print_info(f"Performing OCR on page {page_num}")

                # Perform OCR
                text = pytesseract.image_to_string(image)
                page_texts[page_num] = text

            return page_texts

    except Exception as e:
        print_error(f"Error performing OCR on PDF: {e}")
        raise


async def extract_text_with_ocr_fallback(
    pdf_content: bytes, extracted_text: Dict[int, str]
) -> Dict[int, str]:
    """
    Use OCR as a fallback for pages where text extraction failed

    Args:
        pdf_content: The PDF file content as bytes
        extracted_text: Dictionary of already extracted text by page number

    Returns:
        Updated dictionary with OCR text for pages that had no text
    """
    try:
        # Check which pages need OCR (empty or very short text)
        pages_needing_ocr = []
        for page_num, text in extracted_text.items():
            if not text or len(text.strip()) < 50:  # Arbitrary threshold
                pages_needing_ocr.append(page_num)

        if not pages_needing_ocr:
            return extracted_text

        print_info(
            f"Using OCR for {len(pages_needing_ocr)} pages with insufficient text"
        )

        # Create a temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF to images
            images = convert_from_bytes(
                pdf_content,
                dpi=300,
                output_folder=temp_dir,
                fmt="jpeg",
                thread_count=os.cpu_count() or 1,
            )

            # Process only pages that need OCR
            for page_num in pages_needing_ocr:
                if page_num <= len(images):
                    image = images[page_num - 1]  # Convert to 0-indexed
                    print_info(f"Performing OCR on page {page_num}")
                    text = pytesseract.image_to_string(image)
                    extracted_text[page_num] = text

            return extracted_text

    except Exception as e:
        print_error(f"Error in OCR fallback: {e}")
        # Return original extraction results if OCR fails
        return extracted_text
