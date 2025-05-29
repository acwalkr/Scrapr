import os
import sys
import logging
from core.pdf_extractor import PDFExtractor
from core.doc_extractor import DocExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_type(file_path):
    """
    Identifies the file type based on the file extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file type (e.g., "pdf", "docx", "txt").

    Raises:
        ValueError: If the file type is unsupported.
    """
    _, file_extension = os.path.splitext(file_path)
    ext_lower = file_extension.lower()
    
    if ext_lower == ".pdf":
        return "pdf"
    elif ext_lower in [".docx", ".doc"]:
        return "docx"
    elif ext_lower == ".txt":
        return "txt"
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def extract_text_from_pdf(file_path, enable_ocr=True):
    """
    Extracts text from a PDF file with OCR support.

    Args:
        file_path (str): The path to the PDF file.
        enable_ocr (bool): Enable OCR for scanned documents.

    Returns:
        str: The extracted text in markdown format.
    """
    extractor = PDFExtractor(enable_ocr=enable_ocr)
    return extractor.extract_to_markdown(file_path)

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text in markdown format.
    """
    extractor = DocExtractor()
    return extractor.extract_to_markdown(file_path)

def extract_text_from_txt(file_path):
    """
    Extracts text from a TXT file.

    Args:
        file_path (str): The path to the TXT file.

    Returns:
        str: The extracted text in markdown format.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return f"# {os.path.basename(file_path)}\n\n{text}"
    except Exception as e:
        logger.error(f"Error reading text from TXT {file_path}: {e}")
        return ""

def save_to_markdown(text, output_path):
    """
    Saves the given text to a Markdown file.

    Args:
        text (str): The text to save.
        output_path (str): The path to the output Markdown file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Successfully saved extracted text to {output_path}")
    except Exception as e:
        logger.error(f"Error saving text to Markdown {output_path}: {e}")

def scrape_file(file_path, enable_ocr=True):
    """
    Main function to scrape a file.

    Args:
        file_path (str): The path to the file to scrape.
        enable_ocr (bool): Enable OCR for scanned PDFs.
    """
    try:
        file_type = get_file_type(file_path)
        extracted_text = ""
        
        logger.info(f"Processing {file_type} file: {file_path}")
        
        if file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_path, enable_ocr)
        elif file_type == "docx":
            extracted_text = extract_text_from_docx(file_path)
        elif file_type == "txt":
            extracted_text = extract_text_from_txt(file_path)

        if extracted_text:
            base, _ = os.path.splitext(file_path)
            output_markdown_path = base + ".md"
            save_to_markdown(extracted_text, output_markdown_path)
            logger.info(f"Extraction complete. Output saved to: {output_markdown_path}")
        else:
            logger.warning(f"No text extracted from {file_path}")

    except ValueError as ve:
        logger.error(f"Error processing {file_path}: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path}: {e}")

if __name__ == "__main__":
    # Check for command line arguments
    enable_ocr = True
    
    if "--no-ocr" in sys.argv:
        enable_ocr = False
        sys.argv.remove("--no-ocr")
    
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
    else:
        input_file_path = input("Please enter the path to the file: ")
    
    if os.path.exists(input_file_path):
        scrape_file(input_file_path, enable_ocr)
    else:
        logger.error(f"Error: The file '{input_file_path}' does not exist.")
