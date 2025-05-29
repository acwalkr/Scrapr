"""
PDF Extractor with OCR fallback for scanned documents
"""
import os
import logging
from typing import List, Dict, Optional
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not installed. OCR functionality will be disabled.")

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        if enable_ocr and not OCR_AVAILABLE:
            logger.warning("OCR requested but dependencies not available")
        
    def extract(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from PDF with OCR fallback for scanned pages
        """
        result = {
            "file_path": file_path,
            "pages": [],
            "metadata": {},
            "extraction_method": "text"
        }
        
        # First attempt: Extract with pdfplumber (best for text PDFs)
        try:
            extracted_text = self._extract_with_pdfplumber(file_path)
            if self._is_valid_extraction(extracted_text):
                result["pages"] = extracted_text
                logger.info(f"Successfully extracted text from {file_path} using pdfplumber")
                return result
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Second attempt: PyMuPDF for complex layouts
        try:
            extracted_text = self._extract_with_pymupdf(file_path)
            if self._is_valid_extraction(extracted_text):
                result["pages"] = extracted_text
                logger.info(f"Successfully extracted text from {file_path} using PyMuPDF")
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Final attempt: OCR for scanned documents
        if self.enable_ocr:
            logger.info("Falling back to OCR extraction")
            result["extraction_method"] = "ocr"
            result["pages"] = self._extract_with_ocr(file_path)
        else:
            logger.warning("OCR is disabled. Cannot extract text from scanned PDF.")
            result["pages"] = [{"page_num": 1, "text": "Unable to extract text. OCR is disabled.", "tables": []}]
        
        return result
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Dict]:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "tables": tables
                })
        return pages
    
    def _extract_with_pymupdf(self, file_path: str) -> List[Dict]:
        pages = []
        doc = fitz.open(file_path)
        
        for i, page in enumerate(doc):
            text = page.get_text()
            # Extract tables if needed
            pages.append({
                "page_num": i + 1,
                "text": text,
                "tables": []
            })
        
        doc.close()
        return pages
    
    def _extract_with_ocr(self, file_path: str) -> List[Dict]:
        if not OCR_AVAILABLE:
            return [{"page_num": 1, "text": "OCR dependencies not installed", "tables": []}]
            
        pages = []
        try:
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300)
            
            for i, image in enumerate(images):
                # Preprocess image for better OCR
                processed_image = self._preprocess_image(image)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(
                    processed_image,
                    config='--psm 6'  # Assume uniform block of text
                )
                
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "tables": [],
                    "ocr_confidence": self._get_ocr_confidence(processed_image)
                })
                
                logger.info(f"OCR extracted page {i+1} with confidence: {pages[-1]['ocr_confidence']:.2f}%")
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            pages = [{"page_num": 1, "text": f"OCR extraction failed: {str(e)}", "tables": []}]
        
        return pages
    
    def _preprocess_image(self, image):
        """Enhance image for better OCR results"""
        # Convert PIL to OpenCV format
        open_cv_image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to get better OCR results
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def _is_valid_extraction(self, pages: List[Dict]) -> bool:
        """Check if extraction produced meaningful text"""
        total_text = "".join(page.get("text", "") for page in pages)
        # Consider valid if we have at least 50 characters of text
        return len(total_text.strip()) > 50
    
    def _get_ocr_confidence(self, image) -> float:
        """Get OCR confidence score"""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.0

    def extract_to_markdown(self, file_path: str) -> str:
        """Extract PDF and convert to markdown format"""
        result = self.extract(file_path)
        
        markdown_content = f"# Document: {os.path.basename(file_path)}\n\n"
        markdown_content += f"**Extraction Method**: {result['extraction_method']}\n\n"
        
        for page in result['pages']:
            markdown_content += f"## Page {page['page_num']}\n\n"
            
            # Add confidence if OCR was used
            if 'ocr_confidence' in page:
                markdown_content += f"*OCR Confidence: {page['ocr_confidence']:.2f}%*\n\n"
            
            # Add text content
            markdown_content += page['text'] + "\n\n"
            
            # Add tables if present
            if page['tables']:
                markdown_content += "### Tables\n\n"
                for table_idx, table in enumerate(page['tables']):
                    markdown_content += f"#### Table {table_idx + 1}\n\n"
                    markdown_content += self._table_to_markdown(table) + "\n\n"
        
        return markdown_content
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert table data to markdown format"""
        if not table:
            return ""
        
        markdown = ""
        for i, row in enumerate(table):
            markdown += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
            if i == 0:  # Add header separator
                markdown += "|" + "|".join([" --- " for _ in row]) + "|\n"
        
        return markdown
