"""
Enhanced PDF Extractor with OCR capabilities for engineering documents
"""
import os
import re
from typing import Dict, List, Any, Optional
import logging

# PDF Processing
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract content from PDF files with OCR fallback"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize PDF extractor
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Default paths for Tesseract
            if os.name == 'nt':  # Windows
                default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                if os.path.exists(default_path):
                    pytesseract.pytesseract.tesseract_cmd = default_path
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract content from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        logger.info(f"Starting extraction for: {file_path}")
        
        result = {
            'title': os.path.basename(file_path),
            'content': '',
            'tables': [],
            'images': [],
            'metadata': {},
            'num_pages': 0,
            'ocr_used': False,
            'ocr_confidence': None
        }
        
        try:
            # Try text extraction first with pdfplumber
            content, tables = self._extract_with_pdfplumber(file_path)
            
            if content and len(content.strip()) > 100:
                logger.info("Text extraction successful with pdfplumber")
                result['content'] = content
                result['tables'] = tables
            else:
                # Fall back to PyMuPDF with OCR
                logger.info("Falling back to PyMuPDF with OCR")
                content, images, ocr_confidence = self._extract_with_pymupdf_ocr(file_path)
                result['content'] = content
                result['images'] = images
                result['ocr_used'] = True
                result['ocr_confidence'] = ocr_confidence
            
            # Get metadata
            result['metadata'] = self._extract_metadata(file_path)
            result['num_pages'] = result['metadata'].get('pages', 0)
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise
        
        return result    
    def _extract_with_pdfplumber(self, file_path: str) -> tuple:
        """Extract text and tables using pdfplumber"""
        text_content = []
        all_tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"=== Page {i+1} ===\n{page_text}")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            all_tables.append({
                                'page': i+1,
                                'data': table
                            })
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        return '\n\n'.join(text_content), all_tables
    
    def _extract_with_pymupdf_ocr(self, file_path: str) -> tuple:
        """Extract content using PyMuPDF with OCR for scanned pages"""
        text_content = []
        images_info = []
        ocr_confidences = []
        
        try:
            pdf = fitz.open(file_path)
            
            for page_num, page in enumerate(pdf):
                # First try to get text
                text = page.get_text()
                
                if len(text.strip()) < 50:  # Likely a scanned page
                    # Convert page to image and OCR
                    logger.info(f"Performing OCR on page {page_num + 1}")
                    
                    # Render page at high resolution
                    mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # OCR the image
                    try:
                        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                        text = pytesseract.image_to_string(img)
                        
                        # Calculate confidence
                        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                        if confidences:
                            ocr_confidences.extend(confidences)
                        
                    except Exception as ocr_error:
                        logger.error(f"OCR failed for page {page_num + 1}: {ocr_error}")
                        text = ""
                
                if text:
                    text_content.append(f"=== Page {page_num + 1} ===\n{text}")
                
                # Check for images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    images_info.append({
                        'page': page_num + 1,
                        'index': img_index
                    })
            
            pdf.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        # Calculate average OCR confidence
        avg_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
        
        return '\n\n'.join(text_content), images_info, avg_confidence
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = {}
        
        try:
            pdf = fitz.open(file_path)
            metadata = {
                'pages': pdf.page_count,
                'title': pdf.metadata.get('title', ''),
                'author': pdf.metadata.get('author', ''),
                'subject': pdf.metadata.get('subject', ''),
                'creator': pdf.metadata.get('creator', ''),
                'producer': pdf.metadata.get('producer', ''),
                'creation_date': str(pdf.metadata.get('creationDate', '')),
                'modification_date': str(pdf.metadata.get('modDate', ''))
            }
            pdf.close()
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata


# Add missing import at the top of the file
import io