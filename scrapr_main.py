"""
Simplified Scrapr main module for Docker deployment
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# PDF Processing
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScraprExtractor:
    """Main Scrapr extractor with OCR support"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize extractor with Tesseract path"""
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_pdf(self, file_path: str, use_ocr: bool = True) -> Dict[str, Any]:
        """
        Extract content from PDF with OCR fallback
        
        Args:
            file_path: Path to PDF file
            use_ocr: Whether to use OCR for scanned pages
            
        Returns:
            Dictionary with extracted content
        """
        logger.info(f"Processing: {file_path}")
        
        result = {
            'title': os.path.basename(file_path),
            'content': '',
            'pages': [],
            'metadata': {},
            'ocr_used': False,
            'total_pages': 0
        }
        
        try:
            # Open PDF with PyMuPDF
            pdf = fitz.open(file_path)
            result['total_pages'] = pdf.page_count
            result['metadata'] = pdf.metadata
            
            # Process each page
            for page_num in range(pdf.page_count):
                logger.info(f"Processing page {page_num + 1}/{pdf.page_count}")
                page = pdf[page_num]
                
                # Try text extraction first
                text = page.get_text()
                
                # Check if text is garbled (contains many cid references)
                if self._is_garbled_text(text) and use_ocr:
                    logger.info(f"Page {page_num + 1} needs OCR")
                    text = self._ocr_page(page)
                    result['ocr_used'] = True
                
                result['pages'].append({
                    'page_number': page_num + 1,
                    'text': text
                })
            
            # Combine all pages
            result['content'] = '\n\n'.join([
                f"=== Page {p['page_number']} ===\n{p['text']}" 
                for p in result['pages']
            ])
            
            pdf.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        return result
    
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text contains too many cid references"""
        if not text:
            return True
        # Check for excessive cid references
        cid_count = text.count('(cid:')
        return cid_count > len(text) / 10  # More than 10% cid refs
    
    def _ocr_page(self, page) -> str:
        """OCR a single page"""
        try:
            # Render page at 300 DPI
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def save_as_markdown(self, result: Dict[str, Any], output_path: str):
        """Save extraction result as Markdown"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# {result['title']}\n\n")
            
            # Write metadata
            if result['metadata']:
                f.write("## Metadata\n\n")
                for key, value in result['metadata'].items():
                    if value:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # Write extraction info
            f.write("## Extraction Info\n\n")
            f.write(f"- **Total Pages**: {result['total_pages']}\n")
            f.write(f"- **OCR Used**: {'Yes' if result['ocr_used'] else 'No'}\n\n")
            
            # Write content
            f.write("## Content\n\n")
            f.write(result['content'])


if __name__ == "__main__":
    # Simple CLI interface
    if len(sys.argv) < 2:
        print("Usage: python scrapr.py <pdf_file> [output_file]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else pdf_path.replace('.pdf', '.md')
    
    extractor = ScraprExtractor()
    result = extractor.extract_pdf(pdf_path)
    extractor.save_as_markdown(result, output_path)
    
    print(f"Extraction complete! Saved to: {output_path}")
