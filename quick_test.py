"""
Quick test to verify Scrapr setup before running full PTI extraction
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing Scrapr imports...")
    
    try:
        print("✓ Importing PDF extractor...", end="")
        from core.extractors.pdf_extractor import PDFExtractor
        print(" Success!")
        
        print("✓ Importing Markdown processor...", end="")
        from core.processors.markdown_processor import MarkdownProcessor
        print(" Success!")
        
        print("✓ Importing dependencies...", end="")
        import pdfplumber
        import pytesseract
        import cv2
        print(" Success!")
        
        print("\nAll imports successful! Scrapr is ready to test.")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    if test_imports():
        print("\nYou can now run: python test_pti_document.py")
    else:
        print("\nFix the import errors before proceeding.")