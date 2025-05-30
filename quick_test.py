"""
Quick test script for enhanced PDF extraction
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_extraction(pdf_path: str):
    """Test basic PDF extraction"""
    print(f"Testing extraction on: {pdf_path}")
    
    # First, let's check what we have
    try:
        import pdfplumber
        print("✓ pdfplumber is installed")
    except ImportError:
        print("✗ pdfplumber is NOT installed")
        
    try:
        import pytesseract
        print("✓ pytesseract is installed")
    except ImportError:
        print("✗ pytesseract is NOT installed")
        
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image is installed")
    except ImportError:
        print("✗ pdf2image is NOT installed")
    
    # Now try to import our extractor
    try:
        from scrapr.extractors.enhanced_pdfplumber_extractor import EnhancedContextualPDFExtractor
        print("✓ Enhanced extractor imported successfully")
        
        # Try to extract
        extractor = EnhancedContextualPDFExtractor(
            extract_images=True,
            embed_images_base64=True,
            preserve_context=True
        )
        
        results = extractor.extract_from_pdf(pdf_path)
        
        # Save output
        output_path = Path(pdf_path).stem + "_extracted.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['markdown'])
            
        print(f"\n✓ Extraction complete!")
        print(f"  Output saved to: {output_path}")
        print(f"  Statistics: {results['statistics']}")
        
    except Exception as e:
        print(f"\n✗ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            test_basic_extraction(pdf_path)
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python quick_test.py <path_to_pdf>")
