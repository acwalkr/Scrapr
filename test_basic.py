"""
Simple PDF extraction test without heavy dependencies
This will work without PyTorch/sentence-transformers
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_extraction():
    print("=" * 60)
    print("SCRAPR - Basic PDF Test (No AI Dependencies)")
    print("=" * 60)
    
    pdf_path = r"C:\Users\alex.walker\Desktop\AI Project\Strand\PTI Book\PTI DC10.1-08 Design of Post-Tensioned Slabs-on-Ground (1).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at: {pdf_path}")
        return
    
    print(f"\nTesting with: {os.path.basename(pdf_path)}")
    print(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Test with pdfplumber only
        import pdfplumber
        
        print("\nAttempting basic text extraction with pdfplumber...")
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            
            # Extract first page as sample
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            if text:
                print("\n✓ Text extraction successful!")
                print("\nFirst 500 characters:")
                print("-" * 50)
                print(text[:500])
                print("-" * 50)
                
                # Check for tables
                tables = first_page.extract_tables()
                if tables:
                    print(f"\n✓ Found {len(tables)} table(s) on first page")
            else:
                print("\n✗ No text found - this PDF might need OCR")
                print("\nTo use OCR, you'll need:")
                print("1. Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("2. Run the full extraction script")
                
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nInstall basic dependencies with:")
        print("pip install pdfplumber pymupdf")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    test_basic_extraction()