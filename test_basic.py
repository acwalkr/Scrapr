"""
Basic PDF text extraction test using only pdfplumber
"""
import sys
import os

def test_basic_pdfplumber(pdf_path):
    """Test basic PDF extraction with pdfplumber only"""
    try:
        import pdfplumber
        
        print(f"Opening PDF: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            
            # Extract first page
            page = pdf.pages[0]
            text = page.extract_text()
            
            if text:
                print("\nFirst 500 characters of extracted text:")
                print("-" * 50)
                print(text[:500])
                print("-" * 50)
                
                # Check if text is garbled
                readable_chars = sum(1 for c in text if c.isalnum() or c.isspace())
                total_chars = len(text)
                readability = readable_chars / total_chars if total_chars > 0 else 0
                
                print(f"\nText readability: {readability:.1%}")
                if readability < 0.5:
                    print("WARNING: Text appears to be garbled. OCR may be needed.")
            else:
                print("No text extracted from first page")
                
    except ImportError:
        print("pdfplumber is not installed. Please run:")
        print("pip install pdfplumber")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            test_basic_pdfplumber(pdf_path)
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python test_basic.py <path_to_pdf>")
