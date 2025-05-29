"""
Test script for Scrapr functionality
"""
import requests
import os
import sys

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8001/health")
        print(f"Health Check: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_extract(file_path):
    """Test document extraction"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(
                "http://localhost:8001/extract",
                files=files,
                params={"enable_ocr": True}
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Extraction successful!")
            print(f"Extraction method: {result.get('extraction_method', 'unknown')}")
            
            if 'pages' in result:
                print(f"Pages extracted: {len(result['pages'])}")
                for page in result['pages'][:1]:  # Show first page only
                    text_preview = page.get('text', '')[:200] + "..."
                    print(f"Page {page['page_num']} preview: {text_preview}")
                    if 'ocr_confidence' in page:
                        print(f"OCR Confidence: {page['ocr_confidence']:.2f}%")
            return True
        else:
            print(f"Extraction failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Scrapr API Test ===\n")
    
    # Test health
    print("1. Testing health endpoint...")
    if not test_health():
        print("API is not running. Please start it with: uvicorn api:app --reload")
        sys.exit(1)
    
    print("\n2. Testing document extraction...")
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = input("Enter path to test document: ")
    
    if test_extract(test_file):
        print("\nAll tests passed! ✅")
    else:
        print("\nTests failed! ❌")
