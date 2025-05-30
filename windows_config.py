"""
Windows-specific configuration for Scrapr
"""
import os
import pytesseract
from pathlib import Path

# Common Tesseract installation paths on Windows
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\alex.walker\AppData\Local\Tesseract-OCR\tesseract.exe",
]

# Find and set Tesseract path
for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Found Tesseract at: {path}")
        break
else:
    print("WARNING: Tesseract not found. Please install from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")

# Common Poppler paths on Windows  
poppler_paths = [
    r"C:\Program Files\poppler\bin",
    r"C:\Program Files (x86)\poppler\bin",
    r"C:\poppler\bin",
    r"C:\Users\alex.walker\poppler\bin",
]

# Find Poppler
poppler_path = None
for path in poppler_paths:
    if os.path.exists(path):
        poppler_path = path
        print(f"Found Poppler at: {path}")
        break
else:
    print("WARNING: Poppler not found. Please download from:")
    print("http://blog.alivate.com.au/poppler-windows/")

# Export for use in extraction
POPPLER_PATH = poppler_path
