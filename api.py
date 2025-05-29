"""
FastAPI wrapper for Scrapr - Docker-ready version
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from scrapr_main import ScraprExtractor

# Create FastAPI app
app = FastAPI(
    title="Scrapr API",
    description="PDF extraction service with OCR support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize extractor
extractor = ScraprExtractor()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Scrapr API",
        "version": "1.0.0",
        "ocr_available": True
    }


@app.post("/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    output_format: str = "markdown",
    use_ocr: bool = True
):
    """
    Extract content from uploaded PDF
    
    Args:
        file: PDF file to process
        output_format: Output format (markdown or json)
        use_ocr: Whether to use OCR for scanned pages
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract content
        result = extractor.extract_pdf(str(input_path), use_ocr=use_ocr)
        
        if output_format == "markdown":
            # Save as markdown
            output_filename = f"{timestamp}_{file.filename.replace('.pdf', '.md')}"
            output_path = OUTPUT_DIR / output_filename
            extractor.save_as_markdown(result, str(output_path))
            
            # Return markdown file
            return FileResponse(
                path=str(output_path),
                filename=output_filename,
                media_type="text/markdown"
            )
        else:
            # Return JSON
            return JSONResponse(content={
                "filename": file.filename,
                "title": result['title'],
                "total_pages": result['total_pages'],
                "ocr_used": result['ocr_used'],
                "content": result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content'],
                "full_content_length": len(result['content'])
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up input file
        if input_path.exists():
            input_path.unlink()


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if Tesseract is available
        import pytesseract
        tesseract_version = pytesseract.get_tesseract_version()
        
        return {
            "status": "healthy",
            "tesseract_version": str(tesseract_version),
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
            "disk_space": shutil.disk_usage("/").free // (1024**3)  # GB free
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
