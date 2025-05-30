"""
Enhanced FastAPI wrapper for Scrapr with improved image extraction
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from scrapr.extractors.enhanced_pdfplumber_extractor import EnhancedContextualPDFExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Scrapr Enhanced API",
    description="Advanced PDF extraction service with image extraction and context preservation",
    version="2.0.0"
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
UPLOAD_DIR = Path("/app/input") if os.path.exists("/app") else Path("./uploads")
OUTPUT_DIR = Path("/app/output") if os.path.exists("/app") else Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Scrapr Enhanced API",
        "version": "2.0.0",
        "features": {
            "ocr_available": True,
            "image_extraction": True,
            "table_extraction": True,
            "context_preservation": True,
            "base64_embedding": True
        }
    }


@app.post("/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    output_format: str = Form("markdown"),
    use_ocr: bool = Form(True),
    extract_images: bool = Form(True),
    embed_images_base64: bool = Form(True),
    preserve_context: bool = Form(True),
    ocr_dpi: int = Form(300),
    image_dpi: int = Form(200)
):
    """
    Extract content from uploaded PDF with enhanced capabilities
    
    Args:
        file: PDF file to process
        output_format: Output format (markdown or json)
        use_ocr: Whether to use OCR for scanned/corrupted pages
        extract_images: Whether to extract images from PDF
        embed_images_base64: Whether to embed images as base64 in markdown
        preserve_context: Whether to preserve element relationships
        ocr_dpi: DPI for OCR processing (higher = better quality but slower)
        image_dpi: DPI for image extraction
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
        
        logger.info(f"Processing PDF: {file.filename}")
        
        # Initialize enhanced extractor
        extractor = EnhancedContextualPDFExtractor(
            ocr_dpi=ocr_dpi,
            image_dpi=image_dpi,
            extract_images=extract_images,
            extract_tables=True,
            embed_images_base64=embed_images_base64,
            preserve_context=preserve_context
        )
        
        # Extract content
        result = extractor.extract_from_pdf(str(input_path))
        
        if output_format == "markdown":
            # Save as markdown
            output_filename = f"{timestamp}_{file.filename.replace('.pdf', '_enhanced.md')}"
            output_path = OUTPUT_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['markdown'])
            
            logger.info(f"Extraction complete. Statistics: {result['statistics']}")
            
            # Return markdown file
            return FileResponse(
                path=str(output_path),
                filename=output_filename,
                media_type="text/markdown"
            )
        else:
            # Return JSON with all details
            return JSONResponse(content={
                "filename": file.filename,
                "metadata": result['metadata'],
                "statistics": result['statistics'],
                "total_pages": result['statistics']['total_pages'],
                "total_images": result['statistics']['total_images'],
                "total_tables": result['statistics']['total_tables'],
                "ocr_pages": result['statistics']['ocr_pages'],
                "content_preview": result['markdown'][:2000] + "..." if len(result['markdown']) > 2000 else result['markdown'],
                "full_content_length": len(result['markdown'])
            })
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up input file
        if input_path.exists():
            input_path.unlink()


@app.post("/batch_extract")
async def batch_extract_pdfs(
    files: list[UploadFile] = File(...),
    output_format: str = Form("markdown"),
    extract_images: bool = Form(True),
    embed_images_base64: bool = Form(True)
):
    """
    Extract content from multiple PDFs
    
    Args:
        files: List of PDF files to process
        output_format: Output format (markdown or json)
        extract_images: Whether to extract images from PDFs
        embed_images_base64: Whether to embed images as base64
    """
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize extractor once for all files
    extractor = EnhancedContextualPDFExtractor(
        extract_images=extract_images,
        embed_images_base64=embed_images_base64,
        preserve_context=True
    )
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Not a PDF file"
            })
            continue
        
        input_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
        
        try:
            # Save file
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract content
            result = extractor.extract_from_pdf(str(input_path))
            
            if output_format == "markdown":
                output_filename = f"{timestamp}_{file.filename.replace('.pdf', '_enhanced.md')}"
                output_path = OUTPUT_DIR / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result['markdown'])
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "output_file": output_filename,
                    "statistics": result['statistics']
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "content": result['markdown'][:1000] + "...",
                    "statistics": result['statistics']
                })
                
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
        
        finally:
            if input_path.exists():
                input_path.unlink()
    
    return JSONResponse(content={"results": results})


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if Tesseract is available
        import pytesseract
        tesseract_version = pytesseract.get_tesseract_version()
        
        # Check if pdf2image works
        from pdf2image import convert_from_path
        pdf2image_available = True
        
        # Check available libraries
        libraries = {
            "pdfplumber": True,
            "pytesseract": True,
            "pdf2image": pdf2image_available,
            "numpy": True,
            "PIL": True
        }
        
        try:
            import camelot
            libraries["camelot"] = True
        except:
            libraries["camelot"] = False
        
        return {
            "status": "healthy",
            "tesseract_version": str(tesseract_version),
            "libraries": libraries,
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
            "disk_space_gb": shutil.disk_usage(str(UPLOAD_DIR)).free // (1024**3)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/test_page")
async def test_page():
    """Simple HTML test page for the API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scrapr Enhanced API Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { 
                border: 2px dashed #ccc; 
                padding: 20px; 
                text-align: center; 
                margin: 20px 0;
            }
            .options { margin: 20px 0; }
            .option { margin: 10px 0; }
            button { 
                background-color: #4CAF50; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                cursor: pointer; 
                font-size: 16px;
            }
            button:hover { background-color: #45a049; }
            .results { 
                margin-top: 20px; 
                padding: 20px; 
                background-color: #f0f0f0; 
                white-space: pre-wrap; 
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Scrapr Enhanced API Test</h1>
            <p>Upload a PDF to test the enhanced extraction capabilities</p>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".pdf" />
            </div>
            
            <div class="options">
                <div class="option">
                    <label>
                        <input type="checkbox" id="extractImages" checked />
                        Extract Images
                    </label>
                </div>
                <div class="option">
                    <label>
                        <input type="checkbox" id="embedBase64" checked />
                        Embed Images as Base64
                    </label>
                </div>
                <div class="option">
                    <label>
                        <input type="checkbox" id="useOcr" checked />
                        Use OCR if needed
                    </label>
                </div>
                <div class="option">
                    <label>
                        Output Format:
                        <select id="outputFormat">
                            <option value="markdown">Markdown</option>
                            <option value="json">JSON</option>
                        </select>
                    </label>
                </div>
            </div>
            
            <button onclick="extractPDF()">Extract PDF</button>
            
            <div id="results" class="results"></div>
        </div>
        
        <script>
            async function extractPDF() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a PDF file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('extract_images', document.getElementById('extractImages').checked);
                formData.append('embed_images_base64', document.getElementById('embedBase64').checked);
                formData.append('use_ocr', document.getElementById('useOcr').checked);
                formData.append('output_format', document.getElementById('outputFormat').value);
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.style.display = 'block';
                resultsDiv.textContent = 'Processing...';
                
                try {
                    const response = await fetch('/extract', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        if (document.getElementById('outputFormat').value === 'markdown') {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = file.name.replace('.pdf', '_enhanced.md');
                            a.click();
                            resultsDiv.textContent = 'Download started!';
                        } else {
                            const data = await response.json();
                            resultsDiv.textContent = JSON.stringify(data, null, 2);
                        }
                    } else {
                        const error = await response.json();
                        resultsDiv.textContent = 'Error: ' + error.detail;
                    }
                } catch (error) {
                    resultsDiv.textContent = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
