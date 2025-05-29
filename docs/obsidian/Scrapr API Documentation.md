# Scrapr API Documentation

## Base URL
```
http://localhost:8001
```

## Endpoints

### 1. GET /
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Scrapr API",
  "version": "1.0.0",
  "endpoints": {
    "/extract": "Extract text from single document",
    "/extract_to_markdown": "Extract and convert to markdown",
    "/batch_extract": "Process multiple documents",
    "/health": "Health check"
  }
}
```

### 2. POST /extract
Extract text from a single document.

**Parameters:**
- `file` (form-data, required): Document file to process
- `enable_ocr` (query, optional): Enable OCR for scanned PDFs (default: true)
- `output_format` (query, optional): Output format - "json" or "markdown" (default: "json")

**Example Request:**
```bash
curl -X POST "http://localhost:8001/extract?enable_ocr=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response (JSON format):**
```json
{
  "file_path": "document.pdf",
  "pages": [
    {
      "page_num": 1,
      "text": "Extracted text content...",
      "tables": [],
      "ocr_confidence": 95.5
    }
  ],
  "metadata": {},
  "extraction_method": "ocr"
}
```
### 3. POST /extract_to_markdown
Extract document and return as markdown.

**Parameters:**
- `file` (form-data, required): Document file
- `enable_ocr` (query, optional): Enable OCR (default: true)

**Response:**
```json
{
  "markdown": "# Document: example.pdf\n\n**Extraction Method**: text\n\n## Page 1\n\nContent..."
}
```

### 4. POST /batch_extract
Process multiple documents.

**Parameters:**
- `files` (form-data, required): Multiple document files
- `enable_ocr` (query, optional): Enable OCR (default: true)
- `output_format` (query, optional): "json" or "markdown" (default: "json")

**Response:**
```json
{
  "results": [
    {
      "filename": "doc1.pdf",
      "status": "success",
      "data": { ... }
    },
    {
      "filename": "doc2.docx",
      "status": "success",
      "data": { ... }
    }
  ],
  "total": 2
}
```

### 5. GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "scrapr",
  "ocr_available": true
}
```
## n8n Integration Example

```json
{
  "nodes": [
    {
      "parameters": {
        "method": "POST",
        "url": "http://scrapr:8000/extract",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "file",
              "value": "={{$binary.file}}"
            }
          ]
        },
        "options": {
          "bodyContentType": "multipart/form-data"
        }
      },
      "name": "Extract Document",
      "type": "n8n-nodes-base.httpRequest",
      "position": [250, 300]
    }
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Unsupported file type"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message describing the issue"
}
```

---
Tags: #scrapr #api #documentation #n8n