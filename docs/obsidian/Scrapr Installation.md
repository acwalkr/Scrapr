# Scrapr Installation

## Prerequisites
- Python 3.10 or higher
- Tesseract OCR (for OCR functionality)
- Poppler (for PDF to image conversion)

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/acwalkr/Scrapr.git
cd Scrapr
```

### 2. Install System Dependencies

#### Windows
```powershell
# Install Tesseract
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH after installation

# Install Poppler
# Download from: http://blog.alivate.com.au/poppler-windows/
# Extract and add bin folder to PATH
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

#### macOS
```bash
brew install tesseract poppler
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
# Test CLI
python scraper.py --help

# Test API
python -m uvicorn api:app --reload
# Visit http://localhost:8000/docs
```

## Docker Installation

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Manual Docker Build
```bash
docker build -t scrapr:latest .
docker run -p 8001:8000 -v $(pwd)/data:/app/data scrapr:latest
```

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Key settings:
- `ENABLE_OCR`: Enable/disable OCR functionality
- `TESSERACT_PATH`: Path to Tesseract executable
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Next Steps
- [[Scrapr API Documentation|Learn the API]]
- [[Scrapr OCR Configuration|Configure OCR]]
- [[n8n Integration with Scrapr|Set up n8n workflows]]

---
Tags: #scrapr #installation #setup
