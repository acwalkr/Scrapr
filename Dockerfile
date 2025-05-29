# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # For PyMuPDF
    libmupdf-dev \
    mupdf-tools \
    # For Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # For pdf2image
    poppler-utils \
    # Build tools
    gcc \
    g++ \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/logs

# Expose port for API
EXPOSE 8001

# Default command runs the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
