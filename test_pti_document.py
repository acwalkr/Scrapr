#!/usr/bin/env python3
"""
Test script for PTI DC10.1-08 Design of Post-Tensioned Slabs-on-Ground
This will test Scrapr's ability to extract complex engineering documentation
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.extractors.pdf_extractor import PDFExtractor
from core.extractors.word_extractor import WordExtractor
from core.processors.markdown_processor import MarkdownProcessor

def test_pti_extraction():
    """Test extraction of PTI engineering document"""
    
    # Define paths
    pdf_path = r"C:\Users\alex.walker\Desktop\AI Project\Strand\PTI Book\PTI DC10.1-08 Design of Post-Tensioned Slabs-on-Ground (1).pdf"
    output_dir = r"C:\Users\alex.walker\Scrapr\test_runs"
    obsidian_path = r"C:\Users\alex.walker\Documents\Obsidian\Strand\AI Tools\Projects\Scrapr\Extracted Documents"
    
    print("=" * 60)
    print("SCRAPR - PTI Document Extraction Test")
    print("=" * 60)
    print(f"\nDocument: {os.path.basename(pdf_path)}")
    print(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")
    
    # Initialize extractor
    pdf_extractor = PDFExtractor()
    
    print("\n1. Attempting text extraction...")
    start_time = time.time()
    
    try:
        # Extract content
        result = pdf_extractor.extract(pdf_path)
        
        extraction_time = time.time() - start_time
        print(f"   ✓ Extraction completed in {extraction_time:.2f} seconds")
        
        # Display extraction statistics
        print(f"\n2. Extraction Results:")
        print(f"   - Title: {result.get('title', 'Not detected')}")
        print(f"   - Pages: {result.get('num_pages', 'Unknown')}")
        print(f"   - Text length: {len(result.get('content', ''))} characters")
        print(f"   - Tables found: {len(result.get('tables', []))}")
        print(f"   - Images found: {len(result.get('images', []))}")
        print(f"   - OCR used: {'Yes' if result.get('ocr_used', False) else 'No'}")
        
        if result.get('ocr_confidence'):
            print(f"   - OCR confidence: {result['ocr_confidence']:.2f}")
        
        # Create Obsidian directory if it doesn't exist
        os.makedirs(obsidian_path, exist_ok=True)
        
        # Process to Markdown
        print(f"\n3. Converting to Markdown...")
        processor = MarkdownProcessor()
        markdown_content = processor.process(result)
        
        # Save to test directory
        test_output_path = os.path.join(output_dir, "PTI_DC10_extracted.md")
        with open(test_output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"   ✓ Saved to test directory: {test_output_path}")
        
        # Save to Obsidian vault
        obsidian_output_path = os.path.join(obsidian_path, "PTI DC10.1-08 - Post-Tensioned Slabs.md")
        with open(obsidian_output_path, 'w', encoding='utf-8') as f:
            # Add Obsidian metadata
            obsidian_header = f"""---
tags: [engineering, PTI, post-tensioned, slabs, code-book]
source: PTI DC10.1-08
type: technical-standard
extracted: {time.strftime('%Y-%m-%d %H:%M')}
---

# PTI DC10.1-08 - Design of Post-Tensioned Slabs-on-Ground

[[AI Tools Hub|← Back to AI Tools Hub]]

> **Note**: This document was automatically extracted using Scrapr with {"OCR" if result.get('ocr_used') else "text extraction"}.

---

"""
            f.write(obsidian_header + markdown_content)
        
        print(f"   ✓ Saved to Obsidian vault: {obsidian_output_path}")
        
        # Show sample of extracted content
        print(f"\n4. Sample of extracted content:")
        print("   " + "-" * 50)
        sample = result.get('content', '')[:500].replace('\n', '\n   ')
        print(f"   {sample}...")
        print("   " + "-" * 50)
        
        # If tables were found, show first table
        if result.get('tables'):
            print(f"\n5. Sample table (first table found):")
            print("   " + "-" * 50)
            first_table = result['tables'][0]
            if isinstance(first_table, list) and len(first_table) > 0:
                for row in first_table[:5]:  # Show first 5 rows
                    print(f"   {row}")
            print("   " + "-" * 50)
        
        print(f"\n✓ SUCCESS: Document extracted and saved!")
        print(f"\nNext steps:")
        print(f"1. Review the extracted content in Obsidian")
        print(f"2. Check if tables and formulas were captured correctly")
        print(f"3. Verify any images or diagrams references")
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERROR during extraction: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"1. Check if the PDF file exists and is readable")
        print(f"2. Ensure all dependencies are installed")
        print(f"3. If OCR is needed, ensure Tesseract is installed")
        raise

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(r"C:\Users\alex.walker\Documents\Obsidian\Strand\AI Tools\Projects\Scrapr\Extracted Documents", exist_ok=True)
    
    # Run the test
    result = test_pti_extraction()
    
    # Optionally test vector embedding generation
    print("\n" + "=" * 60)
    print("OPTIONAL: Generate Vector Embeddings")
    print("=" * 60)
    
    user_input = input("\nWould you like to generate vector embeddings? (y/n): ")
    if user_input.lower() == 'y':
        try:
            from sentence_transformers import SentenceTransformer
            
            print("\nGenerating embeddings...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Split content into chunks (simple splitting for now)
            content = result.get('content', '')
            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]  # 800 char chunks with overlap
            
            embeddings = model.encode(chunks[:10])  # Test with first 10 chunks
            print(f"✓ Generated embeddings for {len(embeddings)} chunks")
            print(f"  Embedding dimension: {len(embeddings[0])}")
            
        except ImportError:
            print("✗ sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
