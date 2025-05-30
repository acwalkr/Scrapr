"""
Test script for enhanced PDF extraction with image capabilities
"""
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapr.extractors.enhanced_pdfplumber_extractor import EnhancedContextualPDFExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_extraction(pdf_path: str, output_dir: str = "./test_output"):
    """
    Test the enhanced PDF extraction on a given PDF file
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save output files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get base filename
    base_name = Path(pdf_path).stem
    
    logger.info(f"Testing enhanced extraction on: {pdf_path}")
    
    try:
        # Test 1: Full extraction with embedded base64 images
        logger.info("Test 1: Full extraction with embedded base64 images")
        extractor = EnhancedContextualPDFExtractor(
            extract_images=True,
            embed_images_base64=True,
            preserve_context=True,
            extract_tables=True
        )
        
        results = extractor.extract_from_pdf(pdf_path)
        
        # Save markdown with embedded images
        output_path = output_dir / f"{base_name}_embedded_images.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['markdown'])
        
        logger.info(f"Saved to: {output_path}")
        logger.info(f"Statistics: {results['statistics']}")
        
        # Test 2: Extraction with separate image files
        logger.info("\nTest 2: Extraction with separate image files")
        extractor = EnhancedContextualPDFExtractor(
            extract_images=True,
            embed_images_base64=False,
            preserve_context=True,
            extract_tables=True
        )
        
        results = extractor.extract_from_pdf(pdf_path)
        
        # Save markdown with image references
        output_path = output_dir / f"{base_name}_image_references.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['markdown'])
        
        logger.info(f"Saved to: {output_path}")
        
        # Test 3: Create a summary report
        logger.info("\nCreating extraction summary report")
        
        summary_path = output_dir / f"{base_name}_extraction_report.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# PDF Extraction Report\n\n")
            f.write(f"**File:** {pdf_path}\n\n")
            f.write(f"## Extraction Statistics\n\n")
            
            stats = results['statistics']
            f.write(f"- **Total Pages:** {stats['total_pages']}\n")
            f.write(f"- **Text Blocks Extracted:** {stats['total_text_blocks']}\n")
            f.write(f"- **Images Extracted:** {stats['total_images']}\n")
            f.write(f"- **Tables Extracted:** {stats['total_tables']}\n")
            f.write(f"- **Pages Requiring OCR:** {stats['ocr_pages']}\n")
            f.write(f"- **Figure References Found:** {stats.get('figure_references', 0)}\n\n")
            
            # Add information about extracted elements
            f.write(f"## Page-by-Page Analysis\n\n")
            
            for page_data in results['pages']:
                f.write(f"### Page {page_data['page_num']}\n\n")
                
                if page_data.get('used_ocr'):
                    f.write("- **OCR Used:** Yes\n")
                
                # Count element types
                element_counts = {}
                for elem in page_data['elements']:
                    elem_type = elem.type
                    element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
                
                for elem_type, count in element_counts.items():
                    f.write(f"- **{elem_type.title()}s:** {count}\n")
                
                f.write("\n")
            
            # Add relationship analysis
            f.write(f"## Element Relationships\n\n")
            
            graph = results['document_graph']
            relationship_count = 0
            
            for elem_id, relationships in graph.edges.items():
                if relationships:
                    relationship_count += len(relationships)
            
            f.write(f"- **Total Relationships Detected:** {relationship_count}\n")
            f.write(f"- **Elements with Relationships:** {len([e for e in graph.edges if graph.edges[e]])}\n\n")
        
        logger.info(f"Extraction report saved to: {summary_path}")
        
        # Display sample of extracted content
        logger.info("\n" + "="*60)
        logger.info("SAMPLE OF EXTRACTED CONTENT:")
        logger.info("="*60)
        
        # Show first 1000 characters
        sample = results['markdown'][:1000]
        if len(results['markdown']) > 1000:
            sample += "\n\n[... Content truncated for display ...]"
        
        print(sample)
        
        return results
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise


def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced PDF extraction')
    parser.add_argument('pdf_path', help='Path to PDF file to test')
    parser.add_argument('--output-dir', default='./test_output', 
                       help='Directory to save output files (default: ./test_output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    test_enhanced_extraction(args.pdf_path, args.output_dir)


if __name__ == "__main__":
    main()
