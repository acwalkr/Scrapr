"""
Enhanced PDF Extractor using pdfplumber and pdf2image
Optimized for engineering documents with proper image extraction and context preservation
"""

import pdfplumber
import pytesseract
from PIL import Image
import io
import base64
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import json
import fitz  # PyMuPDF for image extraction only (fallback)
from pdf2image import convert_from_path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """Enhanced element with relationship tracking"""
    type: str  # text, image, table, equation, figure_reference
    content: Any
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    metadata: Dict = None
    relationships: List[str] = None  # IDs of related elements
    element_id: str = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
        if self.element_id is None:
            # Generate unique ID based on position and type
            self.element_id = f"{self.type}_{self.page_num}_{int(self.bbox[0])}_{int(self.bbox[1])}"
    
    @property
    def x0(self):
        return self.bbox[0]
    
    @property
    def y0(self):
        return self.bbox[1]
    
    @property
    def x1(self):
        return self.bbox[2]
    
    @property
    def y1(self):
        return self.bbox[3]
    
    @property
    def center_x(self):
        return (self.bbox[0] + self.bbox[2]) / 2
    
    @property
    def center_y(self):
        return (self.bbox[1] + self.bbox[3]) / 2
    
    @property
    def area(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class DocumentGraph:
    """Graph structure to maintain element relationships"""
    def __init__(self):
        self.nodes = {}  # element_id -> DocumentElement
        self.edges = defaultdict(list)  # element_id -> [related_element_ids]
    
    def add_element(self, element: DocumentElement):
        """Add element to graph"""
        self.nodes[element.element_id] = element
    
    def add_relationship(self, elem1_id: str, elem2_id: str, relationship_type: str):
        """Add relationship between elements"""
        if elem1_id in self.nodes and elem2_id in self.nodes:
            self.edges[elem1_id].append({
                "target": elem2_id,
                "type": relationship_type
            })
            # Update element relationships
            self.nodes[elem1_id].relationships.append(elem2_id)
            self.nodes[elem2_id].relationships.append(elem1_id)
    
    def find_related_elements(self, element_id: str) -> List[DocumentElement]:
        """Find all elements related to given element"""
        related = []
        if element_id in self.nodes:
            for rel_id in self.nodes[element_id].relationships:
                if rel_id in self.nodes:
                    related.append(self.nodes[rel_id])
        return related


class EnhancedContextualPDFExtractor:
    """Advanced PDF extractor with superior image extraction and context preservation"""
    
    def __init__(self,
                 ocr_dpi: int = 300,
                 image_dpi: int = 200,
                 min_text_length: int = 3,
                 x_tolerance: int = 3,
                 y_tolerance: int = 3,
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 embed_images_base64: bool = True,
                 preserve_context: bool = True,
                 use_pymupdf_fallback: bool = False):
        """
        Initialize the enhanced extractor
        
        Args:
            ocr_dpi: DPI for OCR processing
            image_dpi: DPI for image extraction
            min_text_length: Minimum text length to consider valid
            x_tolerance: Horizontal tolerance for text grouping
            y_tolerance: Vertical tolerance for text grouping
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            embed_images_base64: Whether to embed images as base64 in markdown
            preserve_context: Whether to preserve element relationships
            use_pymupdf_fallback: Use PyMuPDF for image extraction (requires license consideration)
        """
        self.ocr_dpi = ocr_dpi
        self.image_dpi = image_dpi
        self.min_text_length = min_text_length
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.embed_images_base64 = embed_images_base64
        self.preserve_context = preserve_context
        self.use_pymupdf_fallback = use_pymupdf_fallback
        self.document_graph = DocumentGraph()
        
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract content from PDF with full context preservation
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted content with preserved layout and relationships
        """
        try:
            results = {
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "path": pdf_path
                },
                "pages": [],
                "document_graph": self.document_graph,
                "markdown": "",
                "statistics": {
                    "total_pages": 0,
                    "total_text_blocks": 0,
                    "total_images": 0,
                    "total_tables": 0,
                    "ocr_pages": 0,
                    "figure_references": 0
                }
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                results["metadata"].update(pdf.metadata or {})
                results["statistics"]["total_pages"] = len(pdf.pages)
                
                # First pass: Extract all elements
                all_elements = []
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {page_num + 1}/{len(pdf.pages)}")
                    
                    page_elements = self._process_page_comprehensive(page, page_num + 1, pdf_path)
                    results["pages"].append(page_elements)
                    
                    # Add to graph
                    for elem in page_elements["elements"]:
                        self.document_graph.add_element(elem)
                        all_elements.append(elem)
                    
                    # Update statistics
                    self._update_statistics(results["statistics"], page_elements)
                
                # Second pass: Establish relationships
                if self.preserve_context:
                    self._establish_element_relationships(all_elements)
            
            # Generate structured markdown preserving layout and context
            results["markdown"] = self._generate_enhanced_markdown(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _process_page_comprehensive(self, page: pdfplumber.PDF.Page, page_num: int, pdf_path: str) -> Dict:
        """Process a single page extracting all elements comprehensively"""
        page_data = {
            "page_num": page_num,
            "width": page.width,
            "height": page.height,
            "elements": [],
            "used_ocr": False
        }
        
        # Extract text with positions
        text_elements = self._extract_text_elements_advanced(page, page_num)
        
        # Check if OCR is needed
        if self._needs_ocr(text_elements, page):
            logger.info(f"Page {page_num} needs OCR")
            ocr_elements = self._ocr_page_improved(page, page_num)
            page_data["elements"].extend(ocr_elements)
            page_data["used_ocr"] = True
        else:
            page_data["elements"].extend(text_elements)
        
        # Extract tables
        if self.extract_tables:
            table_elements = self._extract_tables_enhanced(page, page_num)
            page_data["elements"].extend(table_elements)
        
        # Extract actual embedded images (not full page captures)
        if self.extract_images:
            image_elements = self._extract_actual_images(page, page_num, pdf_path)
            page_data["elements"].extend(image_elements)
        
        # Detect figure references in text
        self._detect_figure_references(page_data["elements"])
        
        # Sort elements by position for better reading order
        page_data["elements"] = self._order_elements_advanced(page_data["elements"])
        
        return page_data
    
    def _extract_text_elements_advanced(self, page: pdfplumber.PDF.Page, page_num: int) -> List[DocumentElement]:
        """Extract text elements with enhanced position and formatting information"""
        elements = []
        
        # Get characters with position info
        chars = page.chars
        if not chars:
            return elements
        
        # Group characters into words with better handling
        words = self._group_chars_to_words_enhanced(chars)
        
        # Group words into lines with improved algorithm
        lines = self._group_words_to_lines_enhanced(words)
        
        # Group lines into blocks with better paragraph detection
        blocks = self._group_lines_to_blocks_enhanced(lines)
        
        # Create elements from blocks
        for block in blocks:
            text = " ".join([line["text"] for line in block["lines"]])
            if len(text) >= self.min_text_length:
                # Clean up text
                text = self._clean_text(text)
                
                elem = DocumentElement(
                    type="text",
                    content=text,
                    page_num=page_num,
                    bbox=(block["x0"], block["y0"], block["x1"], block["y1"]),
                    metadata={
                        "font": block.get("font"),
                        "font_size": block.get("font_size"),
                        "is_header": self._is_header_advanced(block),
                        "is_caption": self._is_caption(text),
                        "block_type": self._classify_text_block(text)
                    }
                )
                elements.append(elem)
        
        return elements
    
    def _extract_actual_images(self, page: pdfplumber.PDF.Page, page_num: int, pdf_path: str) -> List[DocumentElement]:
        """Extract actual embedded images from the PDF, not page screenshots"""
        elements = []
        
        try:
            # Method 1: Try using pdf2image to check for embedded images
            # Convert just this page to check for images
            page_images = convert_from_path(pdf_path, 
                                           first_page=page_num, 
                                           last_page=page_num,
                                           dpi=self.image_dpi)
            
            if page_images:
                page_img = page_images[0]
                
                # Check if page has images by analyzing content
                if self._page_contains_figures(page, page_img):
                    # Extract image regions
                    image_regions = self._detect_image_regions(page, page_img)
                    
                    for i, region in enumerate(image_regions):
                        # Crop the image region
                        cropped_img = page_img.crop(region)
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        cropped_img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        # Scale coordinates to PDF space
                        scale_x = page.width / page_img.width
                        scale_y = page.height / page_img.height
                        
                        elem = DocumentElement(
                            type="image",
                            content=img_base64 if self.embed_images_base64 else f"figure_{page_num}_{i}.png",
                            page_num=page_num,
                            bbox=(
                                region[0] * scale_x,
                                region[1] * scale_y,
                                region[2] * scale_x,
                                region[3] * scale_y
                            ),
                            metadata={
                                "format": "png",
                                "embedded": self.embed_images_base64,
                                "figure_index": i,
                                "original_size": (region[2] - region[0], region[3] - region[1])
                            }
                        )
                        elements.append(elem)
                        
        except Exception as e:
            logger.warning(f"Could not extract images from page {page_num}: {str(e)}")
            
            # Fallback: If PyMuPDF is allowed and available
            if self.use_pymupdf_fallback:
                elements.extend(self._extract_images_pymupdf_fallback(pdf_path, page_num))
        
        return elements
    
    def _page_contains_figures(self, page: pdfplumber.PDF.Page, page_img: Image.Image) -> bool:
        """Detect if page contains figures/images beyond just text"""
        # Check text content for figure references
        text = page.extract_text() or ""
        if re.search(r'(figure|fig\.|image|diagram|chart|graph)\s*\d*', text, re.IGNORECASE):
            return True
        
        # Analyze image content
        # Convert to grayscale numpy array
        img_array = np.array(page_img.convert('L'))
        
        # Calculate variance - higher variance indicates more complex content (images)
        variance = np.var(img_array)
        
        # If variance is high, likely contains images
        if variance > 1000:  # Threshold may need tuning
            return True
        
        return False
    
    def _detect_image_regions(self, page: pdfplumber.PDF.Page, page_img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect regions in the page that contain images"""
        regions = []
        
        # Convert to numpy array for analysis
        img_array = np.array(page_img.convert('L'))
        height, width = img_array.shape
        
        # Simple approach: Look for large rectangular regions with different characteristics
        # This is a simplified version - production would use more sophisticated computer vision
        
        # Get text regions to exclude
        text_regions = self._get_text_regions(page)
        
        # Look for regions with high variance that aren't text
        block_size = 50
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = img_array[y:y+block_size, x:x+block_size]
                
                # Check if this block has image characteristics
                if np.var(block) > 500 and not self._overlaps_text_region(x, y, x+block_size, y+block_size, text_regions, page_img.size, (page.width, page.height)):
                    # Expand region to find full image bounds
                    region = self._expand_image_region(img_array, x, y, block_size)
                    if region and (region[2] - region[0]) > 100 and (region[3] - region[1]) > 100:
                        regions.append(region)
        
        # Merge overlapping regions
        regions = self._merge_overlapping_regions(regions)
        
        return regions
    
    def _get_text_regions(self, page: pdfplumber.PDF.Page) -> List[Tuple[float, float, float, float]]:
        """Get bounding boxes of text regions"""
        regions = []
        
        # Extract words with bounding boxes
        words = page.extract_words()
        for word in words:
            regions.append((word['x0'], word['top'], word['x1'], word['bottom']))
        
        return regions
    
    def _overlaps_text_region(self, x0: int, y0: int, x1: int, y1: int, 
                             text_regions: List[Tuple[float, float, float, float]], 
                             img_size: Tuple[int, int], 
                             page_size: Tuple[float, float]) -> bool:
        """Check if image region overlaps with text"""
        # Scale image coordinates to PDF coordinates
        scale_x = page_size[0] / img_size[0]
        scale_y = page_size[1] / img_size[1]
        
        pdf_x0 = x0 * scale_x
        pdf_y0 = y0 * scale_y
        pdf_x1 = x1 * scale_x
        pdf_y1 = y1 * scale_y
        
        for text_x0, text_y0, text_x1, text_y1 in text_regions:
            # Check for overlap
            if not (pdf_x1 < text_x0 or pdf_x0 > text_x1 or pdf_y1 < text_y0 or pdf_y0 > text_y1):
                return True
        
        return False
    
    def _expand_image_region(self, img_array: np.ndarray, start_x: int, start_y: int, initial_size: int) -> Optional[Tuple[int, int, int, int]]:
        """Expand initial region to find full image bounds"""
        height, width = img_array.shape
        
        # Start with initial region
        x0, y0 = start_x, start_y
        x1, y1 = min(start_x + initial_size, width), min(start_y + initial_size, height)
        
        # Expand in all directions while variance remains high
        threshold_variance = 100
        
        # Expand left
        while x0 > 0 and np.var(img_array[y0:y1, x0-10:x0]) > threshold_variance:
            x0 -= 10
        
        # Expand right
        while x1 < width and np.var(img_array[y0:y1, x1:x1+10]) > threshold_variance:
            x1 += 10
        
        # Expand up
        while y0 > 0 and np.var(img_array[y0-10:y0, x0:x1]) > threshold_variance:
            y0 -= 10
        
        # Expand down
        while y1 < height and np.var(img_array[y1:y1+10, x0:x1]) > threshold_variance:
            y1 += 10
        
        return (x0, y0, x1, y1)
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping image regions"""
        if not regions:
            return regions
        
        merged = []
        regions = sorted(regions, key=lambda r: (r[1], r[0]))  # Sort by y, then x
        
        current = list(regions[0])
        for region in regions[1:]:
            # Check if regions overlap or are very close
            if (region[0] < current[2] + 20 and region[2] > current[0] - 20 and
                region[1] < current[3] + 20 and region[3] > current[1] - 20):
                # Merge regions
                current[0] = min(current[0], region[0])
                current[1] = min(current[1], region[1])
                current[2] = max(current[2], region[2])
                current[3] = max(current[3], region[3])
            else:
                merged.append(tuple(current))
                current = list(region)
        
        merged.append(tuple(current))
        return merged
    
    def _extract_images_pymupdf_fallback(self, pdf_path: str, page_num: int) -> List[DocumentElement]:
        """Fallback method using PyMuPDF for image extraction"""
        elements = []
        
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                # Convert to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Get image position (approximate)
                bbox = page.get_image_bbox(img[7])  # img[7] is the name
                
                elem = DocumentElement(
                    type="image",
                    content=img_base64 if self.embed_images_base64 else f"image_{page_num}_{img_index}.png",
                    page_num=page_num,
                    bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    metadata={
                        "format": "png",
                        "embedded": self.embed_images_base64,
                        "xref": xref
                    }
                )
                elements.append(elem)
                
        except Exception as e:
            logger.warning(f"PyMuPDF fallback failed: {str(e)}")
        
        return elements
    
    def _detect_figure_references(self, elements: List[DocumentElement]):
        """Detect and mark figure references in text"""
        figure_pattern = re.compile(r'(figure|fig\.?)\s*(\d+\.?\d*)', re.IGNORECASE)
        
        for elem in elements:
            if elem.type == "text":
                matches = figure_pattern.findall(elem.content)
                if matches:
                    elem.metadata["contains_figure_references"] = True
                    elem.metadata["referenced_figures"] = [f"{match[0]} {match[1]}" for match in matches]
    
    def _establish_element_relationships(self, elements: List[DocumentElement]):
        """Establish relationships between elements (text-figure, caption-image, etc.)"""
        # Group elements by page
        pages = defaultdict(list)
        for elem in elements:
            pages[elem.page_num].append(elem)
        
        # Process each page
        for page_num, page_elements in pages.items():
            # Find captions and their associated images
            self._link_captions_to_images(page_elements)
            
            # Link figure references to actual figures
            self._link_figure_references(page_elements)
            
            # Link tables to their titles
            self._link_tables_to_titles(page_elements)
    
    def _link_captions_to_images(self, elements: List[DocumentElement]):
        """Link caption text to nearby images"""
        captions = [e for e in elements if e.metadata.get("is_caption")]
        images = [e for e in elements if e.type == "image"]
        
        for caption in captions:
            # Find nearest image
            nearest_image = None
            min_distance = float('inf')
            
            for image in images:
                # Calculate distance (prefer images above caption)
                if image.y1 <= caption.y0:  # Image above caption
                    distance = caption.y0 - image.y1
                else:  # Image below caption
                    distance = (image.y0 - caption.y1) * 2  # Penalize images below
                
                if distance < min_distance and distance < 50:  # Within 50 units
                    min_distance = distance
                    nearest_image = image
            
            if nearest_image:
                self.document_graph.add_relationship(
                    caption.element_id,
                    nearest_image.element_id,
                    "caption_for"
                )
    
    def _link_figure_references(self, elements: List[DocumentElement]):
        """Link figure references in text to actual figures"""
        text_with_refs = [e for e in elements if e.metadata.get("contains_figure_references")]
        images = [e for e in elements if e.type == "image"]
        
        for text_elem in text_with_refs:
            for ref in text_elem.metadata.get("referenced_figures", []):
                # Extract figure number
                match = re.search(r'(\d+\.?\d*)', ref)
                if match:
                    fig_num = match.group(1)
                    
                    # Find matching figure (usually by caption)
                    for elem in elements:
                        if elem.metadata.get("is_caption") and fig_num in elem.content:
                            # Find image associated with this caption
                            related = self.document_graph.find_related_elements(elem.element_id)
                            for rel in related:
                                if rel.type == "image":
                                    self.document_graph.add_relationship(
                                        text_elem.element_id,
                                        rel.element_id,
                                        "references"
                                    )
    
    def _link_tables_to_titles(self, elements: List[DocumentElement]):
        """Link tables to their titles"""
        tables = [e for e in elements if e.type == "table"]
        texts = [e for e in elements if e.type == "text"]
        
        for table in tables:
            # Look for table title above the table
            best_title = None
            min_distance = float('inf')
            
            for text in texts:
                if text.y1 <= table.y0 and "table" in text.content.lower():
                    distance = table.y0 - text.y1
                    if distance < min_distance and distance < 30:
                        min_distance = distance
                        best_title = text
            
            if best_title:
                self.document_graph.add_relationship(
                    table.element_id,
                    best_title.element_id,
                    "has_title"
                )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR errors
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        # Remove isolated special characters
        text = re.sub(r'\s+([^\w\s])\s+', r' \1 ', text)
        
        return text
    
    def _is_caption(self, text: str) -> bool:
        """Detect if text is likely a figure/table caption"""
        caption_patterns = [
            r'^(figure|fig\.?)\s*\d+',
            r'^(table)\s*\d+',
            r'^(chart|graph|diagram)\s*\d+',
            r'^(equation|eq\.?)\s*\d+'
        ]
        
        text_lower = text.lower().strip()
        for pattern in caption_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _classify_text_block(self, text: str) -> str:
        """Classify the type of text block"""
        text_lower = text.lower().strip()
        
        # Check for specific patterns
        if self._is_caption(text):
            return "caption"
        elif re.match(r'^\d+\.?\d*\s+\w+', text) and len(text.split()) < 10:
            return "heading"
        elif re.match(r'^[a-z]\.\s+', text_lower) or re.match(r'^\d+\)\s+', text):
            return "list_item"
        elif len(text) < 50 and text.endswith(':'):
            return "label"
        else:
            return "paragraph"
    
    def _is_header_advanced(self, block: Dict) -> bool:
        """Advanced header detection"""
        text = " ".join([line["text"] for line in block["lines"]])
        
        # Font size check
        if block.get("font_size", 0) > 14:
            return True
        
        # All caps check
        if text.isupper() and len(text.split()) < 10:
            return True
        
        # Section numbering
        if re.match(r'^(\d+\.?\d*)\s+[A-Z]', text):
            return True
        
        # Bold font check (if font name contains "Bold")
        if block.get("font", "").lower().find("bold") != -1:
            return True
        
        return False
    
    def _extract_tables_enhanced(self, page: pdfplumber.PDF.Page, page_num: int) -> List[DocumentElement]:
        """Enhanced table extraction with better structure preservation"""
        elements = []
        
        # Configure table extraction settings
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "text_tolerance": 3,
        }
        
        # Extract tables with custom settings
        tables = page.extract_tables(table_settings)
        
        for i, table in enumerate(tables):
            if table and len(table) > 0:
                # Clean up table data
                cleaned_table = []
                for row in table:
                    cleaned_row = []
                    for cell in row:
                        if cell is not None:
                            # Clean cell content
                            cell_text = str(cell).strip()
                            cell_text = re.sub(r'\s+', ' ', cell_text)
                            cleaned_row.append(cell_text)
                        else:
                            cleaned_row.append("")
                    cleaned_table.append(cleaned_row)
                
                # Try to find table boundaries more accurately
                table_bbox = self._find_table_bbox(page, cleaned_table)
                
                elem = DocumentElement(
                    type="table",
                    content=cleaned_table,
                    page_num=page_num,
                    bbox=table_bbox,
                    metadata={
                        "rows": len(cleaned_table),
                        "cols": len(cleaned_table[0]) if cleaned_table else 0,
                        "has_header": self._table_has_header(cleaned_table)
                    }
                )
                elements.append(elem)
        
        return elements
    
    def _find_table_bbox(self, page: pdfplumber.PDF.Page, table: List[List]) -> Tuple[float, float, float, float]:
        """Find more accurate bounding box for table"""
        # Get all words on page
        words = page.extract_words()
        
        # Find words that match table content
        table_words = []
        for row in table:
            for cell in row:
                if cell:
                    # Find words matching this cell content
                    for word in words:
                        if word['text'] in cell:
                            table_words.append(word)
        
        if table_words:
            x0 = min(w['x0'] for w in table_words)
            y0 = min(w['top'] for w in table_words)
            x1 = max(w['x1'] for w in table_words)
            y1 = max(w['bottom'] for w in table_words)
            return (x0, y0, x1, y1)
        else:
            # Fallback to page margins
            margin = 50
            return (margin, margin, page.width - margin, page.height - margin)
    
    def _table_has_header(self, table: List[List]) -> bool:
        """Detect if first row is a header"""
        if not table or len(table) < 2:
            return False
        
        first_row = table[0]
        
        # Check if first row has different characteristics
        # Usually headers are all text, while data rows have numbers
        first_row_numeric = sum(1 for cell in first_row if re.search(r'\d', str(cell))) / len(first_row)
        
        if len(table) > 1:
            second_row_numeric = sum(1 for cell in table[1] if re.search(r'\d', str(cell))) / len(table[1])
            
            # If first row has significantly fewer numbers, it's likely a header
            if first_row_numeric < 0.3 and second_row_numeric > 0.5:
                return True
        
        return False
    
    def _group_chars_to_words_enhanced(self, chars: List[Dict]) -> List[Dict]:
        """Enhanced word grouping with better space detection"""
        if not chars:
            return []
        
        words = []
        current_word = {
            "chars": [chars[0]],
            "x0": chars[0]["x0"],
            "y0": chars[0]["top"],
            "x1": chars[0]["x1"],
            "y1": chars[0]["bottom"],
            "font": chars[0].get("fontname", ""),
            "size": chars[0].get("size", 0)
        }
        
        for i in range(1, len(chars)):
            char = chars[i]
            prev_char = chars[i-1]
            
            # Calculate space threshold based on font size
            space_threshold = max(self.x_tolerance, prev_char.get("size", 10) * 0.3)
            
            # Check if characters are in same word
            same_line = abs(char["top"] - prev_char["top"]) < self.y_tolerance
            close_enough = abs(char["x0"] - prev_char["x1"]) < space_threshold
            same_font = char.get("fontname") == prev_char.get("fontname")
            
            if same_line and close_enough and same_font:
                # Add to current word
                current_word["chars"].append(char)
                current_word["x1"] = max(current_word["x1"], char["x1"])
                current_word["y1"] = max(current_word["y1"], char["bottom"])
            else:
                # Complete current word
                current_word["text"] = "".join([c.get("text", "") for c in current_word["chars"]])
                if current_word["text"].strip():
                    words.append(current_word)
                
                # Start new word
                current_word = {
                    "chars": [char],
                    "x0": char["x0"],
                    "y0": char["top"],
                    "x1": char["x1"],
                    "y1": char["bottom"],
                    "font": char.get("fontname", ""),
                    "size": char.get("size", 0)
                }
        
        # Add last word
        if current_word["chars"]:
            current_word["text"] = "".join([c.get("text", "") for c in current_word["chars"]])
            if current_word["text"].strip():
                words.append(current_word)
        
        return words
    
    def _group_words_to_lines_enhanced(self, words: List[Dict]) -> List[Dict]:
        """Enhanced line grouping with better alignment detection"""
        if not words:
            return []
        
        # Sort words by vertical position, then horizontal
        words.sort(key=lambda w: (w["y0"], w["x0"]))
        
        lines = []
        current_line = {
            "words": [words[0]],
            "x0": words[0]["x0"],
            "y0": words[0]["y0"],
            "x1": words[0]["x1"],
            "y1": words[0]["y1"],
            "baseline": words[0]["y1"]
        }
        
        for word in words[1:]:
            # Check if word is on same line
            baseline_diff = abs(word["y1"] - current_line["baseline"])
            
            if baseline_diff < self.y_tolerance:
                # Add to current line
                current_line["words"].append(word)
                current_line["x1"] = max(current_line["x1"], word["x1"])
                current_line["y0"] = min(current_line["y0"], word["y0"])
                current_line["y1"] = max(current_line["y1"], word["y1"])
                # Update baseline as average
                current_line["baseline"] = sum(w["y1"] for w in current_line["words"]) / len(current_line["words"])
            else:
                # Sort words in line by x position
                current_line["words"].sort(key=lambda w: w["x0"])
                current_line["text"] = " ".join([w["text"] for w in current_line["words"]])
                lines.append(current_line)
                
                # Start new line
                current_line = {
                    "words": [word],
                    "x0": word["x0"],
                    "y0": word["y0"],
                    "x1": word["x1"],
                    "y1": word["y1"],
                    "baseline": word["y1"]
                }
        
        # Add last line
        if current_line["words"]:
            current_line["words"].sort(key=lambda w: w["x0"])
            current_line["text"] = " ".join([w["text"] for w in current_line["words"]])
            lines.append(current_line)
        
        return lines
    
    def _group_lines_to_blocks_enhanced(self, lines: List[Dict]) -> List[Dict]:
        """Enhanced block grouping with better paragraph detection"""
        if not lines:
            return []
        
        blocks = []
        current_block = {
            "lines": [lines[0]],
            "x0": lines[0]["x0"],
            "y0": lines[0]["y0"],
            "x1": lines[0]["x1"],
            "y1": lines[0]["y1"],
            "font": lines[0]["words"][0].get("font", "") if lines[0]["words"] else "",
            "font_size": lines[0]["words"][0].get("size", 0) if lines[0]["words"] else 0
        }
        
        for i in range(1, len(lines)):
            line = lines[i]
            prev_line = lines[i-1]
            
            # Calculate metrics for grouping decision
            vertical_gap = line["y0"] - prev_line["y1"]
            indent_diff = abs(line["x0"] - current_block["x0"])
            same_font = (line["words"][0].get("font") == current_block["font"]) if line["words"] else True
            
            # Determine if lines should be in same block
            # Allow larger gaps for same paragraph
            max_gap = current_block["font_size"] * 1.5 if current_block["font_size"] else 20
            
            # Check for paragraph continuation
            is_continuation = (
                vertical_gap < max_gap and
                indent_diff < 10 and
                same_font
            )
            
            if is_continuation:
                # Add to current block
                current_block["lines"].append(line)
                current_block["x0"] = min(current_block["x0"], line["x0"])
                current_block["x1"] = max(current_block["x1"], line["x1"])
                current_block["y1"] = line["y1"]
            else:
                # Complete current block
                blocks.append(current_block)
                
                # Start new block
                current_block = {
                    "lines": [line],
                    "x0": line["x0"],
                    "y0": line["y0"],
                    "x1": line["x1"],
                    "y1": line["y1"],
                    "font": line["words"][0].get("font", "") if line["words"] else "",
                    "font_size": line["words"][0].get("size", 0) if line["words"] else 0
                }
        
        # Add last block
        if current_block["lines"]:
            blocks.append(current_block)
        
        return blocks
    
    def _ocr_page_improved(self, page: pdfplumber.PDF.Page, page_num: int) -> List[DocumentElement]:
        """Improved OCR with better text block detection"""
        elements = []
        
        try:
            # Convert page to image
            page_image = page.to_image(resolution=self.ocr_dpi)
            
            # Convert to PIL Image
            img_buffer = io.BytesIO()
            page_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img = Image.open(img_buffer)
            
            # Preprocess image for better OCR
            img = self._preprocess_for_ocr(img)
            
            # Perform OCR with custom config
            custom_config = r'--oem 3 --psm 6'
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)
            
            # Group OCR results into meaningful blocks
            blocks = self._group_ocr_results(ocr_data, page.width / img.width)
            
            for block in blocks:
                elem = DocumentElement(
                    type="text",
                    content=block["text"],
                    page_num=page_num,
                    bbox=(block["x0"], block["y0"], block["x1"], block["y1"]),
                    metadata={
                        "source": "ocr",
                        "confidence": block["confidence"]
                    }
                )
                elements.append(elem)
                
        except Exception as e:
            logger.error(f"Improved OCR failed for page {page_num}: {str(e)}")
        
        return elements
    
    def _preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply slight sharpening
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        return img
    
    def _group_ocr_results(self, ocr_data: Dict, scale: float) -> List[Dict]:
        """Group OCR results into coherent text blocks"""
        blocks = []
        current_block = None
        
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                text = ocr_data['text'][i].strip()
                if text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Check if this belongs to current block
                    if current_block is None:
                        current_block = {
                            'text': text,
                            'x0': x * scale,
                            'y0': y * scale,
                            'x1': (x + w) * scale,
                            'y1': (y + h) * scale,
                            'confidence': ocr_data['conf'][i]
                        }
                    else:
                        # Check proximity
                        gap = y - (current_block['y1'] / scale)
                        
                        if gap < h * 0.5:  # Same block
                            current_block['text'] += ' ' + text
                            current_block['x1'] = max(current_block['x1'], (x + w) * scale)
                            current_block['y1'] = (y + h) * scale
                            current_block['confidence'] = min(current_block['confidence'], ocr_data['conf'][i])
                        else:
                            # New block
                            if len(current_block['text']) >= self.min_text_length:
                                blocks.append(current_block)
                            
                            current_block = {
                                'text': text,
                                'x0': x * scale,
                                'y0': y * scale,
                                'x1': (x + w) * scale,
                                'y1': (y + h) * scale,
                                'confidence': ocr_data['conf'][i]
                            }
        
        # Add last block
        if current_block and len(current_block['text']) >= self.min_text_length:
            blocks.append(current_block)
        
        return blocks
    
    def _order_elements_advanced(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Advanced element ordering considering complex layouts"""
        if not elements:
            return elements
        
        # Detect layout type
        layout_type = self._detect_layout_type(elements)
        
        if layout_type == "multi_column":
            return self._order_multi_column(elements)
        else:
            # Simple top-to-bottom, left-to-right ordering
            return sorted(elements, key=lambda e: (e.y0, e.x0))
    
    def _detect_layout_type(self, elements: List[DocumentElement]) -> str:
        """Detect if page has single or multi-column layout"""
        if not elements:
            return "single_column"
        
        # Get x-positions of text elements
        text_elements = [e for e in elements if e.type == "text"]
        if not text_elements:
            return "single_column"
        
        x_positions = [e.center_x for e in text_elements]
        
        # Simple clustering to detect columns
        if len(set(x_positions)) > len(x_positions) * 0.8:
            # X positions are well distributed
            return "single_column"
        
        # Check for distinct x-position clusters
        x_positions.sort()
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap
                gaps.append(gap)
        
        if len(gaps) >= 1:
            return "multi_column"
        
        return "single_column"
    
    def _order_multi_column(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Order elements for multi-column layout"""
        # Detect columns
        columns = self._detect_columns_advanced(elements)
        
        ordered = []
        for column in columns:
            # Sort elements in column by vertical position
            column.sort(key=lambda e: e.y0)
            ordered.extend(column)
        
        return ordered
    
    def _detect_columns_advanced(self, elements: List[DocumentElement]) -> List[List[DocumentElement]]:
        """Advanced column detection"""
        if not elements:
            return []
        
        # Use text elements for column detection
        text_elements = [e for e in elements if e.type == "text"]
        
        if not text_elements:
            return [elements]
        
        # Find column boundaries using clustering
        x_centers = sorted([(e.center_x, e) for e in text_elements])
        
        columns = []
        current_column = [x_centers[0][1]]
        current_x = x_centers[0][0]
        
        for x, elem in x_centers[1:]:
            if x - current_x > 100:  # New column
                columns.append(current_column)
                current_column = [elem]
                current_x = x
            else:
                current_column.append(elem)
                current_x = (current_x + x) / 2  # Running average
        
        if current_column:
            columns.append(current_column)
        
        # Add non-text elements to appropriate columns
        other_elements = [e for e in elements if e.type != "text"]
        for elem in other_elements:
            # Find best column
            best_column = 0
            min_distance = float('inf')
            
            for i, column in enumerate(columns):
                # Calculate average x position of column
                avg_x = sum(e.center_x for e in column) / len(column)
                distance = abs(elem.center_x - avg_x)
                
                if distance < min_distance:
                    min_distance = distance
                    best_column = i
            
            columns[best_column].append(elem)
        
        return columns
    
    def _needs_ocr(self, text_elements: List[DocumentElement], page: pdfplumber.PDF.Page) -> bool:
        """Improved OCR detection"""
        if not text_elements:
            return True
        
        # Calculate total text content
        total_text = " ".join([elem.content for elem in text_elements])
        
        # Check for quality indicators
        if len(total_text) < 50:
            return True
        
        # Check for garbled text patterns
        garbled_patterns = [
            r'\(cid:\d+\)',  # CID references
            r'[^\x00-\x7F]{3,}',  # Non-ASCII sequences
            r'[\x00-\x1F\x7F-\x9F]',  # Control characters
        ]
        
        for pattern in garbled_patterns:
            if re.search(pattern, total_text):
                return True
        
        # Check character distribution
        alnum_count = sum(1 for c in total_text if c.isalnum())
        total_chars = len(total_text)
        
        if total_chars > 0 and alnum_count / total_chars < 0.5:
            return True
        
        return False
    
    def _generate_enhanced_markdown(self, results: Dict) -> str:
        """Generate markdown with preserved relationships and context"""
        md_lines = []
        
        # Add metadata header
        md_lines.append(f"# {results['metadata'].get('title', results['metadata']['filename'])}")
        md_lines.append(f"\n**Document:** {results['metadata']['filename']}")
        md_lines.append(f"**Total Pages:** {results['statistics']['total_pages']}")
        
        if results['statistics']['total_images'] > 0:
            md_lines.append(f"**Images Found:** {results['statistics']['total_images']}")
        
        if results['statistics']['ocr_pages'] > 0:
            md_lines.append(f"**Pages Requiring OCR:** {results['statistics']['ocr_pages']}")
        
        md_lines.append("\n---\n")
        
        # Process each page
        for page_data in results['pages']:
            md_lines.append(f"## Page {page_data['page_num']}")
            
            if page_data.get('used_ocr'):
                md_lines.append("*Note: This page was processed using OCR due to text extraction issues*\n")
            
            # Group elements by type and relationships
            elements_by_type = defaultdict(list)
            for elem in page_data['elements']:
                elements_by_type[elem.type].append(elem)
            
            # Process elements in logical order
            processed_ids = set()
            
            for elem in page_data['elements']:
                if elem.element_id in processed_ids:
                    continue
                
                # Process element and its related elements together
                if elem.type == "text":
                    # Check if it's a header
                    if elem.metadata.get('is_header'):
                        level = "###" if elem.metadata.get('font_size', 0) > 16 else "####"
                        md_lines.append(f"\n{level} {elem.content}\n")
                    
                    # Check if it references figures
                    elif elem.metadata.get('contains_figure_references'):
                        md_lines.append(f"{elem.content}")
                        
                        # Find and include referenced figures
                        related = self.document_graph.find_related_elements(elem.element_id)
                        for rel in related:
                            if rel.type == "image" and rel.element_id not in processed_ids:
                                md_lines.append(self._format_image_element(rel))
                                processed_ids.add(rel.element_id)
                    
                    # Regular paragraph
                    else:
                        md_lines.append(f"{elem.content}")
                
                elif elem.type == "image" and elem.element_id not in processed_ids:
                    # Check for associated caption
                    caption = None
                    related = self.document_graph.find_related_elements(elem.element_id)
                    for rel in related:
                        if rel.metadata.get('is_caption'):
                            caption = rel.content
                            processed_ids.add(rel.element_id)
                            break
                    
                    md_lines.append(self._format_image_element(elem, caption))
                
                elif elem.type == "table":
                    # Check for title
                    title = None
                    related = self.document_graph.find_related_elements(elem.element_id)
                    for rel in related:
                        if "table" in rel.content.lower():
                            title = rel.content
                            processed_ids.add(rel.element_id)
                            break
                    
                    if title:
                        md_lines.append(f"\n**{title}**")
                    
                    md_lines.append(self._table_to_enhanced_markdown(elem.content))
                
                processed_ids.add(elem.element_id)
            
            md_lines.append("\n---\n")
        
        # Add statistics summary
        stats = results['statistics']
        md_lines.append("\n## Extraction Summary\n")
        md_lines.append(f"- **Text Blocks Extracted:** {stats['total_text_blocks']}")
        md_lines.append(f"- **Images Extracted:** {stats['total_images']}")
        md_lines.append(f"- **Tables Extracted:** {stats['total_tables']}")
        
        if stats.get('figure_references'):
            md_lines.append(f"- **Figure References Found:** {stats['figure_references']}")
        
        return "\n".join(md_lines)
    
    def _format_image_element(self, elem: DocumentElement, caption: str = None) -> str:
        """Format image element for markdown"""
        lines = []
        
        if self.embed_images_base64 and elem.metadata.get('embedded'):
            lines.append(f"\n![{caption or f'Image from page {elem.page_num}'}](data:image/png;base64,{elem.content})")
        else:
            lines.append(f"\n![{caption or f'Image from page {elem.page_num}'}]({elem.content})")
        
        if caption:
            lines.append(f"*{caption}*")
        
        # Add position info as comment
        lines.append(f"<!-- Position: ({elem.x0:.0f}, {elem.y0:.0f}) Size: {elem.x1-elem.x0:.0f}x{elem.y1-elem.y0:.0f} -->")
        
        return "\n".join(lines)
    
    def _table_to_enhanced_markdown(self, table: List[List]) -> str:
        """Convert table to enhanced markdown with better formatting"""
        if not table or not table[0]:
            return ""
        
        md_lines = ["\n"]
        
        # Determine column widths
        col_widths = []
        for col_idx in range(len(table[0])):
            max_width = max(len(str(row[col_idx]) if col_idx < len(row) else "") for row in table)
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        # Format header
        header = table[0]
        header_cells = []
        for i, cell in enumerate(header):
            cell_str = str(cell) if cell else ""
            header_cells.append(cell_str.ljust(col_widths[i]))
        
        md_lines.append("| " + " | ".join(header_cells) + " |")
        
        # Separator
        separators = ["-" * width for width in col_widths]
        md_lines.append("| " + " | ".join(separators) + " |")
        
        # Data rows
        for row in table[1:]:
            row_cells = []
            for i in range(len(col_widths)):
                if i < len(row):
                    cell_str = str(row[i]) if row[i] else ""
                    row_cells.append(cell_str.ljust(col_widths[i]))
                else:
                    row_cells.append(" " * col_widths[i])
            
            md_lines.append("| " + " | ".join(row_cells) + " |")
        
        md_lines.append("")
        return "\n".join(md_lines)
    
    def _update_statistics(self, stats: Dict, page_data: Dict):
        """Update extraction statistics"""
        for elem in page_data['elements']:
            if elem.type == 'text':
                stats['total_text_blocks'] += 1
                if elem.metadata.get('contains_figure_references'):
                    stats['figure_references'] = stats.get('figure_references', 0) + 1
            elif elem.type == 'image':
                stats['total_images'] += 1
            elif elem.type == 'table':
                stats['total_tables'] += 1
        
        if page_data.get('used_ocr'):
            stats['ocr_pages'] += 1


# Example usage function
def extract_engineering_pdf(pdf_path: str, output_path: str = None):
    """
    Extract content from engineering PDF with all enhancements
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save markdown output
    """
    extractor = EnhancedContextualPDFExtractor(
        extract_images=True,
        embed_images_base64=True,
        preserve_context=True
    )
    
    results = extractor.extract_from_pdf(pdf_path)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['markdown'])
    
    return results
