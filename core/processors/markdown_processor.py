"""
Markdown processor for converting extracted content to formatted Markdown
"""

class MarkdownProcessor:
    """Process extracted content into well-formatted Markdown"""
    
    def process(self, extraction_result):
        """
        Convert extraction result to Markdown format
        
        Args:
            extraction_result (dict): Result from extractors
            
        Returns:
            str: Formatted Markdown content
        """
        markdown_parts = []
        
        # Add metadata if available
        if extraction_result.get('metadata'):
            markdown_parts.append(self._format_metadata(extraction_result['metadata']))
        
        # Add main content
        content = extraction_result.get('content', '')
        if content:
            # Clean and format the content
            content = self._clean_content(content)
            markdown_parts.append(content)
        
        # Add tables
        if extraction_result.get('tables'):
            markdown_parts.append("\n\n## Tables\n")
            for i, table in enumerate(extraction_result['tables']):
                markdown_parts.append(f"\n### Table {i+1}\n")
                markdown_parts.append(self._format_table(table))
        
        # Add image references
        if extraction_result.get('images'):
            markdown_parts.append("\n\n## Figures and Diagrams\n")
            for i, image_info in enumerate(extraction_result['images']):
                markdown_parts.append(f"\n- Figure {i+1}: [Image on page {image_info.get('page', 'unknown')}]")
        
        # Add extraction notes
        if extraction_result.get('ocr_used'):
            markdown_parts.append(f"\n\n---\n\n*Note: This document was processed using OCR with confidence: {extraction_result.get('ocr_confidence', 'N/A')}*")
        
        return '\n'.join(markdown_parts)
    
    def _format_metadata(self, metadata):
        """Format metadata as Markdown"""
        lines = ["## Document Metadata\n"]
        for key, value in metadata.items():
            lines.append(f"- **{key.title()}**: {value}")
        lines.append("")
        return '\n'.join(lines)
    
    def _clean_content(self, content):
        """Clean and format content"""
        # Remove excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Detect headers (lines that are all caps or numbered sections)
                if line.isupper() and len(line) < 100:
                    line = f"\n## {line.title()}\n"
                elif self._is_section_header(line):
                    line = f"\n### {line}\n"
                
                cleaned_lines.append(line)
        
        # Join with appropriate spacing
        result = '\n\n'.join(cleaned_lines)
        
        # Fix common OCR issues
        result = result.replace(' .', '.')
        result = result.replace(' ,', ',')
        result = result.replace('  ', ' ')
        
        return result
    
    def _is_section_header(self, line):
        """Check if line is likely a section header"""
        # Common patterns for engineering document headers
        import re
        patterns = [
            r'^\d+\.\d+',  # 1.1, 2.3, etc.
            r'^[A-Z]\.\d+',  # A.1, B.2, etc.
            r'^Chapter \d+',
            r'^Section \d+',
            r'^Appendix [A-Z]'
        ]
        
        for pattern in patterns:
            if re.match(pattern, line):
                return True
        return False    
    def _format_table(self, table_data):
        """Format table data as Markdown table"""
        if not table_data or not isinstance(table_data, list):
            return "*[Table data not available]*"
        
        if len(table_data) == 0:
            return "*[Empty table]*"
        
        # Get the maximum number of columns
        max_cols = max(len(row) if isinstance(row, list) else 1 for row in table_data)
        
        markdown_lines = []
        
        # Process each row
        for i, row in enumerate(table_data):
            if isinstance(row, list):
                # Ensure all rows have same number of columns
                while len(row) < max_cols:
                    row.append("")
                
                # Clean cell contents
                cleaned_row = [str(cell).strip() for cell in row]
                markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
                
                # Add separator after first row (header)
                if i == 0:
                    separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
                    markdown_lines.append(separator)
        
        return '\n'.join(markdown_lines) if markdown_lines else "*[Table formatting error]*"