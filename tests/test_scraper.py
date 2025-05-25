import unittest
import os
import sys

# Add the parent directory to the sys.path to allow importing scraper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scraper import get_file_type, extract_text_from_txt, extract_text_from_pdf, extract_text_from_docx

class TestScraper(unittest.TestCase):

    def setUp(self):
        # Create a dummy txt file for testing
        self.test_txt_file = os.path.join(os.path.dirname(__file__), "sample.txt")
        with open(self.test_txt_file, "w") as f:
            f.write("This is a test text file.")
        
        # Create dummy pdf and docx files for testing placeholder functions
        # These files don't need actual content, just need to exist for path validation
        # if the placeholder functions expect valid paths.
        self.test_pdf_file = os.path.join(os.path.dirname(__file__), "dummy.pdf")
        self.test_docx_file = os.path.join(os.path.dirname(__file__), "dummy.docx")
        with open(self.test_pdf_file, "w") as f:
            f.write("") # Empty file is fine for placeholder testing
        with open(self.test_docx_file, "w") as f:
            f.write("") # Empty file is fine for placeholder testing


    def tearDown(self):
        # Remove the dummy files
        if os.path.exists(self.test_txt_file):
            os.remove(self.test_txt_file)
        if os.path.exists(self.test_pdf_file):
            os.remove(self.test_pdf_file)
        if os.path.exists(self.test_docx_file):
            os.remove(self.test_docx_file)

    def test_get_file_type(self):
        self.assertEqual(get_file_type("document.pdf"), "pdf")
        self.assertEqual(get_file_type("document.docx"), "docx")
        self.assertEqual(get_file_type("document.txt"), "txt")
        self.assertEqual(get_file_type("DOCUMENT.PDF"), "pdf") # Test case-insensitivity
        self.assertEqual(get_file_type("folder/document.docx"), "docx") # Test with path
        with self.assertRaises(ValueError):
            get_file_type("document.xyz")
        with self.assertRaises(ValueError):
            get_file_type("document") # Test no extension
        with self.assertRaises(ValueError):
            get_file_type("archive.tar.gz") # Test double extension

    def test_extract_text_from_txt(self):
        text = extract_text_from_txt(self.test_txt_file)
        self.assertEqual(text, "This is a test text file.")
        # Test non-existent file
        self.assertEqual(extract_text_from_txt("non_existent_file.txt"), "")

    def test_extract_text_from_pdf_placeholder(self):
        # The placeholder returns f"Text extracted from PDF: {file_path}"
        expected_text = f"Text extracted from PDF: {self.test_pdf_file}"
        self.assertEqual(extract_text_from_pdf(self.test_pdf_file), expected_text)
        # Test non-existent file
        self.assertEqual(extract_text_from_pdf("non_existent_file.pdf"), "")

    def test_extract_text_from_docx_placeholder(self):
        # The placeholder returns f"Text extracted from DOCX: {file_path}"
        expected_text = f"Text extracted from DOCX: {self.test_docx_file}"
        self.assertEqual(extract_text_from_docx(self.test_docx_file), expected_text)
        # Test non-existent file
        self.assertEqual(extract_text_from_docx("non_existent_file.docx"), "")

if __name__ == '__main__':
    unittest.main()
