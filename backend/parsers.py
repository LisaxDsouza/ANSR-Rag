import PyPDF2
import docx
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(file_path):
    """Extracts text from PDF page by page with error handling."""
    if os.path.getsize(file_path) == 0:
        logger.warning(f"Empty PDF file: {file_path}")
        return []
        
    text_content = []
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                logger.error(f"PDF is password protected: {file_path}")
                return []
                
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_content.append({
                        "text": text,
                        "metadata": {"location": f"Page {i+1}"}
                    })
            
            if not text_content:
                logger.warning(f"No extractable text found in PDF: {file_path} (Possible scanned image)")
                
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        return []
        
    return text_content

def parse_docx(file_path):
    """Extracts text from Word documents with error handling."""
    if os.path.getsize(file_path) == 0:
        logger.warning(f"Empty DOCX file: {file_path}")
        return []

    try:
        doc = docx.Document(file_path)
        text_content = []
        current_section = "General"
        
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                current_section = para.text
            
            if para.text.strip():
                text_content.append({
                    "text": para.text,
                    "metadata": {"location": f"Section: {current_section}"}
                })
                
        if not text_content:
            logger.warning(f"No text content found in DOCX: {file_path}")
            
    except Exception as e:
        logger.error(f"Error parsing DOCX {file_path}: {e}")
        return []
        
    return text_content

def parse_excel(file_path):
    """Extracts text from Excel with support for multiple sheets and error handling."""
    if os.path.getsize(file_path) == 0:
        logger.warning(f"Empty Excel file: {file_path}")
        return []

    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        text_content = []
        
        for sheet_name, df in all_sheets.items():
            if df.empty:
                logger.info(f"Empty sheet found: {sheet_name}")
                continue
                
            # Handle formula cells and NaNs
            df = df.fillna("")
            
            for index, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                if row_text.strip():
                    text_content.append({
                        "text": row_text,
                        "metadata": {"location": f"Tab: {sheet_name}, Row: {index + 1}"}
                    })
        
        if not text_content:
            logger.warning(f"No valid rows found in Excel: {file_path}")
            
    except Exception as e:
        logger.error(f"Error parsing Excel {file_path}: {e}")
        return []
        
    return text_content

def parse_web(url):
    """Extracts clean text from a URL with error handling for redirects and errors."""
    try:
        # User-agent to handle some basic blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        
        if response.status_code != 200:
            logger.error(f"Web request failed with status {response.status_code}: {url}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check if login might be required
        if "login" in response.url.lower() and "login" not in url.lower():
            logger.warning(f"Redirected to login page: {response.url}")
            return []

        for script_or_style in soup(["script", "style", "nav", "footer"]):
            script_or_style.decompose()
            
        text_content = []
        current_section = "General"
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            if element.name.startswith('h'):
                current_section = element.get_text().strip()
            
            content = element.get_text().strip()
            if content:
                text_content.append({
                    "text": content,
                    "metadata": {"location": f"Section: {current_section}"}
                })
        return text_content
        
    except Exception as e:
        logger.error(f"Error parsing web resource {url}: {e}")
        return []

def get_parser(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return parse_pdf
    elif ext == '.docx':
        return parse_docx
    elif ext in ['.xlsx', '.xls']:
        return parse_excel
    return None
