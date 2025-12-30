from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from PyPDF2 import PdfReader
from docx import Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure API configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_API_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return None
    
# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        print(text)
        return text
    except Exception as e:
        return None, f"DOCX Error: {str(e)}"

# Function to extract text from txt
def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        return None, f"PDF Error: {str(e)}"

# Function to call Azure Document Intelligence OCR
def call_document_intelligence(file):
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_API_KEY),
    )

    poller = document_intelligence_client.begin_analyze_document(
        model_id="prebuilt-read", body=file
    )
    result = poller.result()
    return result.content if result else None