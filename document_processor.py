import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def load_multiple_pdfs(self, pdf_dir):
        documents = []
        for filename in os.listdir(pdf_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                text = self.load_pdf(pdf_path)
                documents.append({"text": text, "source": filename})
        return documents
    
    def chunk_documents(self, documents):
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_text(doc["text"])
            for chunk in doc_chunks:
                chunks.append({
                    "content": chunk,
                    "source": doc["source"]
                })
        return chunks
