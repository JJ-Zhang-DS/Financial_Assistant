import os
import glob
from typing import List, Optional
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from dotenv import load_dotenv
import re

load_dotenv()

class SECIndexer:
    def __init__(self, persist_directory: str = "vectorstore/chroma_db"):
        """Initialize the SEC filings indexer."""
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # or 'cuda' for GPU
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Primary separators that respect document structure
            separators=[
                # Common section headers in SEC filings
                "\n\nITEM ",
                "\n\nPART ",
                # Subsection markers
                "\n\n\d+\.\s+",  # Numbered sections like "1. Business"
                "\n\n[A-Z][A-Z\s]+\n\n",  # ALL CAPS HEADERS
                # Paragraph breaks
                "\n\n",
                "\n",
                " ",
                ""
            ],
            # Optimized for all-MiniLM-L6-v2's context window
            chunk_size=512,
            chunk_overlap=50,
            # Don't split in the middle of sentences if possible
            length_function=len,
            is_separator_regex=True
        )
        
        # Create directories if they don't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs("data/sec_filings", exist_ok=True)
    
    def load_documents(self, file_path: str):
        """Load documents from a file based on its extension."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in ['.html', '.htm']:
            loader = UnstructuredHTMLLoader(file_path)
        elif file_ext in ['.txt', '.md']:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return loader.load()
    
    def preprocess_sec_filing(self, text):
        """Add explicit section markers to improve chunking."""
        # Enhance section headers to make them more detectable
        text = re.sub(r'(\n\s*)(ITEM\s+\d+[\.:]\s*)(.*?)(\n)', 
                     r'\1### SECTION: \2\3 ###\4', 
                     text, flags=re.IGNORECASE)
        
        # Mark subsections
        text = re.sub(r'(\n\s*)(\([a-z]\)\s+)(.*?)(\n)', 
                     r'\1#### SUBSECTION: \2\3 ####\4', 
                     text, flags=re.IGNORECASE)
        
        return text
    
    def extract_section_info(self, text):
        """Extract section information from chunk text."""
        section_match = re.search(r'### SECTION: (ITEM\s+\d+[\.:]\s*)(.*?) ###', text)
        if section_match:
            return f"{section_match.group(1)}{section_match.group(2)}".strip()
        return "Unknown Section"
    
    def index_filing(self, file_path: str) -> str:
        """Index a single SEC filing."""
        try:
            # Load the document
            documents = self.load_documents(file_path)
            print(f"Loaded {len(documents)} document(s) from {file_path}")
            
            # Preprocess the document
            for doc in documents:
                doc.page_content = self.preprocess_sec_filing(doc.page_content)
            
            # Split the document into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            ticker = "UNKNOWN"
            filing_type = "UNKNOWN"
            
            # Try to extract ticker and filing type from filename
            # Assuming format like "AAPL_10-K_2021.pdf"
            parts = filename.split('_')
            if len(parts) >= 2:
                ticker = parts[0]
                filing_type = parts[1]
            
            # Add metadata to chunks
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["ticker"] = ticker
                chunk.metadata["filing_type"] = filing_type
                chunk.metadata["source"] = file_path
                chunk.metadata["section"] = self.extract_section_info(chunk.page_content)
            
            # Create new vector store
            vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            vectorstore.persist()
            
            return f"Successfully indexed {file_path} with {len(chunks)} chunks."
            
        except Exception as e:
            return f"Error indexing filing {file_path}: {str(e)}"
    
    def index_directory(self, directory: str = "data/sec_filings") -> str:
        """Index all SEC filings in a directory (and its subdirectories)."""
        try:
            # Get all files with supported extensions
            file_patterns = [
                os.path.join(directory, "**", "*.pdf"),
                os.path.join(directory, "**", "*.html"),
                os.path.join(directory, "**", "*.htm"),
                os.path.join(directory, "**", "*.txt")
            ]
            
            all_files = []
            for pattern in file_patterns:
                all_files.extend(glob.glob(pattern, recursive=True))
            
            if not all_files:
                return f"No files found in {directory}"
            
            results = []
            for file_path in all_files:
                result = self.index_filing(file_path)
                results.append(result)
            
            return f"Indexed {len(all_files)} files:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error indexing directory {directory}: {str(e)}"
    
    def search(self, query: str, n_results: int = 5, ticker: Optional[str] = None, filing_type: Optional[str] = None) -> List[str]:
        """Search the vector store for documents relevant to the query."""
        try:
            # Load the vector store
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Prepare filter
            filter_dict = {}
            if ticker:
                filter_dict["ticker"] = ticker
            if filing_type:
                filter_dict["filing_type"] = filing_type
            
            # Search with filter if provided
            if filter_dict:
                docs = vectorstore.similarity_search(query, k=n_results, filter=filter_dict)
            else:
                docs = vectorstore.similarity_search(query, k=n_results)
            
            return docs
            
        except Exception as e:
            raise ValueError(f"Error searching SEC filings: {str(e)}")

# Example usage
if __name__ == "__main__":
    indexer = SECIndexer()
    
    # Index a sample filing
    sample_filing = "data/sec_filings/AAPL_10-K_2021.pdf"
    if os.path.exists(sample_filing):
        print(indexer.index_filing(sample_filing))
    
    # Or index an entire directory
    print(indexer.index_directory())
    
    # Search example
    results = indexer.search("What are the risk factors?")
    for doc in results:
        print(f"\n--- Document from {doc.metadata.get('source')} ---")
        print(doc.page_content[:300] + "...\n") 