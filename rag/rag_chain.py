from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class SECRAGChain:
    def __init__(self, persist_directory: str = "vectorstore/chroma_db"):
        """Initialize the SEC RAG chain."""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        
        # Define prompt template
        self.template = """You are a financial analyst AI assistant specializing in SEC filings.
Use the following pieces of retrieved SEC filing information to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Use three sentences maximum and keep the answer concise.

CONTEXT:
{context}

QUESTION: {question}

YOUR RESPONSE:"""
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
    
    def load_retriever(self):
        """Load or create a retriever with optimized similarity search."""
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # Create a hybrid retriever that combines semantic and keyword search
        from langchain.retrievers import EnsembleRetriever
        from langchain.retrievers import BM25Retriever
        
        # 1. Vector similarity retriever (semantic search)
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,
                "fetch_k": 15,
                "lambda_mult": 0.7,
                "score_threshold": 0.75  # Similarity score cutoff (0-1)
            }
        )
        
        # 2. Sparse retriever (keyword-based)
        # Create a document store from the same documents
        documents = list(vectorstore.get())
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        
        # 3. Combine retrievers with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 70% weight to semantic, 30% to keyword
        )
        
        return ensemble_retriever

    def apply_post_filtering(self, documents, query):
        """Apply post-retrieval filtering to improve precision."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # 1. Re-rank by adaptive threshold
        query_embedding = self.embeddings.embed_query(query)
        
        filtered_docs = []
        for doc in documents:
            # Get document embedding
            doc_embedding = self.embeddings.embed_documents([doc.page_content])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [query_embedding], 
                [doc_embedding]
            )[0][0]
            
            # Apply adaptive threshold based on document type
            threshold = 0.65  # Base threshold
            
            # Adjust threshold based on document metadata
            if "section" in doc.metadata:
                section = doc.metadata["section"].lower()
                # Financial sections need higher relevance
                if "risk" in section or "financial" in section:
                    threshold = 0.7
                # Management discussion can be more lenient
                elif "management" in section:
                    threshold = 0.6
            
            # Only keep documents above threshold
            if similarity >= threshold:
                # Add similarity score to metadata for later use
                doc.metadata["similarity_score"] = float(similarity)
                filtered_docs.append(doc)
        
        # 2. Ensure diversity of sections if we have enough documents
        if len(filtered_docs) > 3:
            # Group by section
            sections = {}
            for doc in filtered_docs:
                section = doc.metadata.get("section", "unknown")
                if section not in sections:
                    sections[section] = []
                sections[section].append(doc)
            
            # Take top document from each section first
            diverse_docs = []
            for section_docs in sections.values():
                # Sort by similarity score
                section_docs.sort(key=lambda x: x.metadata.get("similarity_score", 0), reverse=True)
                diverse_docs.append(section_docs[0])
            
            # Fill remaining slots with highest scoring docs
            remaining = sorted(
                [d for d in filtered_docs if d not in diverse_docs],
                key=lambda x: x.metadata.get("similarity_score", 0),
                reverse=True
            )
            
            # Combine and limit to k
            filtered_docs = diverse_docs + remaining
        
        return filtered_docs[:4]  # Return top 4 documents

    def query(self, question: str) -> str:
        """Query with advanced similarity search and filtering."""
        try:
            # Get retriever
            retriever = self.load_retriever()
            
            # Retrieve initial documents
            documents = retriever.get_relevant_documents(question)
            
            # Apply post-retrieval filtering
            filtered_docs = self.apply_post_filtering(documents, question)
            
            # Create a temporary retriever that returns our filtered docs
            from langchain.schema import BaseRetriever
            
            class FilteredRetriever(BaseRetriever):
                def __init__(self, documents):
                    self.documents = documents
                    
                def get_relevant_documents(self, query):
                    return self.documents
            
            filtered_retriever = FilteredRetriever(filtered_docs)
            
            # Create chain with filtered documents
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=filtered_retriever,
                chain_type_kwargs={"prompt": self.prompt}
            )
            
            # Execute query
            response = chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            return f"Error querying SEC filings: {str(e)}"

    def _extract_entities(self, question: str) -> dict:
        """Extract company tickers and filing types from the query."""
        # Simple regex-based extraction (could be replaced with NER)
        import re
        
        # Extract potential tickers (uppercase 1-5 letter words)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', question)
        
        # Extract filing types
        filing_types = []
        if re.search(r'10[-\s]?[kK]|annual', question, re.IGNORECASE):
            filing_types.append("10-K")
        if re.search(r'10[-\s]?[qQ]|quarterly', question, re.IGNORECASE):
            filing_types.append("10-Q")
        
        # Build filter dict
        filter_dict = {}
        if tickers:
            filter_dict["ticker"] = {"$in": tickers}
        if filing_types:
            filter_dict["filing_type"] = {"$in": filing_types}
        
        return filter_dict

    def _enhance_query(self, question: str) -> str:
        """Add contextual instructions to improve retrieval."""
        # Add financial context to the query
        return f"""
        Financial document query: {question}
        
        This query is about financial information in SEC filings. 
        Consider relevant financial terms, metrics, and regulatory language.
        """

# Example usage
if __name__ == "__main__":
    rag_chain = SECRAGChain()
    
    # Example query
    question = "What are Apple's major risk factors?"
    print(f"Question: {question}")
    print(f"Answer: {rag_chain.query(question)}") 