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
        """Load the vector store retriever."""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store not found at {self.persist_directory}. Please index some documents first.")
        
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    
    def create_chain(self):
        """Create the RAG chain."""
        retriever = self.load_retriever()
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        return chain
    
    def query(self, question: str) -> str:
        """Query the RAG chain with a question."""
        try:
            chain = self.create_chain()
            response = chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            return f"Error querying SEC filings: {str(e)}"

# Example usage
if __name__ == "__main__":
    rag_chain = SECRAGChain()
    
    # Example query
    question = "What are Apple's major risk factors?"
    print(f"Question: {question}")
    print(f"Answer: {rag_chain.query(question)}") 