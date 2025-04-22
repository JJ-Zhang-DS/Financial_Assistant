from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from dotenv import load_dotenv
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools
from tools.fred_tool import FREDTool
from tools.yahoo_tool import YahooTool
from tools.sec_tool import SECParserTool
from tools.plot_tool import PlotTool

# Import RAG chain
from rag.rag_chain import SECRAGChain

# Load environment variables
load_dotenv()

class RAGToolWrapper(BaseTool):
    """Wrapper for the SEC RAG Chain to make it compatible with the agent."""
    name = "sec_rag_tool"
    description = """
    Use this tool to answer questions about SEC filings like 10-K and 10-Q reports.
    This tool searches through a database of company filings to find relevant information.
    Use this for questions about company financials, risk factors, business descriptions, etc.
    """
    
    def __init__(self):
        super().__init__()
        self.rag_chain = SECRAGChain()
    
    def _run(self, query: str) -> str:
        return self.rag_chain.query(query)

class FinanceAgent:
    """Agent for financial analysis using LangChain."""
    
    def __init__(self):
        """Initialize the finance agent with tools and LLM."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",
            temperature=0
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize tools
        self.tools = [
            FREDTool(),
            YahooTool(),
            SECParserTool(),
            PlotTool(),
            RAGToolWrapper()
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def run(self, query: str) -> str:
        """Run the agent on a query."""
        try:
            response = self.agent.run(input=query)
            return response
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"

# Example usage
if __name__ == "__main__":
    agent = FinanceAgent()
    response = agent.run("What were Apple's revenue and profit margins in the last quarter?")
    print(response)