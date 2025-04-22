from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
from langchain.tools import BaseTool
from dotenv import load_dotenv
import os
import sys
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any, Tuple
import json
import re
from langchain.vectorstores import Chroma

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

    def assemble_context(self, documents):
        """Assemble retrieved chunks into a coherent context."""
        # Sort by a combination of relevance and logical section order
        documents.sort(key=lambda doc: (
            -doc.metadata.get("similarity_score", 0),  # Higher score first
            self._get_section_order(doc.metadata.get("section", ""))  # Logical section order
        ))
        
        # Format each chunk with metadata for context
        formatted_chunks = []
        for i, doc in enumerate(documents):
            # Add section and source information
            section = doc.metadata.get("section", "Unknown Section")
            source = doc.metadata.get("source", "").split("/")[-1]
            ticker = doc.metadata.get("ticker", "")
            
            formatted_chunk = f"""
[CHUNK {i+1}]
SOURCE: {ticker} - {source}
SECTION: {section}
CONTENT:
{doc.page_content}
"""
            formatted_chunks.append(formatted_chunk)
        
        # Join chunks with clear separators
        return "\n\n" + "\n\n".join(formatted_chunks)

    def create_prompt_template(self):
        """Create an optimized prompt template for financial document RAG."""
        template = """
You are a financial analyst assistant with expertise in SEC filings and financial documents.

CONTEXT INFORMATION:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question" - DO NOT make up information.
3. Focus on financial accuracy and precision.
4. Cite specific sections or documents when possible.
5. Use financial terminology appropriately.
6. Present numerical data clearly, with proper units and time periods.
7. If appropriate, structure your answer with bullet points or tables for clarity.

YOUR RESPONSE:
"""
        return PromptTemplate(template=template, input_variables=["context", "query"])

    def create_cot_prompt_template(self):
        """Create a chain-of-thought prompt for complex financial analysis."""
        template = """
You are a financial analyst assistant with expertise in SEC filings and financial documents.

CONTEXT INFORMATION:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. First, identify the key financial concepts and data points needed to answer this question.
2. Then, locate this information in the provided context.
3. Analyze the information step-by-step, showing your reasoning.
4. Finally, provide a clear, concise answer that directly addresses the question.
5. Cite specific sections or documents.
6. If the context doesn't contain the answer, say so clearly.

STEP-BY-STEP ANALYSIS:
"""
        return PromptTemplate(template=template, input_variables=["context", "query"])

    def select_prompt_template(self, query):
        """Select the appropriate prompt template based on query complexity."""
        # Check for complex analysis needs
        complex_indicators = [
            "compare", "trend", "analysis", "evaluate", 
            "impact", "forecast", "projection", "risk"
        ]
        
        if any(indicator in query.lower() for indicator in complex_indicators):
            return self.create_cot_prompt_template()
        else:
            return self.create_prompt_template()

class FinanceAgentMemory:
    """Memory management optimized for financial documents with all-MiniLM-L6-v2."""
    
    def __init__(self, max_token_limit: int = 8000):
        # Initialize with higher token limit for newer GPT-3.5-Turbo
        self.memory_settings = {
            "auto_prune": True,
            "retain_companies": True,
            "memory_mode": "balanced",
            "context_window": max_token_limit,
            # Add specific settings for your embedding model
            "chunk_size": 512,  # Optimal for all-MiniLM-L6-v2
            "max_chunks_per_query": 6,  # Limit chunks to prevent context overflow
            "min_similarity_score": 0.75  # Higher threshold for your model
        }
        
        # Short-term verbatim memory for recent exchanges
        self.buffer_memory = ConversationBufferMemory(
            memory_key="recent_history",
            return_messages=True,
            input_key="input",
            output_key="output",
            max_token_limit=1000  # Keep last ~1000 tokens verbatim
        )
        
        # Long-term summarized memory
        self.summary_memory = ConversationSummaryMemory(
            memory_key="conversation_summary",
            return_messages=True,
            input_key="input", 
            output_key="output",
            llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        )
        
        # Combined memory system
        self.memory = CombinedMemory(
            memories=[self.buffer_memory, self.summary_memory]
        )
        
        # Financial context memory - stores entities and data points
        self.financial_context = {
            "companies_discussed": set(),
            "metrics_mentioned": set(),
            "time_periods": set(),
            "retrieved_documents": {}  # Map query hash -> document references
        }
        
        # Token limit for entire memory
        self.max_token_limit = max_token_limit
        
        # Memory usage tracking
        self.memory_usage = {
            "buffer_tokens": 0,
            "summary_tokens": 0,
            "financial_context_tokens": 0,
            "total_tokens": 0
        }
    
    def add_interaction(self, user_input: str, agent_response: str, 
                        retrieved_docs: List[Any] = None):
        """Add a new interaction to memory."""
        # Add to conversational memory
        self.memory.save_context(
            {"input": user_input},
            {"output": agent_response}
        )
        
        # Extract and store financial entities
        self._update_financial_context(user_input, agent_response)
        
        # Store document references if provided
        if retrieved_docs:
            import hashlib
            query_hash = hashlib.md5(user_input.encode()).hexdigest()
            self.financial_context["retrieved_documents"][query_hash] = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "section": doc.metadata.get("section", "unknown"),
                    "ticker": doc.metadata.get("ticker", "unknown")
                }
                for doc in retrieved_docs[:3]  # Store references to top 3 docs
            ]
    
    def _update_financial_context(self, user_input: str, agent_response: str):
        """Extract financial entities and update context."""
        combined_text = user_input + " " + agent_response
        
        # Extract company tickers/names (simplified - would use NER in production)
        import re
        # Look for ticker patterns (1-5 uppercase letters)
        tickers = set(re.findall(r'\b[A-Z]{1,5}\b', combined_text))
        self.financial_context["companies_discussed"].update(tickers)
        
        # Extract financial metrics
        metrics = ["revenue", "profit", "margin", "eps", "ebitda", "income", 
                  "cash flow", "debt", "assets", "liabilities"]
        for metric in metrics:
            if metric in combined_text.lower():
                self.financial_context["metrics_mentioned"].add(metric)
        
        # Extract time periods
        time_periods = ["quarter", "Q1", "Q2", "Q3", "Q4", "annual", "year", 
                       "2021", "2022", "2023", "2024"]
        for period in time_periods:
            if period in combined_text:
                self.financial_context["time_periods"].add(period)
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get all memory variables for prompt context."""
        # Get standard memory variables
        memory_variables = self.memory.load_memory_variables({})
        
        # Add financial context summary
        financial_summary = self._generate_financial_context_summary()
        memory_variables["financial_context"] = financial_summary
        
        return memory_variables
    
    def _generate_financial_context_summary(self) -> str:
        """Generate a concise summary of financial context."""
        companies = list(self.financial_context["companies_discussed"])
        metrics = list(self.financial_context["metrics_mentioned"])
        periods = list(self.financial_context["time_periods"])
        
        summary_parts = []
        
        if companies:
            summary_parts.append(f"Companies discussed: {', '.join(companies[:5])}")
        if metrics:
            summary_parts.append(f"Financial metrics mentioned: {', '.join(metrics[:5])}")
        if periods:
            summary_parts.append(f"Time periods referenced: {', '.join(periods[:5])}")
        
        if not summary_parts:
            return ""
        
        return "CONVERSATION CONTEXT:\n" + "\n".join(summary_parts)
    
    def get_relevant_docs_from_memory(self, query: str) -> List[Dict]:
        """Retrieve previously accessed document references relevant to query."""
        # Simple relevance check - would use embeddings in production
        relevant_docs = []
        
        # Check if we've seen similar queries before
        for stored_docs in self.financial_context["retrieved_documents"].values():
            for doc in stored_docs:
                # Check if doc ticker is mentioned in current query
                if doc["ticker"] in query:
                    relevant_docs.append(doc)
        
        return relevant_docs[:3]  # Return top 3 most relevant
    
    def clear(self):
        """Clear all memory."""
        self.buffer_memory.clear()
        self.summary_memory.clear()
        self.financial_context = {
            "companies_discussed": set(),
            "metrics_mentioned": set(),
            "time_periods": set(),
            "retrieved_documents": {}
        }
    
    def update_memory_settings(self, settings_dict: Dict[str, Any]) -> str:
        """Update memory management settings based on user preferences."""
        valid_updates = {}
        
        # Validate and apply settings
        for key, value in settings_dict.items():
            if key in self.memory_settings:
                # Type checking
                if key == "auto_prune" and isinstance(value, bool):
                    valid_updates[key] = value
                elif key == "retain_companies" and isinstance(value, bool):
                    valid_updates[key] = value
                elif key == "memory_mode" and value in ["balanced", "detailed", "minimal"]:
                    valid_updates[key] = value
                elif key == "context_window" and isinstance(value, int) and 1000 <= value <= 8000:
                    valid_updates[key] = value
        
        # Apply valid updates
        self.memory_settings.update(valid_updates)
        
        # Adjust memory based on new settings
        self._adjust_memory_for_settings()
        
        return f"Memory settings updated: {valid_updates}"
    
    def _adjust_memory_for_settings(self):
        """Adjust memory based on current settings."""
        mode = self.memory_settings["memory_mode"]
        
        if mode == "detailed":
            # Maximize buffer memory, reduce summarization
            self.buffer_memory.max_token_limit = int(self.memory_settings["context_window"] * 0.7)
        elif mode == "minimal":
            # Minimize buffer, rely on summaries
            self.buffer_memory.max_token_limit = int(self.memory_settings["context_window"] * 0.3)
            # Force summarization
            self._force_summarize()
        else:  # balanced
            # Default balanced approach
            self.buffer_memory.max_token_limit = int(self.memory_settings["context_window"] * 0.5)
    
    def _force_summarize(self):
        """Force summarization of buffer memory."""
        # Get current buffer content
        buffer_content = self.buffer_memory.load_memory_variables({})
        
        if "recent_history" in buffer_content and buffer_content["recent_history"]:
            # Extract messages
            messages = buffer_content["recent_history"]
            
            # Create input for summarization
            conversation_string = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages
            ])
            
            # Generate summary
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            summary = llm.predict(
                f"Summarize this financial conversation, focusing on key financial entities, metrics, and insights. Keep important numerical data:\n\n{conversation_string}"
            )
            
            # Update summary memory
            self.summary_memory.chat_memory.add_user_message("Previous conversation")
            self.summary_memory.chat_memory.add_ai_message(summary)
            
            # Clear buffer memory
            self.buffer_memory.clear()
    
    def detect_memory_commands(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Detect and process memory management commands in user input."""
        # Define command patterns
        commands = {
            r"(?i)forget (everything|all)": {"action": "clear_all"},
            r"(?i)forget about ([A-Z]{1,5})": {"action": "forget_company", "params": {"ticker": r"\1"}},
            r"(?i)focus on ([A-Z]{1,5})": {"action": "focus_company", "params": {"ticker": r"\1"}},
            r"(?i)set memory to (detailed|balanced|minimal)": {"action": "set_mode", "params": {"mode": r"\1"}},
            r"(?i)summarize (conversation|our discussion)": {"action": "summarize"},
            r"(?i)show memory (usage|status)": {"action": "show_status"}
        }
        
        # Check for commands
        cleaned_input = user_input
        command_result = None
        
        for pattern, command_info in commands.items():
            match = re.search(pattern, user_input)
            if match:
                # Extract command and parameters
                action = command_info["action"]
                params = {}
                
                if "params" in command_info:
                    for param_name, param_pattern in command_info["params"].items():
                        # Extract parameter using the specified group
                        param_value = match.expand(param_pattern)
                        params[param_name] = param_value
                
                # Process command
                command_result = self._process_memory_command(action, params)
                
                # Remove command from input
                cleaned_input = re.sub(pattern, "", user_input).strip()
                break
        
        return cleaned_input, command_result
    
    def _process_memory_command(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a memory management command."""
        result = {"action": action, "success": True, "message": ""}
        
        try:
            if action == "clear_all":
                self.clear()
                result["message"] = "Memory cleared. Starting fresh conversation."
            
            elif action == "forget_company":
                ticker = params.get("ticker", "")
                if ticker in self.financial_context["companies_discussed"]:
                    self.financial_context["companies_discussed"].remove(ticker)
                    result["message"] = f"I'll no longer reference {ticker} from our previous conversation."
                else:
                    result["message"] = f"I don't have {ticker} in my memory."
            
            elif action == "focus_company":
                ticker = params.get("ticker", "")
                # Keep only this company in memory
                other_companies = set(self.financial_context["companies_discussed"])
                self.financial_context["companies_discussed"] = {ticker}
                result["message"] = f"I'll focus on {ticker} and temporarily set aside information about {', '.join(other_companies - {ticker})}."
            
            elif action == "set_mode":
                mode = params.get("mode", "balanced").lower()
                self.update_memory_settings({"memory_mode": mode})
                result["message"] = f"Memory mode set to {mode}."
            
            elif action == "summarize":
                self._force_summarize()
                summary = self.summary_memory.load_memory_variables({}).get("conversation_summary", "")
                result["message"] = f"Here's a summary of our conversation:\n\n{summary}"
            
            elif action == "show_status":
                self._update_memory_usage()
                usage = self.memory_usage
                settings = self.memory_settings
                companies = list(self.financial_context["companies_discussed"])
                
                status = f"Memory Status:\n"
                status += f"- Mode: {settings['memory_mode']}\n"
                status += f"- Usage: {usage['total_tokens']}/{settings['context_window']} tokens\n"
                status += f"- Companies in memory: {', '.join(companies[:5])}\n"
                status += f"- Auto-pruning: {'Enabled' if settings['auto_prune'] else 'Disabled'}"
                
                result["message"] = status
        
        except Exception as e:
            result["success"] = False
            result["message"] = f"Error processing memory command: {str(e)}"
        
        return result
    
    def _update_memory_usage(self):
        """Update memory usage statistics."""
        # Get memory content
        buffer_vars = self.buffer_memory.load_memory_variables({})
        summary_vars = self.summary_memory.load_memory_variables({})
        
        # Estimate token counts (rough approximation)
        self.memory_usage["buffer_tokens"] = len(json.dumps(buffer_vars).split())
        self.memory_usage["summary_tokens"] = len(json.dumps(summary_vars).split())
        self.memory_usage["financial_context_tokens"] = len(json.dumps(self.financial_context).split())
        
        # Calculate total
        self.memory_usage["total_tokens"] = (
            self.memory_usage["buffer_tokens"] + 
            self.memory_usage["summary_tokens"] + 
            self.memory_usage["financial_context_tokens"]
        )
    
    def prune_if_needed(self):
        """Prune memory if it exceeds token limit."""
        # Skip if auto-prune is disabled
        if not self.memory_settings["auto_prune"]:
            return
        
        # Update usage stats
        self._update_memory_usage()
        
        # Check if pruning is needed
        if self.memory_usage["total_tokens"] > self.memory_settings["context_window"]:
            # Strategy depends on memory mode
            mode = self.memory_settings["memory_mode"]
            
            if mode == "detailed":
                # Reduce buffer slightly but preserve detail
                self.buffer_memory.max_token_limit = int(self.buffer_memory.max_token_limit * 0.8)
            elif mode == "minimal":
                # Aggressively summarize
                self._force_summarize()
            else:  # balanced
                # Standard approach - summarize and reduce buffer
                self._force_summarize()
                self.buffer_memory.max_token_limit = int(self.buffer_memory.max_token_limit * 0.9)

    def load_retriever(self):
        """Load retriever with optimized context limits."""
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # Configure retriever with optimal settings for your model and document size
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.memory_settings["max_chunks_per_query"],
                "fetch_k": 15,
                "lambda_mult": 0.7,
                "score_threshold": self.memory_settings["min_similarity_score"]
            }
        )
        
        return retriever

    def optimize_context_length(self, context, max_tokens=3000):
        """Ensure context fits within allocated token budget."""
        # Adjust max_tokens to match your context allocation
        tokens = len(context.split())
        
        if tokens > max_tokens:
            # Prioritize keeping the most relevant chunks
            # ... existing optimization code ...
        
        return context

class FinanceAgent:
    """Agent for financial analysis with enhanced memory management."""
    
    def __init__(self):
        """Initialize the finance agent with tools, LLM, and memory."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Initialize advanced memory
        self.memory_manager = FinanceAgentMemory(max_token_limit=8000)
        
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
            memory=self.memory_manager.buffer_memory,  # Use buffer memory for agent
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def run(self, query: str) -> str:
        """Run the agent on a query with enhanced memory management."""
        try:
            # Check for memory commands
            cleaned_query, command_result = self.memory_manager.detect_memory_commands(query)
            
            # If this was a memory command with no additional query
            if command_result and not cleaned_query:
                return command_result["message"]
            
            # Use cleaned query for processing
            query = cleaned_query
            
            # Check if we have relevant document references in memory
            relevant_docs = self.memory_manager.get_relevant_docs_from_memory(query)
            
            # If we have relevant docs, add them as context
            if relevant_docs:
                context_str = "Previously retrieved relevant documents:\n"
                for doc in relevant_docs:
                    context_str += f"- {doc['ticker']} {doc['source']} ({doc['section']})\n"
                query = f"{context_str}\n\nWith this in mind, please answer: {query}"
            
            # Get memory variables
            memory_vars = self.memory_manager.get_memory_variables()
            
            # Add financial context to query if available
            if "financial_context" in memory_vars and memory_vars["financial_context"]:
                query = f"{memory_vars['financial_context']}\n\n{query}"
            
            # Run agent
            response = self.agent.run(input=query)
            
            # Add memory command result to response if applicable
            if command_result:
                response = f"{command_result['message']}\n\n{response}"
            
            # Store interaction in memory
            self.memory_manager.add_interaction(query, response)
            
            # Prune memory if needed
            self.memory_manager.prune_if_needed()
            
            return response
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory_manager.clear()
        return "Memory cleared. Starting fresh conversation."

# Example usage
if __name__ == "__main__":
    agent = FinanceAgent()
    response = agent.run("What were Apple's revenue and profit margins in the last quarter?")
    print(response)