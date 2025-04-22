import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent and tools
from agents.finance_agent import FinanceAgent
from tools.plot_tool import PlotTool

# Page configuration
st.set_page_config(
    page_title="FinAgent: Financial Analysis Assistant",
    page_icon="💼",
    layout="wide",
)

# Initialize agent
@st.cache_resource
def get_agent():
    return FinanceAgent()

agent = get_agent()
plot_tool = PlotTool()

# Header
st.title("💼 FinAgent: Financial Analysis Assistant")
st.markdown("""
This AI-powered assistant can answer questions about:
- 📊 Stock market data and company financials
- 📈 Macroeconomic indicators (GDP, CPI, interest rates)
- 📑 SEC filings and company reports
- 📉 Historical financial trends and visualizations
""")

# Sidebar
with st.sidebar:
    st.header("About FinAgent")
    st.info("""
    FinAgent combines RAG over financial documents with 
    structured data analysis and visualization tools.
    
    Powered by LangChain and OpenAI.
    """)
    
    st.header("Sample Questions")
    st.markdown("""
    - What was Apple's revenue growth in the last fiscal year?
    - Plot the S&P 500 performance for the past 6 months
    - What are the key risk factors mentioned in Tesla's latest 10-K?
    - How has inflation (CPI) changed since 2020?
    - Compare Microsoft and Google stock performance
    """)
    
    # Optional file uploader for SEC filings
    st.header("Upload SEC Filing")
    uploaded_file = st.file_uploader("Upload a 10-K or 10-Q filing (PDF)", type=["pdf"])
    if uploaded_file:
        with open(os.path.join("data/sec_filings", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")

# Main chat interface
st.header("Ask Your Financial Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "image" in message:
            st.image(message["image"])
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("What would you like to know about financial markets or companies?")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):
            # Get response from agent
            response = agent.run(prompt)
            
            # Check if response contains a plot path
            plot_path = None
            if "Plot generated successfully" in response and "saved to" in response:
                # Extract plot path
                import re
                match = re.search(r"saved to (.*\.png)", response)
                if match:
                    plot_path = match.group(1)
                    
                    # Display the plot
                    if os.path.exists(plot_path):
                        image = Image.open(plot_path)
                        st.image(image, caption="Generated Financial Visualization")
                        
                        # Store image in session state
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="PNG")
                        
                        # Add assistant response with image to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "image": img_bytes.getvalue()
                        })
                    else:
                        st.warning(f"Plot file not found: {plot_path}")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Display text response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*FinAgent is a proof-of-concept application. Financial data may not be real-time.*") 