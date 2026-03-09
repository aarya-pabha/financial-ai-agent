# 🏛️ Wall St. AI Analyst: Hybrid RAG & Live Market Data Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-black.svg)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-green.svg)](https://openai.com/)

## 🚀 Live Interactive Demo
**Interact with the deployed agent here:** [Wall St. AI Analyst on Hugging Face Spaces](https://huggingface.co/spaces/Aarya003/Financial-Analyst-Agent)

## 📌 Project Overview
Standard Retrieval-Augmented Generation (RAG) applications fail in the financial sector because they attempt to extract highly volatile, real-time numbers (like stock prices or PE ratios) from static, outdated PDF documents. 

The **Wall St. AI Analyst** solves this by implementing a **Semantic Routing Architecture**. It dynamically intercepts user queries and routes them to the appropriate specialized tool: a Vector Database for deep, qualitative 10-K analysis, or a live financial API for quantitative, real-time trading metrics.

## 📂 Repository Structure
```text
financial-ai-agent/
│
├── app.py                                 # The final Streamlit application & routing logic
├── nasdaq-listed.csv                      # Universe of supported tickers for Market Data
├── requirements.txt                       # Explicit dependencies for deployment
│
├── evaluation/                            
│   ├── phase3_validation_set.csv          # Synthetic evaluation questions & contexts
│   └── phase3_ragas_scores_adjusted.csv   # Final Ragas scores correcting for honest refusals
│
├── notebooks/                             
│   └── Financial_RAG.ipynb # Complete logic, architecture, and Ragas evaluation flow
│
└── README.md                              # Project documentation
```

🏗️ System Architecture & Engineering Decisions
-----------------------------------------------

This agent was built with a decoupled architecture prioritizing strict boundary enforcement to completely eliminate LLM hallucinations.

### 1\. The Ingestion Pipeline (Handling Massive Financial Tables)

Financial SEC 10-K reports contain "Consolidated Financial Statements"—massive grids of data. Standard character-based text splitters destroy these tables by cutting them in half, causing the LLM to lose column headers and hallucinate numbers.

*   **Solution:** I utilized LlamaIndex's MarkdownElementNodeParser. This structural parser treats Markdown tables as atomic elements, refusing to split them.
    
*   **Trade-off & Fix:** This resulted in massive context chunks (up to 3,000 tokens per table). To ensure the LLM found the exact data without overflowing context windows, I engineered the retrieval engine to fetch an expanded window (similarity\_top\_k=8) coupled with strict Metadata Filtering (e.g., ticker='AAPL') to prevent cross-company data contamination.
    

### 2\. The Pydantic Semantic Router

The core decision engine is powered by an OpenAIPydanticProgram. Instead of letting the LLM guess how to answer, the router enforces strict tool selection:

*   **Tool 1 (financial\_rag):** Triggered for internal company details, historical revenue, risks, and strategy. Queries Pinecone.
    
*   **Tool 2 (market\_data):** Triggered ONLY for real-time trading metrics (Price, Market Cap, Volume). Queries the yfinance API for all NASDAQ-listed companies.
    
*   **Hybrid Mode:** If a user asks _"Compare Tesla's current price to its supply chain risks"_, the router triggers _both_ tools in parallel and synthesizes the final answer.
    

📊 Rigorous Evaluation Pipeline (Ragas)
---------------------------------------

An AI agent is only as good as its validation. I evaluated this system against a custom synthetic benchmark of 20 high-complexity financial questions using the **Ragas** framework, utilizing gpt-4o as the judge.

### Overcoming the "Honest Refusal" Penalty

To prevent hallucinations (the "Nvidia Trap"), the agent was strictly prompted: _"If the data is missing, simply output: 'Data not available in the context.'"_ While the agent successfully followed this rule, standard Ragas algorithms penalized these honest refusals with a 0.0 Answer Relevancy score because the evaluator LLM could not reverse-engineer the original question from a refusal.

I engineered a custom Python evaluation script (available in the notebooks/ directory) to algorithmically adjust for these correct programmatic refusals, revealing the system's true, production-grade metrics:

*   **Faithfulness:** 0.8349 (83.5% of generated claims are perfectly backed by the retrieved 10-K context).
    
*   **Answer Relevancy:** 0.9177 (91.8% direct alignment with user intent, without conversational filler).
    

💻 Tech Stack
-------------

*   **LLM & Embeddings:** OpenAI (gpt-4o-mini, gpt-4o, text-embedding-3-small)
    
*   **Orchestration:** LlamaIndex (v0.10+)
    
*   **Vector Database:** Pinecone
    
*   **Market Data API:** yfinance
    
*   **Evaluation Framework:** Ragas, Pandas, AsyncIO
    
*   **Frontend/Deployment:** Streamlit, Hugging Face Spaces
    

🛠️ Running Locally
-------------------

1.  git clone https://github.com/aarya-pabha/financial-ai-agent.git
  
2.  cd financial-ai-agent
    
3.  pip install -r requirements.txt
    
4. OPENAI\_API\_KEY="sk-..."PINECONE\_API\_KEY="pcsk\_..."
    
5.  streamlit run app.py
