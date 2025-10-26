# Receipt Analyzer LLM

**Receipt Analyzer LLM** is an AI-powered application that uses **Qwen 2.5** models to extract and analyze shopping receipts.  
It parses receipts from PDF or image files, stores them in a database, and allows users to ask natural language questions about their spending, stores, or items.

---

## Overview

The system combines document parsing, structured storage, and large language model reasoning.  
Users can upload receipts, automatically extract information, and interact with their data through a Gradio-based chat interface.

---

## Features

- Receipt extraction from PDF and image files  
- Qwen2.5-1.5B and Qwen2.5-7B LLM support for reasoning  
- Local SQLite database for structured storage  
- Natural language queries about spend, stores, and items  
- Off-topic question filtering  
- Comparative evaluation of model accuracy and latency  

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/receipt-analyzer-llm.git
cd receipt-analyzer-llm

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
