# Llama Multi-Agent System

A local multi-agent system using Llama models for various specialized tasks.

## Features

- Multiple specialized agents (Research, Coding, Analytics)
- Agent-to-agent communication
- Web-based user interface
- Real-time updates via WebSocket
- Local deployment with Llama models

## Setup

1. Install the multi_agent_system.py script then Install dependencies in your terminal(pls make sure your python version is updated to 3.8+):

`pip install fastapi uvicorn python-dotenv requests pydantic jinja2 python-multipart`


3. Download Llama models (GGUF format) from Hugging Face:
   - [llama-3.1b-instruct.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-3.1B-Instruct-GGUF/resolve/main/llama-3.1b-instruct.Q4_K_M.gguf)
OR use an Open source Endpoint

4. Regardless of what you do in step 2, setup your model and setup your API Key as an environment variable in an .env. Then change your Endpoint address url in the TextAgent and VisionAgent. *NOTE : This is key as unless you do this the script will not work*

5. Run the script

6. Access the web interface at http://localhost:8000

## Usage

The system provides a web interface where you can interact with the agents. You can ask various types of questions:

- Coding questions: "Write a Python function to calculate Fibonacci numbers"
- Research questions: "What are the main differences between transformer and RNN architectures?"
- Analytics questions: "How would you analyze customer churn in a subscription business?"
- Complex questions that need multiple agents: "Create a data visualization script for analyzing stock market trends"

## System Architecture

The system consists of several components:

1. **IssueIdentifier Agent**: Analyzes user prompts and matches patterns for common issues to generate a prompt for error in the code
2. **Coordinator Agent**: Works with the IssueIdentifier Agent and Ticket Agent to 
3. **Ticket Agent**: Handles tickets and their generation/creation
4. **Coding Agent**: Generates and explains code
5. **Analytics Agent**: Performs data analysis and generates insights

Each agent runs on a specialized Llama model optimized for its specific task. The system uses:
- Llama 3.1B Instruct model for most agents (router, research, analytics)

## Requirements

- Python 3.8+
- 8GB+ RAM recommended (reduced from 16GB due to smaller model size)
- GPU recommended for better performance
- Internet connection for initial model download 
