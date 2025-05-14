# Llama Multi-Agent System

A local multi-agent system using Llama models for various specialized tasks.

## Features

- Multiple specialized agents (Research, Coding, Analytics)
- Agent-to-agent communication
- Web-based user interface
- Real-time updates via WebSocket
- Local deployment with Llama models

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Llama models (GGUF format) from Hugging Face:
   - [llama-3.1b-instruct.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-3.1B-Instruct-GGUF/resolve/main/llama-3.1b-instruct.Q4_K_M.gguf)
   - [codellama-7b.Q4_K_M.gguf](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf)

3. Create a `models` directory and place the downloaded models there:
```bash
mkdir models
# Copy the downloaded .gguf files to the models directory
```

4. Run the server:
```bash
python llama_multi_agent_system.py
```

5. Access the web interface at http://localhost:8000

## Usage

The system provides a web interface where you can interact with the agents. You can ask various types of questions:

- Coding questions: "Write a Python function to calculate Fibonacci numbers"
- Research questions: "What are the main differences between transformer and RNN architectures?"
- Analytics questions: "How would you analyze customer churn in a subscription business?"
- Complex questions that need multiple agents: "Create a data visualization script for analyzing stock market trends"

## System Architecture

The system consists of several components:

1. **Router Agent**: Analyzes user requests and routes them to appropriate specialized agents
2. **Research Agent**: Handles factual questions and research tasks
3. **Coding Agent**: Generates and explains code
4. **Analytics Agent**: Performs data analysis and generates insights

Each agent runs on a specialized Llama model optimized for its specific task. The system uses:
- Llama 3.1B Instruct model for most agents (router, research, analytics)
- CodeLlama 7B for the coding agent

## Requirements

- Python 3.8+
- 8GB+ RAM recommended (reduced from 16GB due to smaller model size)
- GPU recommended for better performance
- Internet connection for initial model download 