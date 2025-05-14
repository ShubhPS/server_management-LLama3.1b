"""
Multi-Agent System for Llama Vision
===================================
A system of specialized agents working together to handle complex queries
"""

import os
import json
import time
import base64
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import re

import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ===============================
# Ticket Models
# ===============================

class Ticket(BaseModel):
    issue: str
    importance: str
    time: str
    ip: str
    ticket_id: str = str(uuid.uuid4())
    status: str = "open"
    created_at: str = datetime.now().isoformat()
    auto_generated: bool = False

class TicketRequest(BaseModel):
    issue: str
    importance: str

# ===============================
# Base Agent Class
# ===============================

class Agent(ABC):
    """Base class for all agents in the system"""

    def __init__(self, name: str):
        self.name = name
        self.memory = []

    @abstractmethod
    async def process(self, input_data: Any) -> str:
        """Process input and return response"""
        pass

    def add_to_memory(self, data: Any):
        """Add information to agent's memory"""
        self.memory.append(data)

    def get_memory(self) -> List[Any]:
        """Retrieve agent's memory"""
        return self.memory

# ===============================
# Specialized Agents
# ===============================

class VisionAgent(Agent):
    """Agent specialized in visual understanding"""

    def __init__(self):
        super().__init__("Vision Agent")
        self.api_key = "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"  # Your OpenRouter API key
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process image and prompt"""
        image_data = input_data.get('image_data')
        prompt = input_data.get('prompt', 'Describe this image')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",  # Replace with your site
            "X-Title": "Multi-Agent Llama Vision System"
        }

        base64_image = base64.b64encode(image_data).decode('utf-8')

        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",  # Or another vision-capable model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            self.add_to_memory({"type": "vision_analysis", "result": result})
            return result
        except Exception as e:
            return f"Error in vision processing: {str(e)}"


class TextAgent(Agent):
    """Agent specialized in text processing and reasoning"""

    def __init__(self):
        super().__init__("Text Agent")
        self.api_key = "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"  # Your OpenRouter API key
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process text query"""
        prompt = input_data.get('prompt')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",  # Replace with your site
            "X-Title": "Multi-Agent Llama Vision System"
        }

        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",  # Choose an appropriate model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            self.add_to_memory({"type": "text_analysis", "result": result})
            return result
        except Exception as e:
            return f"Error in text processing: {str(e)}"


class TicketAgent(Agent):
    """Agent specialized in handling support tickets"""

    def __init__(self):
        super().__init__("Ticket Agent")
        self.tickets: Dict[str, Ticket] = {}

    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process ticket creation and management"""
        action = input_data.get('action', 'create')

        if action == 'create':
            ticket = Ticket(
                issue=input_data['issue'],
                importance=input_data['importance'],
                time=datetime.now().isoformat(),
                ip=input_data['ip'],
                auto_generated=input_data.get('auto_generated', False)
            )
            self.tickets[ticket.ticket_id] = ticket
            self.add_to_memory({"type": "ticket_created", "ticket_id": ticket.ticket_id})
            return f"Ticket created successfully. Ticket ID: {ticket.ticket_id}"

        elif action == 'list':
            return json.dumps([ticket.dict() for ticket in self.tickets.values()])

        elif action == 'get':
            ticket_id = input_data.get('ticket_id')
            if ticket_id in self.tickets:
                return json.dumps(self.tickets[ticket_id].dict())
            return "Ticket not found"

        return "Invalid action"

class IssueDetectionAgent(Agent):
    """Agent specialized in detecting issues that require a support ticket"""

    def __init__(self):
        super().__init__("Issue Detection Agent")
        # Keywords that might indicate issues requiring tickets
        self.issue_keywords = [
            "error", "bug", "crash", "fail", "broken", "not working",
            "help", "problem", "issue", "fatal", "exception", "down",
            "doesn't work", "malfunction", "glitch", "fault", "defect",
            "urgent", "emergency", "critical", "fix", "repair",
            "unresponsive", "freeze", "hang", "timeout"
        ]

        # Severity indicators to determine ticket importance
        self.severity_indicators = {
            "high": ["urgent", "critical", "emergency", "fatal", "immediately", "security", "breach", "data loss"],
            "medium": ["important", "significant", "moderate", "soon", "affecting", "performance"],
            "low": []  # Default level if no high or medium indicators are found
        }

    def determine_importance(self, text: str) -> str:
        """Determine ticket importance based on text content"""
        text = text.lower()

        # Check for high severity indicators
        for indicator in self.severity_indicators["high"]:
            if indicator in text:
                return "critical"

        # Check for medium severity indicators
        for indicator in self.severity_indicators["medium"]:
            if indicator in text:
                return "high"

        # Default to medium importance for any detected issue
        return "medium"

    def detect_issue(self, text: str) -> bool:
        """Detect if text contains issue indicators"""
        text = text.lower()

        # Check for issue keywords
        for keyword in self.issue_keywords:
            if keyword in text:
                return True

        # Additional pattern matching for error messages
        error_patterns = [
            r"error\s*:\s*.*",
            r"exception\s*:\s*.*",
            r"failed\s*to\s*.*",
            r"cannot\s*.*",
            r"unable\s*to\s*.*"
        ]

        for pattern in error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text to detect issues and create ticket data if needed"""
        text = input_data.get('text', '')

        if self.detect_issue(text):
            importance = self.determine_importance(text)

            # Format the issue text
            issue = f"Auto-detected issue: {text[:200]}" + ("..." if len(text) > 200 else "")

            result = {
                "issue_detected": True,
                "issue": issue,
                "importance": importance
            }

            self.add_to_memory({"type": "issue_detected", "issue": issue, "importance": importance})
            return result

        return {"issue_detected": False}

class CoordinatorAgent(Agent):
    """Agent that coordinates between other agents"""

    def __init__(self):
        super().__init__("Coordinator Agent")
        self.agents = {}

    def register_agent(self, agent: Agent):
        """Register a new agent"""
        self.agents[agent.name] = agent

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate between agents and combine their responses"""
        query_type = input_data.get('type', 'text')
        prompt = input_data.get('prompt', '')

        # First, check if there's an issue requiring a ticket
        issue_agent = self.agents.get('Issue Detection Agent')
        ticket_agent = self.agents.get('Ticket Agent')
        auto_ticket = None

        if issue_agent and ticket_agent and query_type == 'text':
            issue_result = await issue_agent.process({'text': prompt})

            if issue_result.get('issue_detected', False):
                # Create a ticket automatically
                ticket_result = await ticket_agent.process({
                    'action': 'create',
                    'issue': issue_result['issue'],
                    'importance': issue_result['importance'],
                    'ip': input_data.get('ip', '127.0.0.1'),
                    'auto_generated': True
                })
                auto_ticket = {
                    "created": True,
                    "message": ticket_result,
                    "issue": issue_result['issue'],
                    "importance": issue_result['importance']
                }

        # Process the actual query
        response = ""

        if query_type == 'vision':
            vision_agent = self.agents.get('Vision Agent')
            if vision_agent:
                response = await vision_agent.process(input_data)
        elif query_type == 'text':
            text_agent = self.agents.get('Text Agent')
            if text_agent:
                response = await text_agent.process(input_data)
        else:
            # For complex queries, we can combine multiple agents
            combined_response = []
            for agent in self.agents.values():
                if agent.name not in ['Issue Detection Agent']:  # Skip the issue detection in combined response
                    agent_response = await agent.process(input_data)
                    combined_response.append(f"{agent.name}: {agent_response}")
            response = "\n".join(combined_response)

        self.add_to_memory({
            "type": "coordination",
            "query_type": query_type,
            "response": response,
            "auto_ticket": auto_ticket
        })

        result = {"response": response}
        if auto_ticket:
            result["auto_ticket"] = auto_ticket

        return result

# ===============================
# FastAPI Implementation
# ===============================

app = FastAPI(title="Multi-Agent Llama Vision System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
coordinator = CoordinatorAgent()
vision_agent = VisionAgent()
text_agent = TextAgent()
ticket_agent = TicketAgent()
issue_agent = IssueDetectionAgent()

# Register agents with coordinator
coordinator.register_agent(vision_agent)
coordinator.register_agent(text_agent)
coordinator.register_agent(ticket_agent)
coordinator.register_agent(issue_agent)

class QueryRequest(BaseModel):
    query: str
    type: str = "text"

@app.post("/query")
async def submit_query(request: QueryRequest, client_request: Request):
    """Submit a query to the multi-agent system"""
    try:
        response = await coordinator.process({
            "prompt": request.query,
            "type": request.type,
            "ip": client_request.client.host
        })
        return response
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/vision")
async def submit_vision_query(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Submit a vision query to the multi-agent system"""
    try:
        contents = await image.read()
        response = await coordinator.process({
            "image_data": contents,
            "prompt": prompt,
            "type": "vision"
        })
        return {"status": "success", "response": response["response"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/tickets")
async def create_ticket(request: TicketRequest, client_request: Request):
    """Create a new support ticket"""
    try:
        response = await ticket_agent.process({
            "action": "create",
            "issue": request.issue,
            "importance": request.importance,
            "ip": client_request.client.host
        })
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/tickets")
async def list_tickets():
    """List all tickets"""
    try:
        response = await ticket_agent.process({"action": "list"})
        return {"status": "success", "tickets": json.loads(response)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get a specific ticket"""
    try:
        response = await ticket_agent.process({
            "action": "get",
            "ticket_id": ticket_id
        })
        return {"status": "success", "ticket": json.loads(response)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Multi-Agent Llama Vision System</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                textarea { width: 100%; height: 100px; }
                input, select { width: 100%; padding: 8px; margin: 5px 0; }
                button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #45a049; }
                .response { background-color: #f9f9f9; padding: 15px; border-radius: 5px; white-space: pre-wrap; }
                .ticket-list { margin-top: 20px; }
                .ticket-item { padding: 10px; border: 1px solid #ddd; margin-bottom: 10px; border-radius: 4px; }
                .notification { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin-top: 15px; display: none; }
                .auto-ticket { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin-top: 15px; display: none; }
            </style>
        </head>
        <body>
            <h1>Multi-Agent Llama Vision System</h1>
            
            <div class="section">
                <h2>Text Query</h2>
                <form id="textForm">
                    <div>
                        <label for="textPrompt">Prompt:</label><br>
                        <textarea id="textPrompt" name="query" placeholder="Enter your prompt here..."></textarea>
                    </div>
                    <button type="submit">Submit Text Query</button>
                </form>
                <div id="autoTicketNotification" class="auto-ticket"></div>
                <div id="textResponse" class="response"></div>
            </div>
            
            <div class="section">
                <h2>Vision Query</h2>
                <form id="visionForm" enctype="multipart/form-data">
                    <div>
                        <label for="imageUpload">Upload Image:</label><br>
                        <input type="file" id="imageUpload" name="image" accept="image/*">
                    </div>
                    <div>
                        <label for="visionPrompt">Prompt:</label><br>
                        <textarea id="visionPrompt" name="prompt" placeholder="Ask about the image..."></textarea>
                    </div>
                    <button type="submit">Submit Vision Query</button>
                </form>
                <div id="visionResponse" class="response"></div>
            </div>

            <div class="section">
                <h2>Create Support Ticket</h2>
                <form id="ticketForm">
                    <div>
                        <label for="issue">Issue Description:</label><br>
                        <textarea id="issue" name="issue" placeholder="Describe your issue..."></textarea>
                    </div>
                    <div>
                        <label for="importance">Importance:</label><br>
                        <select id="importance" name="importance">
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                            <option value="critical">Critical</option>
                        </select>
                    </div>
                    <button type="submit">Submit Ticket</button>
                </form>
                <div id="ticketResponse" class="response"></div>
            </div>

            <div class="section">
                <h2>Recent Tickets</h2>
                <button onclick="loadTickets()">Refresh Tickets</button>
                <div id="ticketList" class="ticket-list"></div>
            </div>
            
            <script>
                // Text form handler with auto-ticket feature
                document.getElementById('textForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const prompt = document.getElementById('textPrompt').value;
                    const response = document.getElementById('textResponse');
                    const autoTicketNotification = document.getElementById('autoTicketNotification');
                    
                    response.textContent = "Loading...";
                    autoTicketNotification.style.display = "none";
                    
                    try {
                        const res = await fetch('/query', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ query: prompt, type: 'text' }),
                        });
                        
                        const data = await res.json();
                        
                        // Display the response
                        if (data.response) {
                            response.textContent = data.response;
                        } else if (data.status === 'error') {
                            response.textContent = `Error: ${data.error}`;
                        }
                        
                        // Check if an auto-ticket was created
                        if (data.auto_ticket && data.auto_ticket.created) {
                            autoTicketNotification.style.display = "block";
                            autoTicketNotification.innerHTML = `
                                <strong>Automatic Ticket Created</strong><br>
                                We detected an issue in your message and created a support ticket automatically:<br>
                                ${data.auto_ticket.issue}<br>
                                <strong>Importance:</strong> ${data.auto_ticket.importance}<br>
                                ${data.auto_ticket.message}
                            `;
                            
                            // Refresh the ticket list
                            loadTickets();
                        }
                    } catch (error) {
                        response.textContent = `Error: ${error.message}`;
                    }
                });
                
                // Vision form handler
                document.getElementById('visionForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData();
                    const imageFile = document.getElementById('imageUpload').files[0];
                    const prompt = document.getElementById('visionPrompt').value;
                    
                    if (!imageFile) {
                        alert('Please select an image');
                        return;
                    }
                    
                    formData.append('image', imageFile);
                    formData.append('prompt', prompt);
                    
                    const response = document.getElementById('visionResponse');
                    response.textContent = "Loading...";
                    
                    try {
                        const res = await fetch('/vision', {
                            method: 'POST',
                            body: formData,
                        });
                        
                        const data = await res.json();
                        response.textContent = data.response;
                    } catch (error) {
                        response.textContent = `Error: ${error.message}`;
                    }
                });

                // Ticket form handler
                document.getElementById('ticketForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const issue = document.getElementById('issue').value;
                    const importance = document.getElementById('importance').value;
                    const response = document.getElementById('ticketResponse');
                    
                    response.textContent = "Submitting ticket...";
                    
                    try {
                        const res = await fetch('/tickets', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ issue, importance }),
                        });
                        
                        const data = await res.json();
                        response.textContent = data.response;
                        loadTickets(); // Refresh ticket list
                    } catch (error) {
                        response.textContent = `Error: ${error.message}`;
                    }
                });

                // Load tickets function
                async function loadTickets() {
                    const ticketList = document.getElementById('ticketList');
                    ticketList.innerHTML = "Loading tickets...";
                    
                    try {
                        const res = await fetch('/tickets');
                        const data = await res.json();
                        
                        if (data.status === 'success') {
                            if (data.tickets.length === 0) {
                                ticketList.innerHTML = "<p>No tickets found.</p>";
                            } else {
                                ticketList.innerHTML = data.tickets.map(ticket => `
                                    <div class="ticket-item">
                                        <strong>Ticket ID:</strong> ${ticket.ticket_id}<br>
                                        <strong>Issue:</strong> ${ticket.issue}<br>
                                        <strong>Importance:</strong> ${ticket.importance}<br>
                                        <strong>Status:</strong> ${ticket.status}<br>
                                        <strong>Created:</strong> ${new Date(ticket.created_at).toLocaleString()}<br>
                                        ${ticket.auto_generated ? '<strong style="color:#28a745">Auto-generated</strong>' : ''}
                                    </div>
                                `).join('');
                            }
                        } else {
                            ticketList.innerHTML = "Error loading tickets";
                        }
                    } catch (error) {
                        ticketList.innerHTML = `Error: ${error.message}`;
                    }
                }

                // Load tickets on page load
                loadTickets();
            </script>
        </body>
    </html>
    """

def main():
    """Main entry point for the server"""
    import uvicorn
    print("Starting Multi-Agent Llama Vision System...")
    print("\nServer will be available at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()