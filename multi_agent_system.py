import os
import json
import base64
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import asyncio
import io
import re
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv("API_KEY")


# Define models
class Ticket(BaseModel):
    ticket_id: str = None
    issue: str
    importance: str
    status: str = "open"
    time: str
    ip: str
    auto_generated: bool = False
    created_at: str = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.ticket_id:
            self.ticket_id = f"ticket_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def dict(self):
        return {
            "ticket_id": self.ticket_id,
            "issue": self.issue,
            "importance": self.importance,
            "status": self.status,
            "time": self.time,
            "ip": self.ip,
            "auto_generated": self.auto_generated,
            "created_at": self.created_at
        }

class TicketManager:
    """Manages persistent storage and retrieval of tickets"""
    
    def __init__(self, storage_path="ticket_storage"):
        self.storage_path = storage_path
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
    
    def save_ticket(self, ticket: Ticket) -> bool:
        """Save a ticket to persistent storage"""
        try:
            file_path = os.path.join(self.storage_path, f"{ticket.ticket_id}.json")
            with open(file_path, 'w') as f:
                json.dump(ticket.dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving ticket: {str(e)}")
            return False
    
    def load_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Load a ticket from persistent storage"""
        file_path = os.path.join(self.storage_path, f"{ticket_id}.json")
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                ticket_data = json.load(f)
                return Ticket(**ticket_data)
        except Exception as e:
            print(f"Error loading ticket: {str(e)}")
            return None
    
    def list_tickets(self, limit: int = 100, offset: int = 0) -> List[Ticket]:
        """List tickets with pagination"""
        tickets = []
        
        # Get all ticket files
        if os.path.exists(self.storage_path):
            ticket_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
            # Sort by creation time (newest first)
            ticket_files.sort(reverse=True)
            
            # Apply pagination
            paginated_files = ticket_files[offset:offset+limit]
            
            # Load ticket data
            for file_name in paginated_files:
                file_path = os.path.join(self.storage_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        ticket_data = json.load(f)
                        tickets.append(Ticket(**ticket_data))
                except Exception as e:
                    print(f"Error loading ticket {file_name}: {str(e)}")
        
        return tickets
    
    def delete_ticket(self, ticket_id: str) -> bool:
        """Delete a ticket from persistent storage"""
        file_path = os.path.join(self.storage_path, f"{ticket_id}.json")
        if not os.path.exists(file_path):
            return False
            
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting ticket: {str(e)}")
            return False
    
    def search_tickets(self, query: str) -> List[Ticket]:
        """Search tickets by content"""
        matching_tickets = []
        
        if os.path.exists(self.storage_path):
            for file_name in os.listdir(self.storage_path):
                if not file_name.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.storage_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        ticket_data = json.load(f)
                        
                        # Search in ticket content
                        ticket_str = json.dumps(ticket_data).lower()
                        if query.lower() in ticket_str:
                            matching_tickets.append(Ticket(**ticket_data))
                except Exception as e:
                    print(f"Error searching ticket {file_name}: {str(e)}")
        
        return matching_tickets

# Define base agent class
class Agent:
    """Base agent class with memory capabilities"""
    
    def __init__(self, name):
        self.name = name
        self.memory = []
    
    def add_to_memory(self, data):
        """Add data to agent memory"""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Keep memory size manageable
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process input data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process method")

# Define specialized agents
class VisionAgent(Agent):
    """Agent specialized in visual understanding"""

    def __init__(self):
        super().__init__("Vision Agent")


        self.api_key = hf_api_key
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process image and prompt"""
        image_data = input_data.get('image_data')
        prompt = input_data.get('prompt', 'Describe this image')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Multi-Agent Llama Vision System"
        }

        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Corrected for Llama 3.1 Vision
        payload = {
            "model": "meta-llama/llama-3.1-8b-vision:free",  # Use the vision-specific model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
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
        self.api_key = hf_api_key
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def process(self, input_data: Dict[str, Any]) -> str:
        """Process text query"""
        prompt = input_data.get('prompt')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Multi-Agent Llama Vision System"
        }

        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
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
            self.add_to_memory({"type": "text_analysis", "result": result})
            return result
        except Exception as e:
            return f"Error in text processing: {str(e)}"

class TicketAgent(Agent):
    """Agent specialized in handling support tickets"""

    def __init__(self):
        super().__init__("Ticket Agent")
        self.tickets: Dict[str, Ticket] = {}
        self.ticket_manager = TicketManager()
        
        # Load existing tickets from storage
        stored_tickets = self.ticket_manager.list_tickets()
        for ticket in stored_tickets:
            self.tickets[ticket.ticket_id] = ticket

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
            # Save to persistent storage
            self.ticket_manager.save_ticket(ticket)
            self.add_to_memory({"type": "ticket_created", "ticket_id": ticket.ticket_id})
            return f"Ticket created successfully. Ticket ID: {ticket.ticket_id}"

        elif action == 'list':
            # Get tickets from storage to ensure we have the latest
            stored_tickets = self.ticket_manager.list_tickets(
                limit=input_data.get('limit', 100),
                offset=input_data.get('offset', 0)
            )
            return json.dumps([ticket.dict() for ticket in stored_tickets])

        elif action == 'get':
            ticket_id = input_data.get('ticket_id')
            # Try to get from memory first, then from storage
            if ticket_id in self.tickets:
                return json.dumps(self.tickets[ticket_id].dict())
            else:
                ticket = self.ticket_manager.load_ticket(ticket_id)
                if ticket:
                    self.tickets[ticket_id] = ticket
                    return json.dumps(ticket.dict())
            return "Ticket not found"
            
        elif action == 'delete':
            ticket_id = input_data.get('ticket_id')
            if ticket_id in self.tickets:
                # Remove from memory
                del self.tickets[ticket_id]
                # Remove from storage
                success = self.ticket_manager.delete_ticket(ticket_id)
                if success:
                    self.add_to_memory({"type": "ticket_deleted", "ticket_id": ticket_id})
                    return f"Ticket {ticket_id} deleted successfully"
                else:
                    return f"Error deleting ticket {ticket_id} from storage"
            else:
                # Check if it exists in storage
                if self.ticket_manager.load_ticket(ticket_id):
                    success = self.ticket_manager.delete_ticket(ticket_id)
                    if success:
                        self.add_to_memory({"type": "ticket_deleted", "ticket_id": ticket_id})
                        return f"Ticket {ticket_id} deleted successfully"
                return "Ticket not found"
                
        elif action == 'search':
            query = input_data.get('query', '')
            if not query:
                return "Search query is required"
                
            matching_tickets = self.ticket_manager.search_tickets(query)
            return json.dumps([ticket.dict() for ticket in matching_tickets])

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

# Initialize agents
vision_agent = VisionAgent()
text_agent = TextAgent()
ticket_agent = TicketAgent()

# Create FastAPI app
app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def home():
    """Render home page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Agent Vision System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .panel {
                flex: 1;
                min-width: 300px;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }
            input, select, textarea {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 4px;
                white-space: pre-wrap;
            }
            .ticket-item {
                background-color: #f8f9fa;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }
            .delete-btn {
                background-color: #dc3545;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
            }
            .delete-btn:hover {
                background-color: #c82333;
            }
            .search-box {
                display: flex;
                margin-bottom: 15px;
            }
            .search-box input {
                flex: 1;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Multi-Agent Vision System</h1>
        
        <div class="container">
            <div class="panel">
                <h2>Vision Analysis</h2>
                <form id="visionForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="imageUpload">Upload Image:</label>
                        <input type="file" id="imageUpload" name="image" accept="image/*" required>
                    </div>
                    <div class="form-group">
                        <label for="visionPrompt">Prompt:</label>
                        <textarea id="visionPrompt" name="prompt" rows="3" placeholder="Describe this image in detail">Describe this image in detail</textarea>
                    </div>
                    <button type="submit">Analyze Image</button>
                </form>
                <div id="visionResult" class="result" style="display: none;"></div>
            </div>
            
            <div class="panel">
                <h2>Text Analysis</h2>
                <form id="textForm">
                    <div class="form-group">
                        <label for="textPrompt">Prompt:</label>
                        <textarea id="textPrompt" name="prompt" rows="5" placeholder="Enter your text query here" required></textarea>
                    </div>
                    <button type="submit">Process Text</button>
                </form>
                <div id="textResult" class="result" style="display: none;"></div>
            </div>
        </div>
        
        <div class="container" style="margin-top: 30px;">
            <div class="panel">
                <h2>Create Support Ticket</h2>
                <form id="ticketForm">
                    <div class="form-group">
                        <label for="ticketIssue">Issue Description:</label>
                        <textarea id="ticketIssue" name="issue" rows="3" placeholder="Describe the issue" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="ticketImportance">Importance:</label>
                        <select id="ticketImportance" name="importance" required>
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                            <option value="critical">Critical</option>
                        </select>
                    </div>
                    <button type="submit">Create Ticket</button>
                </form>
                <div id="ticketResult" class="result" style="display: none;"></div>
            </div>
            
            <div class="panel">
                <h2>Recent Tickets</h2>
                <div class="search-box">
                    <input type="text" id="ticketSearch" placeholder="Search tickets...">
                    <button onclick="searchTickets()">Search</button>
                </div>
                <div id="ticketList">Loading tickets...</div>
            </div>
        </div>
        
        <script>
            // Load tickets on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadTickets();
            });
            
            // Vision form submission
            document.getElementById('visionForm').addEventListener('submit', async function(e) {
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
                
                const visionResult = document.getElementById('visionResult');
                visionResult.style.display = 'block';
                visionResult.textContent = 'Processing image...';
                
                try {
                    const res = await fetch('/vision', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await res.json();
                    visionResult.textContent = data.result;
                } catch (error) {
                    visionResult.textContent = `Error: ${error.message}`;
                }
            });
            
            // Text form submission
            document.getElementById('textForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const prompt = document.getElementById('textPrompt').value;
                
                const textResult = document.getElementById('textResult');
                textResult.style.display = 'block';
                textResult.textContent = 'Processing text...';
                
                try {
                    const res = await fetch('/text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ prompt })
                    });
                    
                    const data = await res.json();
                    textResult.textContent = data.result;
                } catch (error) {
                    textResult.textContent = `Error: ${error.message}`;
                }
            });
            
            // Ticket form submission
            document.getElementById('ticketForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const issue = document.getElementById('ticketIssue').value;
                const importance = document.getElementById('ticketImportance').value;
                
                const ticketResult = document.getElementById('ticketResult');
                ticketResult.style.display = 'block';
                ticketResult.textContent = 'Creating ticket...';
                
                try {
                    const res = await fetch('/ticket', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ issue, importance })
                    });
                    
                    const data = await res.json();
                    ticketResult.textContent = data.result;
                    
                    // Reload ticket list
                    loadTickets();
                    
                    // Clear form
                    document.getElementById('ticketIssue').value = '';
                } catch (error) {
                    ticketResult.textContent = `Error: ${error.message}`;
                }
            });
            
            // Load tickets function
            async function loadTickets() {
                const ticketList = document.getElementById('ticketList');
                ticketList.innerHTML = "Loading tickets...";
                
                try {
                    const res = await fetch('/tickets/list?limit=10');
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
                                    <button onclick="deleteTicket('${ticket.ticket_id}')" class="delete-btn">Delete</button>
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
            
            // Delete ticket function
            async function deleteTicket(ticketId) {
                if (confirm(`Are you sure you want to delete ticket ${ticketId}?`)) {
                    try {
                        const res = await fetch(`/tickets/${ticketId}`, {
                            method: 'DELETE'
                        });
                        
                        const data = await res.json();
                        if (data.status === 'success') {
                            alert(data.message);
                            loadTickets(); // Refresh ticket list
                        } else {
                            alert(`Error: ${data.message}`);
                        }
                    } catch (error) {
                        alert(`Error: ${error.message}`);
                    }
                }
            }
            
            // Search tickets function
            async function searchTickets() {
                const query = document.getElementById('ticketSearch').value;
                if (query.length < 2) {
                    alert('Search query must be at least 2 characters');
                    return;
                }
                
                const ticketList = document.getElementById('ticketList');
                ticketList.innerHTML = "Searching...";
                
                try {
                    const res = await fetch(`/tickets/search?q=${encodeURIComponent(query)}`);
                    const data = await res.json();
                    
                    if (data.status === 'success') {
                        if (data.tickets.length === 0) {
                            ticketList.innerHTML = "<p>No matching tickets found.</p>";
                        } else {
                            ticketList.innerHTML = data.tickets.map(ticket => `
                                <div class="ticket-item">
                                    <strong>Ticket ID:</strong> ${ticket.ticket_id}<br>
                                    <strong>Issue:</strong> ${ticket.issue}<br>
                                    <strong>Importance:</strong> ${ticket.importance}<br>
                                    <strong>Status:</strong> ${ticket.status}<br>
                                    <strong>Created:</strong> ${new Date(ticket.created_at).toLocaleString()}<br>
                                    ${ticket.auto_generated ? '<strong style="color:#28a745">Auto-generated</strong>' : ''}
                                    <button onclick="deleteTicket('${ticket.ticket_id}')" class="delete-btn">Delete</button>
                                </div>
                            `).join('');
                        }
                    } else {
                        ticketList.innerHTML = "Error searching tickets";
                    }
                } catch (error) {
                    ticketList.innerHTML = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/vision")
async def process_vision(image: UploadFile = File(...), prompt: str = Form("Describe this image")):
    """Process image with vision agent"""
    try:
        image_data = await image.read()
        result = await vision_agent.process({
            "image_data": image_data,
            "prompt": prompt
        })
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/text")
async def process_text(prompt: Dict[str, str] = Body(...)):
    """Process text with text agent"""
    try:
        result = await text_agent.process({
            "prompt": prompt.get("prompt", "")
        })
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ticket")
async def create_ticket(ticket_data: Dict[str, str] = Body(...), request: Request = None):
    """Create a new support ticket"""
    try:
        client_ip = request.client.host if request else "unknown"
        
        result = await ticket_agent.process({
            "action": "create",
            "issue": ticket_data.get("issue", ""),
            "importance": ticket_data.get("importance", "medium"),
            "ip": client_ip
        })
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/tickets/{ticket_id}")
async def delete_ticket(ticket_id: str):
    """Delete a specific ticket"""
    try:
        response = await ticket_agent.process({
            "action": "delete",
            "ticket_id": ticket_id
        })
        if "deleted successfully" in response:
            return {"status": "success", "message": response}
        else:
            return {"status": "error", "message": response}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/tickets/search")
async def search_tickets(q: str):
    """Search for tickets"""
    try:
        if not q or len(q) < 2:
            return {"status": "error", "message": "Search query must be at least 2 characters"}
            
        response = await ticket_agent.process({
            "action": "search",
            "query": q
        })
        return {"status": "success", "tickets": json.loads(response)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
        
@app.get("/tickets/list")
async def paginated_tickets(limit: int = 10, offset: int = 0):
    """List tickets with pagination"""
    try:
        response = await ticket_agent.process({
            "action": "list",
            "limit": limit,
            "offset": offset
        })
        return {"status": "success", "tickets": json.loads(response)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
