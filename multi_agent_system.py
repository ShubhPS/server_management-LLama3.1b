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
            paginated_files = ticket_files[offset:offset + limit]

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
            "X-Title": "ITSM & Operations Automation Portal"
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
            "X-Title": "ITSM & Operations Automation Portal"
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
                print(f"Issue detected: {issue_result['issue']}")
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

        result = {
            "response": response,
            "result": response  # Add this line to maintain compatibility with existing frontend
        }
        if auto_ticket:
            result["auto_ticket"] = auto_ticket

        return result


# Initialize agents
vision_agent = VisionAgent()
text_agent = TextAgent()
ticket_agent = TicketAgent()

issue_detection_agent = IssueDetectionAgent()
coordinator = CoordinatorAgent()
coordinator.register_agent(vision_agent)
coordinator.register_agent(text_agent)
coordinator.register_agent(ticket_agent)
coordinator.register_agent(issue_detection_agent)

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
        <title>ITSM & Operations Automation Portal</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
            }
            .navbar-brand {
                font-weight: 600;
            }
            .card {
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .card-header {
                font-weight: 600;
                background-color: #f8f9fa;
                border-bottom: 1px solid rgba(0,0,0,0.125);
            }
            .priority-badge {
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }
            .priority-critical {
                background-color: #dc3545;
                color: white;
            }
            .priority-high {
                background-color: #fd7e14;
                color: white;
            }
            .priority-medium {
                background-color: #ffc107;
                color: black;
            }
            .priority-low {
                background-color: #6c757d;
                color: white;
            }
            .status-badge {
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
            }
            .status-open {
                background-color: #0d6efd;
                color: white;
            }
            .status-in-progress {
                background-color: #6f42c1;
                color: white;
            }
            .status-resolved {
                background-color: #198754;
                color: white;
            }
            .status-closed {
                background-color: #6c757d;
                color: white;
            }
            .dashboard-card {
                text-align: center;
                padding: 20px;
            }
            .dashboard-number {
                font-size: 36px;
                font-weight: 700;
                margin: 10px 0;
            }
            .ticket-list {
                max-height: 600px;
                overflow-y: auto;
            }
            .nav-tabs .nav-link {
                font-weight: 500;
            }
            .tab-content {
                padding: 20px;
                background-color: white;
                border: 1px solid #dee2e6;
                border-top: none;
                border-radius: 0 0 8px 8px;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 4px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
            <div class="container">
                <a class="navbar-brand" href="#">
                    <i class="bi bi-gear-fill me-2"></i>ITSM & Operations Automation Portal
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item">
                            <a class="nav-link active" href="#"><i class="bi bi-house-door me-1"></i>Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="#"><i class="bi bi-ticket-perforated me-1"></i>Tickets</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="#"><i class="bi bi-book me-1"></i>Knowledge Base</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container">
            <!-- Dashboard Overview -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Open Tickets</h6>
                            <div class="dashboard-number text-primary" id="openTicketsCount">-</div>
                            <p class="card-text"><small>Last updated: <span id="lastUpdated">-</span></small></p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Critical Issues</h6>
                            <div class="dashboard-number text-danger" id="criticalCount">-</div>
                            <p class="card-text"><small>Requires immediate attention</small></p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">SLA Compliance</h6>
                            <div class="dashboard-number text-success" id="slaCompliance">-</div>
                            <p class="card-text"><small>Based on last 30 days</small></p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Auto-Resolved</h6>
                            <div class="dashboard-number text-info" id="autoResolved">-</div>
                            <p class="card-text"><small>Issues fixed automatically</small></p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content Area -->
            <div class="row">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <ul class="nav nav-tabs card-header-tabs" id="mainTabs" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="tickets-tab" data-bs-toggle="tab" data-bs-target="#tickets" type="button" role="tab">Ticket Management</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="ai-assistant-tab" data-bs-toggle="tab" data-bs-target="#ai-assistant" type="button" role="tab">AI Assistant</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="vision-tab" data-bs-toggle="tab" data-bs-target="#vision" type="button" role="tab">Vision Analysis</button>
                                </li>
                            </ul>
                        </div>
                        <div class="card-body p-0">
                            <div class="tab-content" id="mainTabsContent">
                                <!-- Tickets Tab -->
                                <div class="tab-pane fade show active" id="tickets" role="tabpanel">
                                    <div class="mb-3">
                                        <h5>Create New Ticket</h5>
                                        <form id="ticketForm" class="row g-3">
                                            <div class="col-md-12">
                                                <label for="ticketIssue" class="form-label">Issue Description</label>
                                                <textarea id="ticketIssue" name="issue" class="form-control" rows="3" placeholder="Describe the issue in detail" required></textarea>
                                            </div>
                                            <div class="col-md-6">
                                                <label for="ticketImportance" class="form-label">Importance</label>
                                                <select id="ticketImportance" name="importance" class="form-select" required>
                                                    <option value="low">Low</option>
                                                    <option value="medium" selected>Medium</option>
                                                    <option value="high">High</option>
                                                    <option value="critical">Critical</option>
                                                </select>
                                            </div>
                                            <div class="col-12">
                                                <button type="submit" class="btn btn-primary">Create Ticket</button>
                                            </div>
                                        </form>
                                    </div>
                                    <div id="ticketResult" class="alert alert-success mt-3" style="display: none;"></div>
                                </div>

                                <!-- AI Assistant Tab -->
                                <div class="tab-pane fade" id="ai-assistant" role="tabpanel">
                                    <h5>AI Support Assistant</h5>
                                    <p class="text-muted">Ask questions or describe issues for automated assistance</p>
                                    <form id="textForm">
                                        <div class="mb-3">
                                            <textarea id="textPrompt" name="prompt" class="form-control" rows="5" placeholder="Describe your IT issue or ask a question..." required></textarea>
                                        </div>
                                        <button type="submit" class="btn btn-primary">Get Assistance</button>
                                    </form>
                                    <div id="textResult" class="mt-3 p-3 bg-light rounded" style="display: none;"></div>
                                    <div id="autoTicketNotification" class="alert alert-success mt-3" style="display: none;"></div>
                                </div>

                                <!-- Vision Analysis Tab -->
                                <div class="tab-pane fade" id="vision" role="tabpanel">
                                    <h5>Visual Issue Analysis</h5>
                                    <p class="text-muted">Upload screenshots or images for AI analysis</p>
                                    <form id="visionForm" enctype="multipart/form-data">
                                        <div class="mb-3">
                                            <label for="imageUpload" class="form-label">Upload Image</label>
                                            <input class="form-control" type="file" id="imageUpload" name="image" accept="image/*" required>
                                        </div>
                                        <div class="mb-3">
                                            <label for="visionPrompt" class="form-label">Analysis Instructions</label>
                                            <textarea id="visionPrompt" name="prompt" class="form-control" rows="3" placeholder="What would you like to know about this image?">Analyze this screenshot and identify any errors or issues</textarea>
                                        </div>
                                        <button type="submit" class="btn btn-primary">Analyze Image</button>
                                    </form>
                                    <div id="visionResult" class="mt-3 p-3 bg-light rounded" style="display: none;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span>Recent Tickets</span>
                            <div class="input-group" style="width: 60%;">
                                <input type="text" id="ticketSearch" class="form-control form-control-sm" placeholder="Search tickets...">
                                <button class="btn btn-sm btn-outline-secondary" type="button" onclick="searchTickets()">
                                    <i class="bi bi-search"></i>
                                </button>
                            </div>
                        </div>
                        <div class="card-body p-0">
                            <div id="ticketList" class="ticket-list p-3">
                                <div class="d-flex justify-content-center">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Load tickets on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadTickets();
                updateDashboardStats();
                startTicketPolling();
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
                visionResult.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Processing...</span></div></div>';

                try {
                    const res = await fetch('/vision', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await res.json();
                    visionResult.innerHTML = `<div class="mb-2"><strong>Analysis Result:</strong></div><div>${data.result.replace(/\\n/g, '<br>')}</div>`;
                } catch (error) {
                    visionResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                }
            });

            // Text form submission
            document.getElementById('textForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const prompt = document.getElementById('textPrompt').value;
                const textResult = document.getElementById('textResult');
                textResult.style.display = 'block';
                textResult.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Processing...</span></div></div>';

                try {
                    const res = await fetch('/text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ prompt })
                    });

                    const data = await res.json();

                    if (data.result) {
                        textResult.innerHTML = data.result.replace(/\\n/g, '<br>');
                    } else if (data.response) {
                        textResult.innerHTML = data.response.replace(/\\n/g, '<br>');
                    } else {
                        textResult.innerHTML = '<div class="alert alert-warning">No response content received</div>';
                    }

                    if (data.auto_ticket && data.auto_ticket.created) {
                        const notification = document.getElementById('autoTicketNotification');
                        notification.style.display = 'block';
                        notification.innerHTML = `<strong>Auto-generated ticket created:</strong><br>
                                                Issue: ${data.auto_ticket.issue}<br>
                                                Importance: ${data.auto_ticket.importance}`;
                        // Refresh the ticket list
                        loadTickets();
                        updateDashboardStats();
                    }
                } catch (error) {
                    textResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                }
            });

            // Ticket form submission
            document.getElementById('ticketForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const issue = document.getElementById('ticketIssue').value;
                const importance = document.getElementById('ticketImportance').value;

                const ticketResult = document.getElementById('ticketResult');
                ticketResult.style.display = 'block';
                ticketResult.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Creating ticket...</span></div></div>';

                try {
                    const res = await fetch('/ticket', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 
                            issue, 
                            importance
                        })
                    });

                    const data = await res.json();
                    ticketResult.innerHTML = `<div class="alert alert-success">${data.result}</div>`;

                    // Reload ticket list and update stats
                    loadTickets();
                    updateDashboardStats();

                    // Clear form
                    document.getElementById('ticketIssue').value = '';
                } catch (error) {
                    ticketResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                }
            });

            // Load tickets function
            async function loadTickets() {
                const ticketList = document.getElementById('ticketList');
                ticketList.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>';

                try {
                    const res = await fetch('/tickets/list?limit=10');
                    const data = await res.json();

                    if (data.status === 'success') {
                        if (data.tickets.length === 0) {
                            ticketList.innerHTML = "<p class='text-center text-muted'>No tickets found.</p>";
                        } else {
                            ticketList.innerHTML = data.tickets.map(ticket => {
                                // Determine priority class
                                let priorityClass = 'priority-medium';
                                if (ticket.importance === 'critical') {
                                    priorityClass = 'priority-critical';
                                } else if (ticket.importance === 'high') {
                                    priorityClass = 'priority-high';
                                } else if (ticket.importance === 'low') {
                                    priorityClass = 'priority-low';
                                }

                                // Format the date
                                const createdDate = new Date(ticket.created_at);
                                const formattedDate = createdDate.toLocaleString();

                                return `
                                    <div class="card mb-3">
                                        <div class="card-body p-3">
                                            <div class="d-flex justify-content-between align-items-center mb-2">
                                                <span class="badge status-${ticket.status}">${ticket.status.toUpperCase()}</span>
                                                <span class="badge ${priorityClass}">${ticket.importance.toUpperCase()}</span>
                                            </div>
                                            <h6 class="card-title">${ticket.issue.substring(0, 50)}${ticket.issue.length > 50 ? '...' : ''}</h6>
                                            <div class="d-flex justify-content-between align-items-center mt-3">
                                                <small class="text-muted">${formattedDate}</small>
                                                <button class="btn btn-sm btn-outline-danger" onclick="deleteTicket('${ticket.ticket_id}')">
                                                    <i class="bi bi-trash"></i>
                                                </button>
                                            </div>
                                            ${ticket.auto_generated ? '<div class="mt-2"><span class="badge bg-info">Auto-generated</span></div>' : ''}
                                        </div>
                                    </div>
                                `;
                            }).join('');
                        }
                    } else {
                        ticketList.innerHTML = '<div class="alert alert-danger">Error loading tickets</div>';
                    }
                } catch (error) {
                    ticketList.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
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
                            loadTickets();
                            updateDashboardStats();
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
                ticketList.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Searching...</span></div></div>';

                try {
                    const res = await fetch(`/tickets/search?q=${encodeURIComponent(query)}`);
                    const data = await res.json();

                    if (data.status === 'success') {
                        if (data.tickets.length === 0) {
                            ticketList.innerHTML = "<p class='text-center text-muted'>No matching tickets found.</p>";
                        } else {
                            // Use the same rendering logic as loadTickets
                            ticketList.innerHTML = data.tickets.map(ticket => {
                                // Determine priority class
                                let priorityClass = 'priority-medium';
                                if (ticket.importance === 'critical') {
                                    priorityClass = 'priority-critical';
                                } else if (ticket.importance === 'high') {
                                    priorityClass = 'priority-high';
                                } else if (ticket.importance === 'low') {
                                    priorityClass = 'priority-low';
                                }

                                // Format the date
                                const createdDate = new Date(ticket.created_at);
                                const formattedDate = createdDate.toLocaleString();

                                return `
                                    <div class="card mb-3">
                                        <div class="card-body p-3">
                                            <div class="d-flex justify-content-between align-items-center mb-2">
                                                <span class="badge status-${ticket.status}">${ticket.status.toUpperCase()}</span>
                                                <span class="badge ${priorityClass}">${ticket.importance.toUpperCase()}</span>
                                            </div>
                                            <h6 class="card-title">${ticket.issue.substring(0, 50)}${ticket.issue.length > 50 ? '...' : ''}</h6>
                                            <div class="d-flex justify-content-between align-items-center mt-3">
                                                <small class="text-muted">${formattedDate}</small>
                                                <button class="btn btn-sm btn-outline-danger" onclick="deleteTicket('${ticket.ticket_id}')">
                                                    <i class="bi bi-trash"></i>
                                                </button>
                                            </div>
                                            ${ticket.auto_generated ? '<div class="mt-2"><span class="badge bg-info">Auto-generated</span></div>' : ''}
                                        </div>
                                    </div>
                                `;
                            }).join('');
                        }
                    } else {
                        ticketList.innerHTML = '<div class="alert alert-danger">Error searching tickets</div>';
                    }
                } catch (error) {
                    ticketList.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                }
            }

            // Update dashboard statistics
            async function updateDashboardStats() {
                try {
                    // Get all tickets to calculate stats
                    const res = await fetch('/tickets/list?limit=100');
                    const data = await res.json();

                    if (data.status === 'success') {
                        const tickets = data.tickets;
                        const openTickets = tickets.filter(t => t.status === 'open').length;
                        const criticalTickets = tickets.filter(t => t.importance === 'critical').length;
                        const autoGenerated = tickets.filter(t => t.auto_generated).length;

                        document.getElementById('openTicketsCount').textContent = openTickets;
                        document.getElementById('criticalCount').textContent = criticalTickets;
                        document.getElementById('autoResolved').textContent = autoGenerated;
                        document.getElementById('slaCompliance').textContent = '-';  // Placeholder
                        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
                    }
                } catch (error) {
                    console.error('Error updating dashboard stats:', error);
                }
            }

            // Ticket polling function
            function startTicketPolling() {
                // Check for new tickets every 10 seconds
                setInterval(loadTickets, 10000);
                // Update dashboard stats every minute
                setInterval(updateDashboardStats, 60000);
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
async def process_text(prompt: Dict[str, str] = Body(...), request: Request = None):
    """Process text with text agent and automatically create tickets if issues detected"""
    try:
        client_ip = request.client.host if request else "unknown"

        result = await coordinator.process({
            "type": "text",
            "prompt": prompt.get("prompt", ""),
            "ip": client_ip
        })

        return result
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