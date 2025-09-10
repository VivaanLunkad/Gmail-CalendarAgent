import os
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from src.tools.calendar_tools import create_calendar_tools

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def setup_calendar_credentials():
    """Set up Calendar API credentials"""
    credentials = None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    token_path = os.path.join(project_root, "calendar_token.json")
    credentials_path = os.path.join(project_root, "credentials.json")

    # Check if token exists
    if os.path.exists(token_path):
        credentials = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If there are no valid credentials, authenticate
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"credentials.json not found at: {credentials_path}. "
                    "Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save credentials
        with open(token_path, 'w') as token:
            token.write(credentials.to_json())

    return credentials


def _get_system_prompt() -> str:
    """Define the agent's methodology and capabilities"""
    # Get current datetime dynamically
    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return f"""You are a helpful Google Calendar assistant specialized in managing calendar events.

Current date and time: {current_datetime}

When asked to create events, follow this workflow:
1. Parse the user's request to extract event details (title, date/time, location, description)
2. Use 'create_calendar_event' to create the event
3. Confirm creation with the event ID and link

When asked to find/search events, follow this workflow:
1. Use 'search_calendar_events' to find events based on text or time range
2. If needed, use 'get_calendar_event' to get full details of specific events
3. Present the results clearly to the user

When asked to update events, follow this workflow:
1. Use 'search_calendar_events' to find the event if ID not provided
2. Use 'update_calendar_event' to modify the event details
3. Confirm the changes made

When asked to delete events, follow this workflow:
1. Use 'search_calendar_events' to find the event if ID not provided
2. Optionally use 'get_calendar_event' to confirm it's the right event
3. Use 'delete_calendar_event' to remove the event
4. Confirm deletion

Important guidelines:
- Be smart about parsing dates and times from natural language using the current date/time as reference
- If time is not specified, ask if it should be an all-day event
- Always confirm successful operations with relevant details
- When multiple events match a search, help the user identify the correct one
- Be clear about what actions were taken

Be helpful, accurate, and efficient in managing calendar events."""


class CalendarAgent:
    """Calendar Agent for event management"""

    def __init__(self, credentials: Credentials, model_name: str = "llama3.2:3b", temperature: float = 0.1):
        """Initialize the Calendar agent"""
        self.credentials = credentials
        self.system_prompt = _get_system_prompt()
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.tools: List[BaseTool] = []

    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent's toolkit"""
        self.tools.append(tool)
        self.model = self.model.bind_tools(self.tools)

    def get_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return self.tools

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Process messages and return a response"""
        return self.model.invoke(messages)

    def process_request(self, request: str) -> str:
        """Process a single calendar-related request"""
        # Regenerate system prompt to get current datetime
        messages = [
            {"role": "system", "content": _get_system_prompt()},
            HumanMessage(content=request)
        ]

        iteration = 0
        while True:
            iteration += 1
            response = self.invoke(messages)
            messages.append(response)

            # Execute tool calls if present
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                    if tool:
                        try:
                            tool_output = tool.invoke(tool_call["args"])
                            tool_message = ToolMessage(
                                content=str(tool_output),
                                tool_call_id=tool_call["id"]
                            )
                            messages.append(tool_message)
                        except Exception as e:
                            error_message = ToolMessage(
                                content=f"Tool execution error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                            messages.append(error_message)
            else:
                # No more tool calls, return final response
                return response.content

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent's configuration"""
        return {
            "name": "Calendar Agent",
            "model": self.model.model,
            "temperature": self.model.temperature,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools]
        }


def create_calendar_agent(model_name: str = "llama3.2:3b") -> CalendarAgent:
    """Create a Calendar agent with tools"""
    credentials = setup_calendar_credentials()
    tools = create_calendar_tools(credentials)

    agent = CalendarAgent(credentials=credentials, model_name=model_name)

    # Add tools
    for tool in tools:
        agent.add_tool(tool)

    return agent


# Example usage
if __name__ == "__main__":
    # Initialize Agent
    agent = create_calendar_agent(model_name="qwen3:8b")

    # Test creating an event
    request = "Create a meeting called 'Team Standup' tomorrow at 10am for 30 minutes"
    response = agent.process_request(request)
    print(response)

    # Test searching events
    request = "Find all my meetings this week"
    response = agent.process_request(request)
    print(response)

    # Test updating an event
    request = "Change the Team Standup meeting to 11am"
    response = agent.process_request(request)
    print(response)

    # Test checking availability
    request = "Am I free tomorrow between 2pm and 5pm?"
    response = agent.process_request(request)
    print(response)