import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from src.tools.gmail_tools import create_gmail_tools

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def setup_gmail_credentials():
    """Set up Gmail API credentials"""
    credentials = None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    token_path = os.path.join(project_root, "token.json")
    credentials_path = os.path.join(project_root, "credentials.json")

    # Check if token.json exists 
    if os.path.exists(token_path):
        credentials = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If there are no credentials, then authenticate 
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            # Use your credentials.json file
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"credentials.json not found. Looked for: {credentials_path}. "
                    "Place your Google OAuth client file there or set GOOGLE_CREDENTIALS_FILE to its path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save credentials for the next run 
        with open(token_path, 'w') as token:
            token.write(credentials.to_json())

    return credentials


def _get_system_prompt() -> str:
    """Define the agent's methodology and capabilities"""
    return """You are a helpful Gmail assistant specialized in email management and organization.


    When asked to draft emails, follow this workflow:
    1. Analyze the user's request to determine the email recipient, content and subject
    2. Use 'draft_email' to create a draft email

    When asked to search and label emails, follow this workflow:
    1. Use 'search_emails' to find relevant emails (returns email IDs)
    2. Use 'get_email_content' to retrieve the content of each email using the IDs
    3. Analyze the email content to determine appropriate categorization
    4. Use 'apply_email_label' to apply the appropriate label to each email

    Available predefined labels: "Spam", "News", "University", "Financial", "Personal", "Work", "Promotions", "Meeting", "Other"

    When asked to search for specific email content, follow this workflow:
    1. Use 'search_emails' to find relevant emails (returns email IDs)
    2. Use 'get_email_content' to retrieve the content of each email using the IDs
    3. Analyze the email content to determine if it matches the user's request

    Important guidelines:
    - Always examine email content before applying labels to ensure accurate categorization
    - When search returns multiple emails, process each one individually
    - Parse email IDs carefully from search results (e.g., "Found 3 emails with IDs: abc123, def456" → use "abc123", "def456")
    - Be specific about what actions were taken on each email
    - If an email doesn't clearly fit a category, explain why and suggest the best match

    Always be thorough and accurate"""


class GmailAgent:
    """Gmail Agent for email management and organization"""

    def __init__(self, credentials: Credentials, model_name: str = "qwen3:8b", temperature: float = 0.1):
        """Initialize the Gmail agent"""
        self.credentials = credentials
        self.system_prompt = _get_system_prompt()
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.tools: List[BaseTool] = []

    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent's toolkit and bind it to the model"""
        self.tools.append(tool)
        self.model = self.model.bind_tools(self.tools)

    def get_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return self.tools

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Process messages and return a response"""
        response = self.model.invoke(messages)
        return response

    def process_request(self, request: str) -> str:
        """Process a single email-related request"""
        # Add system prompt and user request
        messages = [
            {"role": "system", "content": self.system_prompt},
            HumanMessage(content=request)
        ]

        # Keep processing until the desired output is received
        iteration = 0
        while True:
            iteration += 1
            response = self.invoke(messages)
            messages.append(response)

            # If there are tool calls, execute them
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Find the matching tool
                    tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                    if tool:
                        try:
                            # Execute the tool
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
                # No more tool calls, return the final response
                return response.content

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent's configuration"""
        return {
            "name": "Gmail Agent",
            "model": self.model.model,
            "temperature": self.model.temperature,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools]
        }


# Factory function to create a configured Gmail agent
def create_gmail_agent(model_name: str = "qwen3:8b") -> GmailAgent:
    """Create a Gmail agent with tools"""

    credentials = setup_gmail_credentials()
    tools = create_gmail_tools(credentials)

    agent = GmailAgent(credentials=credentials, model_name=model_name)

    # Add tools
    if tools:
        for tool in tools:
            agent.add_tool(tool)

    return agent


# Example usage
if __name__ == "__main__":
    # Initialize Agent
    agent = create_gmail_agent(model_name="qwen3:8b")

    # Labeling Test
    request = "Search for 5 emails with college and assign the appropriate label to each one"
    response = agent.process_request(request)
    print(response)

    # Draft Test
    request = "Draft an email to send to my Professor about a late assignment"
    response = agent.process_request(request)
    print(response)