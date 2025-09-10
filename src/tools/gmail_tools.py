from typing import Dict, List, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Pre-defined labels for emails
DEFAULT_LABELS = [
    "Spam",
    "News",
    "University",
    "Financial",
    "Personal",
    "Work",
    "Promotions",
    "Meeting",
    "Other"
]


class GmailBaseTool(BaseTool):
    """Base class for all Gmail tools"""

    _service: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

    def set_service(self, service: Any):
        """Set the Gmail service instance"""
        self._service = service

    @staticmethod
    def _handle_error(error: Exception, context: str) -> str:
        """Centralized error handling"""
        if isinstance(error, HttpError):
            if hasattr(error, "resp") and hasattr(error.resp, "status"):
                if error.resp.status == 404:
                    return f"Error: Resource not found in {context}"
                elif error.resp.status == 403:
                    return f"Error: Permission denied for {context}"
            return f"API Error in {context}: {error}"
        return f"Error in {context}: {str(error)}"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class EmailHelper:
    """Helper class for all email operations"""

    @staticmethod
    def create_message(to: str, subject: str, body: str,
                       cc: Optional[List[str]] = None,
                       bcc: Optional[List[str]] = None,
                       is_html: bool = False) -> Dict[str, Any]:
        """Create a message for an email"""
        if is_html:
            message = MIMEMultipart('alternative')
            message.attach(MIMEText(body, 'html'))
        else:
            message = MIMEText(body)

        message['to'] = to
        message['subject'] = subject

        if cc:
            message['cc'] = ', '.join(cc)
        if bcc:
            message['bcc'] = ', '.join(bcc)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {'raw': raw}

    @staticmethod
    def parse_email_body(payload: Dict[str, Any]) -> str:
        """Extract email body from payload"""
        body = ""

        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

        return body.strip()


class GmailToolkit:
    """Factory class for creating all Gmail tools"""

    def __init__(self, credentials: Credentials, allowed_labels: Optional[List[str]] = None):
        self.service = build('gmail', 'v1', credentials=credentials)
        self.allowed_labels = allowed_labels or DEFAULT_LABELS
        self._label_cache: Dict[str, str] = {}

    def get_label_id(self, label_name: str) -> Optional[str]:
        """Get label ID from label name"""
        try:
            results = self.service.users().labels().list(userId='me').execute()
            labels = results.get('labels', [])

            for label in labels:
                if label['name'].lower() == label_name.lower():
                    return label['id']
            return None
        except Exception as e:
            print(f"Error getting label ID for label '{label_name}': {str(e)}")
            return None

    def create_tools(self) -> List[BaseTool]:
        """Create all Gmail tools with a shared service"""
        # Create tools
        draft_tool = GmailCreateDraftTool()
        draft_tool.set_service(self.service)

        search_tool = GmailSearchTool()
        search_tool.set_service(self.service)

        get_tool = GmailGetEmailTool()
        get_tool.set_service(self.service)

        label_tool = GmailApplyLabelTool(
            allowed_labels=self.allowed_labels,
            get_label_id=self.get_label_id
        )
        label_tool.set_service(self.service)

        return [draft_tool, search_tool, get_tool, label_tool]


class EmailDraftInput(BaseModel):
    """Input schema for creating an email draft"""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    cc: Optional[List[str]] = Field(default=None, description="CC recipients")
    bcc: Optional[List[str]] = Field(default=None, description="BCC recipients")
    is_html: bool = Field(default=False, description="Whether the body is HTML format")


class GmailCreateDraftTool(GmailBaseTool):
    """Tool for creating email drafts in Gmail"""

    name: str = "create_gmail_draft"
    description: str = """Create a draft email in Gmail. 
    Use this when you need to compose an email but not send it immediately.
    Returns the draft ID that can be used to send or edit the draft later."""
    args_schema: type[BaseModel] = EmailDraftInput

    def _run(self, to: str, subject: str, body: str,
             cc: Optional[List[str]] = None,
             bcc: Optional[List[str]] = None,
             is_html: bool = False) -> str:
        try:
            message = EmailHelper.create_message(to, subject, body, cc, bcc, is_html)
            draft = {'message': message}

            result = self._service.users().drafts().create(
                userId='me',
                body=draft
            ).execute()

            draft_id = result['id']
            return f"Successfully created draft with ID: {draft_id}. The email draft to {to} with subject '{subject}' is ready for review."

        except Exception as e:
            return self._handle_error(e, "creating draft email")


class ApplyLabelInput(BaseModel):
    """Input schema for applying a label to an email"""
    email_id: str = Field(description="The ID of the email to label")
    label: str = Field(description="The pre-defined label name to apply to the email")


class GmailApplyLabelTool(GmailBaseTool):
    """Tool for applying labels to emails"""

    name: str = "apply_email_label"
    description: str = """Apply a label to an email that has been analyzed.
    Use this after determining which category/label an email belongs to.
    The label must be one of the pre-defined labels configured for the system."""
    args_schema: type[BaseModel] = ApplyLabelInput
    allowed_labels: List[str] = Field(default_factory=list)
    get_label_id: Any = Field(default=None)

    def __init__(self, allowed_labels: List[str], get_label_id: Any, **data):
        super().__init__(**data)
        self.allowed_labels = allowed_labels
        self.get_label_id = get_label_id
        self.description += f"\nAllowed labels: {', '.join(allowed_labels)}"

    def _run(self, email_id: str, label: str) -> str:
        # Validate label
        if label.lower() not in [l.lower() for l in self.allowed_labels]:
            return f"Error: '{label}' is not an allowed label. Allowed labels are: {', '.join(self.allowed_labels)}"

        try:
            # Get label ID
            label_id = self.get_label_id(label)
            if not label_id:
                return f"Error: Label '{label}' not found in Gmail. Please ensure it exists in your Gmail account."

            # Apply the label
            body = {'addLabelIds': [label_id]}

            result = self._service.users().messages().modify(
                userId='me',
                id=email_id,
                body=body
            ).execute()

            return f"Successfully applied label '{label}' to email {email_id}"

        except Exception as e:
            return self._handle_error(e, f"applying label to email {email_id}")


class EmailSearchInput(BaseModel):
    """Input Schema for searching emails"""
    query: str = Field(
        description="Gmail search query (e.g., 'from:example@gmail.com', 'is:unread', 'subject:invoice')")
    max_results: int = Field(default=10, description="Maximum number of results to return")


class GmailSearchTool(GmailBaseTool):
    """Tool for search in Gmail"""
    name: str = "search_emails"
    description: str = """Search Gmail using Gmail's search syntax. 
    Returns email IDs in a structured format for use with other tools."""
    args_schema: type[BaseModel] = EmailSearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        try:
            results = self._service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                return "No emails found matching the search criteria."

            # Get basic info for each email to help with selection
            email_info = []
            for i, msg in enumerate(messages[:10]):  # Limit to 10 for testing
                try:
                    # Get basic headers for context
                    message = self._service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='metadata',
                        metadataHeaders=['Subject', 'From']
                    ).execute()

                    headers = message['payload'].get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')

                    email_info.append(
                        f"Email {i + 1}: ID={msg['id']} | From: {sender[:40]}... | Subject: {subject[:40]}...")
                except:
                    email_info.append(f"Email {i + 1}: ID={msg['id']}")

            result = f"Found {len(messages)} emails. Showing first {len(email_info)}:\n"
            result += "\n".join(email_info)
            result += "\n\nUse the ID value (e.g., '18c7f8a5b2d3e4f5') with other tools."

            return result

        except Exception as e:
            return self._handle_error(e, "searching emails")


class EmailGetInput(BaseModel):
    """Input Schema for retrieving email content"""
    email_id: str = Field(description="The Gmail message ID")
    include_body: bool = Field(default=True, description="Include email body content")


class GmailGetEmailTool(GmailBaseTool):
    """Tool to retrieve full email content"""
    name: str = "get_email_content"
    description: str = "Retrieve the full content of an email including sender, subject, body, and metadata"
    args_schema: type[BaseModel] = EmailGetInput

    def _parse_email(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email message into readable format"""
        headers = message['payload'].get('headers', [])

        # Extract headers
        email_data = {
            'id': message['id'],
            'subject': next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject'),
            'from': next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown'),
            'to': next((h['value'] for h in headers if h['name'] == 'To'), 'Unknown'),
            'date': next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown'),
        }

        # Get email body
        body = EmailHelper.parse_email_body(message['payload'])
        email_data['body'] = body

        return email_data

    def _run(self, email_id: str, include_body: bool = True) -> str:
        try:
            message = self._service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()

            email_data = self._parse_email(message, include_body)

            # Format response
            result = f"Email ID: {email_data['id']}\n"
            result += f"From: {email_data['from']}\n"
            result += f"To: {email_data['to']}\n"
            result += f"Subject: {email_data['subject']}\n"
            result += f"Date: {email_data['date']}\n"

            if include_body:
                result += f"\nBody:\n{email_data['body'][:500]}..."  # Truncate for readability

            return result

        except Exception as e:
            return self._handle_error(e, f"retrieving email {email_id}")


def create_gmail_tools(credentials: Credentials, allowed_labels=None) -> List[BaseTool]:
    """Create and return Gmail tools initialized with credentials"""
    toolkit = GmailToolkit(credentials, allowed_labels)
    return toolkit.create_tools()
