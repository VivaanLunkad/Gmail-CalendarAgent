from typing import Dict, List, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import pytz

# Default timezone
DEFAULT_TIMEZONE = "America/New_York"


class CalendarBaseTool(BaseTool):
    """Base class for all Calendar tools"""

    _service: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

    def set_service(self, service: Any):
        """Set the Calendar service instance"""
        self._service = service

    @staticmethod
    def _handle_error(error: Exception, context: str) -> str:
        """Centralized error handling"""
        if isinstance(error, HttpError):
            if hasattr(error, "resp") and hasattr(error.resp, "status"):
                if error.resp.status == 404:
                    return f"Error: Resource not found in {context}"
                elif error.resp.status == 403:
                    return f"Error: Permission denied for {context}. Please check your calendar permissions."
                elif error.resp.status == 400:
                    return f"Error: Invalid request in {context}. Please check the input parameters."
            return f"API Error in {context}: {error}"
        return f"Error in {context}: {str(error)}"


class CalendarHelper:
    """Helper class for calendar operations"""

    @staticmethod
    def parse_datetime(date_str: str, timezone: str = DEFAULT_TIMEZONE) -> Dict[str, str]:
        """Parse datetime string and return a Google Calendar format"""
        try:
            # Try parsing different formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If no format works, try to parse naturally
                raise ValueError(f"Could not parse date: {date_str}")

            # If time wasn't specified, assume it's an all-day event
            if dt.hour == 0 and dt.minute == 0 and ":" not in date_str:
                return {"date": dt.strftime("%Y-%m-%d")}
            else:
                tz = pytz.timezone(timezone)
                dt = tz.localize(dt)
                return {"dateTime": dt.isoformat(), "timeZone": timezone}

        except Exception as e:
            raise ValueError(f"Error parsing datetime '{date_str}': {str(e)}")


class CalendarToolkit:
    """Factory class for creating all Calendar tools"""

    def __init__(self, credentials: Any):
        self.service = build('calendar', 'v3', credentials=credentials)

    def create_tools(self) -> List[BaseTool]:
        """Create all Calendar tools with a shared service"""
        tools = []

        # Create event tool
        create_tool = CreateEventTool()
        create_tool.set_service(self.service)
        tools.append(create_tool)

        # Search events tool
        search_tool = SearchEventsTool()
        search_tool.set_service(self.service)
        tools.append(search_tool)

        # Update event tool
        update_tool = UpdateEventTool()
        update_tool.set_service(self.service)
        tools.append(update_tool)

        # Delete event tool
        delete_tool = DeleteEventTool()
        delete_tool.set_service(self.service)
        tools.append(delete_tool)

        # Get event details tool
        get_tool = GetEventTool()
        get_tool.set_service(self.service)
        tools.append(get_tool)

        return tools


class CreateEventInput(BaseModel):
    """Input schema for creating a calendar event"""
    summary: str = Field(description="Event title/summary")
    start_time: str = Field(description="Start time (e.g., '2025-01-15 14:00' or '2025-01-15' for all-day)")
    end_time: str = Field(description="End time (same format as start_time)")
    description: Optional[str] = Field(default=None, description="Event description")
    location: Optional[str] = Field(default=None, description="Event location")
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee email addresses")
    timezone: Optional[str] = Field(default=DEFAULT_TIMEZONE, description="Timezone for the event")


class CreateEventTool(CalendarBaseTool):
    """Tool for creating calendar events"""

    name: str = "create_calendar_event"
    description: str = """Create a new calendar event. 
    Specify start and end times as 'YYYY-MM-DD HH:MM' for timed events or 'YYYY-MM-DD' for all-day events.
    Returns the event ID and link."""
    args_schema: type[BaseModel] = CreateEventInput

    def _run(self, summary: str, start_time: str, end_time: str,
             description: Optional[str] = None,
             location: Optional[str] = None,
             attendees: Optional[List[str]] = None,
             timezone: Optional[str] = None) -> str:

        if not timezone:
            timezone = DEFAULT_TIMEZONE

        try:
            # Build event body
            event = {
                'summary': summary,
                'start': CalendarHelper.parse_datetime(start_time, timezone),
                'end': CalendarHelper.parse_datetime(end_time, timezone),
            }

            if description:
                event['description'] = description
            if location:
                event['location'] = location
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]

            # Create the event
            result = self._service.events().insert(
                calendarId='primary',
                body=event
            ).execute()

            return f"Successfully created event '{summary}' with ID: {result['id']}. Link: {result.get('htmlLink', 'N/A')}"

        except Exception as e:
            return self._handle_error(e, "creating calendar event")


class SearchEventsInput(BaseModel):
    """Input schema for searching calendar events"""
    query: Optional[str] = Field(default=None, description="Search query for event text")
    time_min: Optional[str] = Field(default=None, description="Start of time range (YYYY-MM-DD)")
    time_max: Optional[str] = Field(default=None, description="End of time range (YYYY-MM-DD)")
    max_results: int = Field(default=10, description="Maximum number of results")


class SearchEventsTool(CalendarBaseTool):
    """Tool for searching calendar events"""

    name: str = "search_calendar_events"
    description: str = """Search for calendar events by text or time range.
    Returns event IDs and basic info for further operations."""
    args_schema: type[BaseModel] = SearchEventsInput

    def _run(self, query: Optional[str] = None,
             time_min: Optional[str] = None,
             time_max: Optional[str] = None,
             max_results: int = 10) -> str:
        try:
            params = {
                'calendarId': 'primary',
                'maxResults': max_results,
                'singleEvents': True,
                'orderBy': 'startTime'
            }

            # Add time constraints
            if time_min:
                if ' ' in time_min:  # Has time component
                    dt = datetime.strptime(time_min, "%Y-%m-%d %H:%M")
                else:
                    dt = datetime.strptime(time_min, "%Y-%m-%d")
                params['timeMin'] = dt.isoformat() + 'Z'
            else:
                params['timeMin'] = datetime.now().isoformat() + 'Z'

            if time_max:
                if ' ' in time_max:  # Has time component
                    dt = datetime.strptime(time_max, "%Y-%m-%d %H:%M")
                else:
                    dt = datetime.strptime(time_max, "%Y-%m-%d") + timedelta(days=1)
                params['timeMax'] = dt.isoformat() + 'Z'

            if query:
                params['q'] = query

            # Execute search
            results = self._service.events().list(**params).execute()
            events = results.get('items', [])

            if not events:
                return "No events found matching the criteria."

            # Format results
            event_info = []
            for i, event in enumerate(events):
                start = event['start'].get('dateTime', event['start'].get('date', 'Unknown'))
                summary = event.get('summary', 'No Title')
                event_info.append(
                    f"Event {i + 1}: ID={event['id']} | {start} | {summary[:50]}..."
                )

            result = f"Found {len(events)} events:\n"
            result += "\n".join(event_info)
            result += "\n\nUse the ID value with other tools to update or delete events."

            return result

        except Exception as e:
            return self._handle_error(e, "searching calendar events")


class UpdateEventInput(BaseModel):
    """Input schema for updating a calendar event"""
    event_id: str = Field(description="The calendar event ID")
    summary: Optional[str] = Field(default=None, description="New event title")
    start_time: Optional[str] = Field(default=None, description="New start time")
    end_time: Optional[str] = Field(default=None, description="New end time")
    description: Optional[str] = Field(default=None, description="New description")
    location: Optional[str] = Field(default=None, description="New location")
    timezone: Optional[str] = Field(default=DEFAULT_TIMEZONE, description="Timezone for the event")


class UpdateEventTool(CalendarBaseTool):
    """Tool for updating calendar events"""

    name: str = "update_calendar_event"
    description: str = """Update an existing calendar event.
    Only provide the fields you want to change."""
    args_schema: type[BaseModel] = UpdateEventInput

    def _run(self, event_id: str,
             summary: Optional[str] = None,
             start_time: Optional[str] = None,
             end_time: Optional[str] = None,
             description: Optional[str] = None,
             location: Optional[str] = None,
             timezone: str = DEFAULT_TIMEZONE) -> str:
        try:
            # Get existing event
            event = self._service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()

            # Update fields if provided
            if summary is not None:
                event['summary'] = summary
            if description is not None:
                event['description'] = description
            if location is not None:
                event['location'] = location
            if start_time is not None:
                event['start'] = CalendarHelper.parse_datetime(start_time, timezone)
            if end_time is not None:
                event['end'] = CalendarHelper.parse_datetime(end_time, timezone)

            # Update the event
            updated = self._service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event
            ).execute()

            return f"Successfully updated event '{updated.get('summary', 'Untitled')}' (ID: {event_id})"

        except Exception as e:
            return self._handle_error(e, f"updating event {event_id}")


class DeleteEventInput(BaseModel):
    """Input schema for deleting a calendar event"""
    event_id: str = Field(description="The calendar event ID to delete")


class DeleteEventTool(CalendarBaseTool):
    """Tool for deleting calendar events"""

    name: str = "delete_calendar_event"
    description: str = "Delete a calendar event by its ID"
    args_schema: type[BaseModel] = DeleteEventInput

    def _run(self, event_id: str) -> str:
        try:
            # Try to get event details first
            try:
                event = self._service.events().get(
                    calendarId='primary',
                    eventId=event_id
                ).execute()
                event_summary = event.get('summary', 'Untitled')
            except:
                event_summary = "Unknown"

            # Delete the event
            self._service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()

            return f"Successfully deleted event '{event_summary}' (ID: {event_id})"

        except Exception as e:
            return self._handle_error(e, f"deleting event {event_id}")


class GetEventInput(BaseModel):
    """Input schema for getting event details"""
    event_id: str = Field(description="The calendar event ID")


class GetEventTool(CalendarBaseTool):
    """Tool to retrieve full event details"""
    name: str = "get_calendar_event"
    description: str = "Get full details of a calendar event"
    args_schema: type[BaseModel] = GetEventInput

    def _run(self, event_id: str) -> str:
        try:
            event = self._service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()

            # Format response
            result = f"Event ID: {event['id']}\n"
            result += f"Title: {event.get('summary', 'No Title')}\n"
            result += f"Start: {event['start'].get('dateTime', event['start'].get('date', 'Unknown'))}\n"
            result += f"End: {event['end'].get('dateTime', event['end'].get('date', 'Unknown'))}\n"

            if 'location' in event:
                result += f"Location: {event['location']}\n"
            if 'description' in event:
                result += f"Description: {event['description']}\n"
            if 'attendees' in event:
                attendees = [att.get('email', 'Unknown') for att in event['attendees']]
                result += f"Attendees: {', '.join(attendees)}\n"

            result += f"Link: {event.get('htmlLink', 'N/A')}"

            return result

        except Exception as e:
            return self._handle_error(e, f"retrieving event {event_id}")


def create_calendar_tools(credentials: Any) -> List[BaseTool]:
    """Create and return Calendar tools initialized with credentials"""
    toolkit = CalendarToolkit(credentials)
    return toolkit.create_tools()