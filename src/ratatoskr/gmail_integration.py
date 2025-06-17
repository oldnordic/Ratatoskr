"""
Gmail integration for Ratatoskr AI Assistant.

This module provides Gmail API integration for:
- Calendar event monitoring
- Email checking
- Calendar alerts and notifications
- Daily activity summaries
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    logging.warning("Gmail integration not available. Install google-auth-oauthlib and google-api-python-client")

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/calendar.events.readonly'
]

@dataclass
class CalendarEvent:
    """Calendar event data structure."""
    id: str
    summary: str
    description: str = ""
    start_time: datetime = None
    end_time: datetime = None
    location: str = ""
    attendees: List[str] = None
    
    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []

@dataclass
class EmailMessage:
    """Email message data structure."""
    id: str
    subject: str
    sender: str
    snippet: str
    date: datetime = None
    is_read: bool = False

class GmailService:
    """Gmail and Calendar service integration."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.credentials = None
        self.gmail_service = None
        self.calendar_service = None
        self.is_authenticated = False
        self.alert_thread = None
        self.stop_alerts = False
        
    def authenticate(self) -> bool:
        """Authenticate with Gmail API."""
        if not GMAIL_AVAILABLE:
            logging.error("Gmail API libraries not available")
            return False
            
        gmail_config = self.config_manager.get_gmail_config()
        if not gmail_config.enabled or not gmail_config.email:
            logging.warning("Gmail integration not enabled or email not configured")
            return False
            
        try:
            # Check for existing credentials
            creds = None
            token_path = 'token.json'
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
            # If no valid credentials, authenticate
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # Use app password for service account
                    if gmail_config.app_password:
                        # For app password authentication
                        creds = self._authenticate_with_app_password(gmail_config)
                    else:
                        # For OAuth flow
                        creds = self._authenticate_with_oauth()
                
                # Save credentials
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.credentials = creds
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            self.is_authenticated = True
            logging.info("Gmail authentication successful")
            return True
            
        except Exception as e:
            logging.error(f"Gmail authentication failed: {e}")
            return False
    
    def _authenticate_with_app_password(self, gmail_config) -> Credentials:
        """Authenticate using Gmail app password."""
        # This is a simplified approach - in production you'd want proper OAuth
        logging.info("Using app password authentication")
        # For now, we'll use OAuth flow as app password requires different setup
        return self._authenticate_with_oauth()
    
    def _authenticate_with_oauth(self) -> Credentials:
        """Authenticate using OAuth flow."""
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        return creds
    
    def get_today_events(self) -> List[CalendarEvent]:
        """Get today's calendar events."""
        if not self.is_authenticated:
            return []
            
        try:
            now = datetime.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=start_of_day.isoformat() + 'Z',
                timeMax=end_of_day.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            calendar_events = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse datetime
                if 'T' in start:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
                else:
                    start_time = datetime.fromisoformat(start)
                    end_time = datetime.fromisoformat(end)
                
                calendar_events.append(CalendarEvent(
                    id=event['id'],
                    summary=event.get('summary', 'No title'),
                    description=event.get('description', ''),
                    start_time=start_time,
                    end_time=end_time,
                    location=event.get('location', ''),
                    attendees=[attendee.get('email', '') for attendee in event.get('attendees', [])]
                ))
            
            return calendar_events
            
        except Exception as e:
            logging.error(f"Error fetching calendar events: {e}")
            return []
    
    def get_upcoming_events(self, minutes: int = 60) -> List[CalendarEvent]:
        """Get events starting within the next N minutes."""
        if not self.is_authenticated:
            return []
            
        try:
            now = datetime.utcnow()
            end_time = now + timedelta(minutes=minutes)
            
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=end_time.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            calendar_events = []
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Parse datetime
                if 'T' in start:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
                else:
                    start_time = datetime.fromisoformat(start)
                    end_time = datetime.fromisoformat(end)
                
                calendar_events.append(CalendarEvent(
                    id=event['id'],
                    summary=event.get('summary', 'No title'),
                    description=event.get('description', ''),
                    start_time=start_time,
                    end_time=end_time,
                    location=event.get('location', ''),
                    attendees=[attendee.get('email', '') for attendee in event.get('attendees', [])]
                ))
            
            return calendar_events
            
        except Exception as e:
            logging.error(f"Error fetching upcoming events: {e}")
            return []
    
    def get_recent_emails(self, max_results: int = 10) -> List[EmailMessage]:
        """Get recent emails."""
        if not self.is_authenticated:
            return []
            
        try:
            results = self.gmail_service.users().messages().list(
                userId='me', maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            email_messages = []
            
            for message in messages:
                msg = self.gmail_service.users().messages().get(
                    userId='me', id=message['id']
                ).execute()
                
                headers = msg['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                
                # Parse date
                try:
                    date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
                except:
                    date = datetime.now()
                
                email_messages.append(EmailMessage(
                    id=message['id'],
                    subject=subject,
                    sender=sender,
                    snippet=msg.get('snippet', ''),
                    date=date,
                    is_read='UNREAD' not in msg.get('labelIds', [])
                ))
            
            return email_messages
            
        except Exception as e:
            logging.error(f"Error fetching emails: {e}")
            return []
    
    def start_alert_monitoring(self, callback_function):
        """Start monitoring calendar events for alerts."""
        if not self.is_authenticated:
            logging.warning("Cannot start alert monitoring - not authenticated")
            return
            
        self.stop_alerts = False
        self.alert_thread = threading.Thread(
            target=self._alert_monitor_loop,
            args=(callback_function,),
            daemon=True
        )
        self.alert_thread.start()
        logging.info("Calendar alert monitoring started")
    
    def stop_alert_monitoring(self):
        """Stop calendar alert monitoring."""
        self.stop_alerts = True
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        logging.info("Calendar alert monitoring stopped")
    
    def _alert_monitor_loop(self, callback_function):
        """Main loop for monitoring calendar alerts."""
        gmail_config = self.config_manager.get_gmail_config()
        alert_times = gmail_config.alert_times
        
        while not self.stop_alerts:
            try:
                # Get upcoming events
                upcoming_events = self.get_upcoming_events(minutes=30)
                
                for event in upcoming_events:
                    time_until_event = event.start_time - datetime.utcnow()
                    minutes_until = int(time_until_event.total_seconds() / 60)
                    
                    # Check if we should alert
                    for alert_time in alert_times:
                        if minutes_until == alert_time:
                            alert_message = f"Calendar Alert: '{event.summary}' starts in {alert_time} minutes"
                            if event.location:
                                alert_message += f" at {event.location}"
                            callback_function(alert_message)
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Error in alert monitoring: {e}")
                time.sleep(60)
    
    def get_daily_summary(self) -> str:
        """Generate daily activity summary."""
        if not self.is_authenticated:
            return "Gmail integration not available"
            
        try:
            today_events = self.get_today_events()
            recent_emails = self.get_recent_emails(max_results=5)
            
            summary = "📅 Daily Summary\n\n"
            
            # Calendar events
            if today_events:
                summary += "🗓️ Today's Events:\n"
                for event in today_events:
                    time_str = event.start_time.strftime("%H:%M")
                    summary += f"  • {time_str} - {event.summary}\n"
            else:
                summary += "🗓️ No events scheduled for today\n"
            
            summary += "\n📧 Recent Emails:\n"
            for email in recent_emails:
                time_str = email.date.strftime("%H:%M")
                status = "📬" if not email.is_read else "📭"
                summary += f"  {status} {time_str} - {email.sender}: {email.subject}\n"
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating daily summary: {e}")
            return f"Error generating summary: {e}"

# Global Gmail service instance
gmail_service = None 