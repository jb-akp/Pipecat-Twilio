"""Bot tool functions for Calendar, Gmail, and WhatsApp.

Provides functions for fetching calendar events, Gmail emails, and sending WhatsApp reminders.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams
from twilio.rest import Client

load_dotenv(override=True)

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/gmail.readonly'
]


def get_google_credentials():
    """Get authenticated Google credentials for Calendar and Gmail APIs.
    
    Returns:
        Credentials: Authenticated Google OAuth2 credentials
    """
    creds = None
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    
    # Load existing token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, request authorization
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google credentials file not found at {credentials_path}. "
                    "Please set GOOGLE_CREDENTIALS_PATH in your .env file or place credentials.json in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return creds


async def get_calendar_events(params: FunctionCallParams):
    """Get calendar events for today.
    
    Args:
        params: FunctionCallParams (no arguments needed)
        
    Returns:
        str: JSON string of events for today
    """
    try:
        # Get today's date in local timezone
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        today = datetime.now(local_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info(f"📅 Fetching calendar events for today: {today.strftime('%Y-%m-%d')}")
        
        # Convert to UTC for Google Calendar API
        start_of_day = today
        end_of_day = today + timedelta(days=1) - timedelta(seconds=1)
        time_min = start_of_day.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        time_max = end_of_day.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Get authenticated calendar service
        creds = get_google_credentials()
        service = build('calendar', 'v3', credentials=creds)
        
        # Fetch events from primary calendar
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=50,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Filter events to only include summary and simplified times
        filtered_events = []
        for event in events:
            summary = event.get('summary', 'No title')
            
            # Extract and format start time
            start = event['start'].get('dateTime', event['start'].get('date'))
            if 'T' in start:
                # Normalize 'Z' to '+00:00' for fromisoformat
                start_normalized = start.replace('Z', '+00:00')
                start_dt = datetime.fromisoformat(start_normalized)
                # Convert to local time and format
                if start_dt.tzinfo:
                    start_dt = start_dt.astimezone()
                start_time = start_dt.strftime("%I:%M %p")
            else:
                start_time = "All day"
            
            # Extract and format end time
            end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date', ''))
            if end and 'T' in end:
                # Normalize 'Z' to '+00:00' for fromisoformat
                end_normalized = end.replace('Z', '+00:00')
                end_dt = datetime.fromisoformat(end_normalized)
                # Convert to local time and format
                if end_dt.tzinfo:
                    end_dt = end_dt.astimezone()
                end_time = end_dt.strftime("%I:%M %p")
            elif end:
                end_time = "All day"
            else:
                end_time = None
            
            filtered_events.append({
                'summary': summary,
                'start_time': start_time,
                'end_time': end_time
            })
        
        result = json.dumps(filtered_events, indent=2)
        
        logger.info(f"✅ Calendar events retrieved: {len(events)} events")
        await params.result_callback(result)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get calendar events: {e}")
        error_result = f"Error retrieving calendar events: {str(e)}"
        await params.result_callback(error_result)
        return error_result


async def get_gmail_emails(params: FunctionCallParams):
    """Get the 3 most recent Gmail emails.
    
    Args:
        params: FunctionCallParams (no arguments needed)
        
    Returns:
        str: JSON string of 3 most recent emails
    """
    try:
        logger.info(f"📧 Fetching 3 most recent Gmail emails")
        
        # Get authenticated Gmail service
        creds = get_google_credentials()
        service = build('gmail', 'v1', credentials=creds)
        
        # Get message IDs (list() only returns IDs, not full emails)
        message_ids = service.users().messages().list(
            userId='me',
            maxResults=3
        ).execute().get('messages', [])
        
        # Extract snippet, subject, and from for each email
        emails_list = []
        for msg in message_ids:
            message = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata'
            ).execute()
            
            # Extract snippet (email summary)
            snippet = message.get('snippet', '')
            
            # Extract subject and from from headers
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown sender')
            
            emails_list.append({
                'snippet': snippet,
                'subject': subject,
                'from': sender
            })
        
        result = json.dumps(emails_list, indent=2)
        
        logger.info(f"✅ Gmail emails retrieved: {len(emails_list)} emails")
        await params.result_callback(result)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get Gmail emails: {e}")
        error_result = f"Error retrieving Gmail emails: {str(e)}"
        await params.result_callback(error_result)
        return error_result


async def send_whatsapp_reminder(params: FunctionCallParams):
    """Send a reminder message via Twilio WhatsApp.
    
    Args:
        params: FunctionCallParams containing the reminder_text in arguments
        
    Returns:
        str: Confirmation message
    """
    try:
        reminder_text = params.arguments.get("reminder_text", "")
        
        # Get Twilio credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Should be: +14155238886 (just the number, no whatsapp: prefix)
        from_number = f"whatsapp:{from_number}"  # Add whatsapp: prefix for Twilio
        
        recipient_number = os.getenv("RECIPIENT_NUMBER")  # Should be: +16507303690 (just the number)
        to_number = f"whatsapp:{recipient_number}"  # Format for Twilio
        
        logger.info(f"📤 Preparing WhatsApp reminder:")
        logger.info(f"   From: {from_number}")
        logger.info(f"   To: {to_number}")
        logger.info(f"   Message: {reminder_text}")
        
        # Send message
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=from_number,
            body=reminder_text,
            to=to_number
        )
        
        logger.info(f"✅ WhatsApp reminder sent successfully. SID: {message.sid}")
        
        result = f"Reminder sent to WhatsApp successfully!"
        await params.result_callback(result)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to send WhatsApp reminder: {e}")
        error_result = f"Error sending WhatsApp reminder: {str(e)}"
        await params.result_callback(error_result)
        return error_result

