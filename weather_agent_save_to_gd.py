# Import relevant functionality
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import os
import pickle
import io
from typing import Optional, List, Dict
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Google Drive OAuth2 setup
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    """Authenticate and return Google Drive service object"""
    creds = None
    
    # Token file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Create the flow using the client secrets file
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',  # Download this from Google Cloud Console
                SCOPES
            )
            
            try:
                # Try to run local server without opening browser
                creds = flow.run_local_server(
                    port=0,  # Use any available port
                    open_browser=False  # Don't try to open browser automatically
                )
            except Exception:
                # Fall back to manual authentication
                print("\n=== Manual Google Drive Authentication ===")
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true'
                )
                
                print(f"\n1. Open this URL in your browser:\n{auth_url}")
                print("\n2. Grant permissions and copy the authorization code")
                code = input("\n3. Enter the authorization code here: ")
                
                # Exchange code for token
                flow.fetch_token(code=code)
                creds = flow.credentials
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

# Initialize Google Drive service
try:
    drive_service = authenticate_google_drive()
    print("Google Drive authentication successful!")
except Exception as e:
    print(f"Google Drive authentication failed: {e}")
    print("Please ensure you have credentials.json from Google Cloud Console")
    drive_service = None

# Custom Google Drive Tools
@tool
def search_google_drive(query: str, file_type: Optional[str] = None) -> str:
    """
    Search for files in Google Drive.
    
    Args:
        query: Search query string
        file_type: Optional file type filter (e.g., 'pdf', 'docx', 'sheet', 'doc')
    
    Returns:
        String with search results
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        # Build query
        drive_query = f"name contains '{query}'"
        
        if file_type:
            mime_types = {
                'pdf': 'application/pdf',
                'doc': 'application/vnd.google-apps.document',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'sheet': 'application/vnd.google-apps.spreadsheet',
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'slide': 'application/vnd.google-apps.presentation',
                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            }
            if file_type.lower() in mime_types:
                drive_query += f" and mimeType='{mime_types[file_type.lower()]}'"
        
        # Execute search
        results = drive_service.files().list(
            q=drive_query,
            pageSize=10,
            fields="files(id, name, mimeType, modifiedTime, size)"
        ).execute()
        
        files = results.get('files', [])
        if not files:
            return "No files found matching your search."
        
        file_list = []
        for f in files:
            file_info = f"- {f['name']} (ID: {f['id']}, Type: {f['mimeType'].split('.')[-1]})"
            file_list.append(file_info)
        
        return f"Found {len(files)} files:\n" + "\n".join(file_list)
    except Exception as e:
        return f"Error searching Google Drive: {str(e)}"

@tool
def read_google_drive_file(file_id: str) -> str:
    """
    Read content from a Google Drive file.
    
    Args:
        file_id: The ID of the file to read
    
    Returns:
        The content of the file as a string
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        # Get file metadata
        file = drive_service.files().get(fileId=file_id).execute()
        mime_type = file.get('mimeType', '')
        file_name = file.get('name', 'Unknown')
        
        # Handle Google Docs files
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            request = drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/csv'
            )
        elif mime_type == 'application/vnd.google-apps.presentation':
            request = drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
        else:
            # For other files, download as is
            request = drive_service.files().get_media(fileId=file_id)
        
        # Download file content
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        file_content.seek(0)
        content = file_content.read()
        
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
            return f"Content of '{file_name}':\n\n{text_content}"
        except:
            return f"'{file_name}' is a binary file (size: {len(content)} bytes)"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def create_google_drive_file(name: str, content: str, file_type: str = 'text') -> str:
    """
    Create a new file in Google Drive.
    
    Args:
        name: Name of the file to create
        content: Content of the file
        file_type: Type of file - 'text', 'google-doc', or 'google-sheet'
    
    Returns:
        Success message with file details
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        # Create file metadata
        file_metadata = {'name': name}
        
        # Handle different file types
        if file_type == 'google-doc':
            file_metadata['mimeType'] = 'application/vnd.google-apps.document'
            media = MediaIoBaseUpload(
                io.BytesIO(content.encode('utf-8')),
                mimetype='text/plain',
                resumable=True
            )
        elif file_type == 'google-sheet':
            file_metadata['mimeType'] = 'application/vnd.google-apps.spreadsheet'
            media = MediaIoBaseUpload(
                io.BytesIO(content.encode('utf-8')),
                mimetype='text/csv',
                resumable=True
            )
        else:
            # Default to text file
            if not name.endswith('.txt'):
                name += '.txt'
                file_metadata['name'] = name
            media = MediaIoBaseUpload(
                io.BytesIO(content.encode('utf-8')),
                mimetype='text/plain',
                resumable=True
            )
        
        # Create file
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        return f"Successfully created file '{file.get('name')}'\nFile ID: {file.get('id')}\nView link: {file.get('webViewLink', 'No link available')}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

# Also update the update_google_drive_file function:
@tool
def update_google_drive_file(file_id: str, new_content: str) -> str:
    """
    Update the content of an existing Google Drive file.
    
    Args:
        file_id: ID of the file to update
        new_content: New content for the file
    
    Returns:
        Success or error message
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        # Get current file metadata
        file = drive_service.files().get(fileId=file_id).execute()
        file_name = file.get('name', 'Unknown')
        
        # Prepare media upload
        media = MediaIoBaseUpload(
            io.BytesIO(new_content.encode('utf-8')),
            mimetype='text/plain',
            resumable=True
        )
        
        # Update file
        updated_file = drive_service.files().update(
            fileId=file_id,
            media_body=media,
            fields='id, name, modifiedTime'
        ).execute()
        
        return f"Successfully updated '{updated_file.get('name')}'\nModified at: {updated_file.get('modifiedTime')}"
    except Exception as e:
        return f"Error updating file: {str(e)}"

@tool
def delete_google_drive_file(file_id: str) -> str:
    """
    Delete a file from Google Drive.
    
    Args:
        file_id: ID of the file to delete
    
    Returns:
        Success or error message
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        # Get file name before deletion
        file = drive_service.files().get(fileId=file_id).execute()
        file_name = file.get('name', 'Unknown')
        
        # Delete file
        drive_service.files().delete(fileId=file_id).execute()
        return f"Successfully deleted '{file_name}'"
    except Exception as e:
        return f"Error deleting file: {str(e)}"

@tool
def list_google_drive_files(max_results: int = 10) -> str:
    """
    List files in Google Drive.
    
    Args:
        max_results: Maximum number of files to return
    
    Returns:
        String with list of files
    """
    if not drive_service:
        return "Google Drive is not authenticated. Please set up credentials.json"
    
    try:
        results = drive_service.files().list(
            pageSize=max_results,
            fields="files(id, name, mimeType, modifiedTime, size)",
            orderBy="modifiedTime desc"
        ).execute()
        
        files = results.get('files', [])
        if not files:
            return "No files found in Google Drive."
        
        file_list = [f"Recent files in Google Drive (showing up to {max_results}):"]
        for f in files:
            size = f.get('size', 'N/A')
            if size != 'N/A':
                size = f"{int(size) / 1024 / 1024:.2f} MB" if int(size) > 1024*1024 else f"{int(size) / 1024:.2f} KB"
            
            modified = f.get('modifiedTime', 'Unknown')
            if modified != 'Unknown':
                modified = modified.split('T')[0]  # Just the date
            
            file_info = f"- {f['name']} (Modified: {modified}, Size: {size}, ID: {f['id']})"
            file_list.append(file_info)
        
        return "\n".join(file_list)
    except Exception as e:
        return f"Error listing files: {str(e)}"

# Create the agent with all tools
memory = MemorySaver()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini")

# Combine all tools
search = TavilySearch(max_results=2)
tools = [
    search,
    search_google_drive,
    read_google_drive_file,
    create_google_drive_file,
    update_google_drive_file,
    delete_google_drive_file,
    list_google_drive_files
]

agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Example usage
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc123"}}
    
    # Initial greeting
    print("=== Greeting ===")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="Hi im Roman! and i live in Poland, Wroclaw")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    # Weather query
    print("\n=== Weather Query ===")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="Whats the weather where I live today?")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    # Get weather and save to Google Drive
    print("\n=== Weather Report to Google Drive ===")
    today = datetime.now().strftime("%Y-%m-%d")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(
            content=f"Get the current weather report for Wroclaw, Poland and save it as a file named 'Wroclaw_Weather_{today}' in my Google Drive"
        )]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    # List files to confirm
    print("\n=== Checking Files ===")
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="List my recent files in Google Drive")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()