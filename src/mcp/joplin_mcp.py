"""Joplin MCP Server implementation."""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.joplin.joplin_api import JoplinAPI, JoplinNote, OrderDirection
from src.joplin.joplin_utils import get_token_from_env, MarkdownContent

# Initialize FastMCP server
mcp = FastMCP("joplin")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Joplin API client
try:
    api = JoplinAPI(token=get_token_from_env())
    logger.info("Successfully initialized Joplin API client")
except Exception as e:
    logger.error(f"Failed to initialize Joplin API client: {e}")
    api = None

# Input Models
class SearchNotesInput(BaseModel):
    """Input parameters for searching notes."""
    query: str
    limit: Optional[int] = 100

class CreateNoteInput(BaseModel):
    """Input parameters for creating a note."""
    title: str
    body: Optional[str] = None
    parent_id: Optional[str] = None
    is_todo: Optional[bool] = False

class UpdateNoteInput(BaseModel):
    """Input parameters for updating a note."""
    note_id: str
    title: Optional[str] = None
    body: Optional[str] = None
    parent_id: Optional[str] = None
    is_todo: Optional[bool] = None

class ImportMarkdownInput(BaseModel):
    """Input parameters for importing markdown files."""
    file_path: str

# MCP Tools
@mcp.tool()
async def search_notes(args: SearchNotesInput) -> Dict[str, Any]:
    """Search for notes in Joplin.
    
    Args:
        args: Search parameters
            query: Search query string
            limit: Maximum number of results (default: 100)
    
    Returns:
        Dictionary containing search results
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        results = api.search_notes(query=args.query, limit=args.limit)
        return {
            "status": "success",
            "total": len(results.items),
            "has_more": results.has_more,
            "notes": [
                {
                    "id": note.id,
                    "title": note.title,
                    "body": note.body,
                    "created_time": note.created_time.isoformat() if note.created_time else None,
                    "updated_time": note.updated_time.isoformat() if note.updated_time else None,
                    "is_todo": note.is_todo
                }
                for note in results.items
            ]
        }
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_note(note_id: str) -> Dict[str, Any]:
    """Get a specific note by ID.
    
    Args:
        note_id: ID of the note to retrieve
    
    Returns:
        Dictionary containing the note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        note = api.get_note(note_id)
        return {
            "status": "success",
            "note": {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "created_time": note.created_time.isoformat() if note.created_time else None,
                "updated_time": note.updated_time.isoformat() if note.updated_time else None,
                "is_todo": note.is_todo
            }
        }
    except Exception as e:
        logger.error(f"Error getting note: {e}")
        return {"error": str(e)}

@mcp.tool()
async def create_note(args: CreateNoteInput) -> Dict[str, Any]:
    """Create a new note in Joplin.
    
    Args:
        args: Note creation parameters
            title: Note title
            body: Note content in Markdown (optional)
            parent_id: ID of parent folder (optional)
            is_todo: Whether this is a todo item (optional)
    
    Returns:
        Dictionary containing the created note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        note = api.create_note(
            title=args.title,
            body=args.body,
            parent_id=args.parent_id,
            is_todo=args.is_todo
        )
        return {
            "status": "success",
            "note": {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "created_time": note.created_time.isoformat() if note.created_time else None,
                "updated_time": note.updated_time.isoformat() if note.updated_time else None,
                "is_todo": note.is_todo
            }
        }
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        return {"error": str(e)}

@mcp.tool()
async def update_note(args: UpdateNoteInput) -> Dict[str, Any]:
    """Update an existing note in Joplin.
    
    Args:
        args: Note update parameters
            note_id: ID of note to update
            title: New title (optional)
            body: New content (optional)
            parent_id: New parent folder ID (optional)
            is_todo: New todo status (optional)
    
    Returns:
        Dictionary containing the updated note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        note = api.update_note(
            note_id=args.note_id,
            title=args.title,
            body=args.body,
            parent_id=args.parent_id,
            is_todo=args.is_todo
        )
        return {
            "status": "success",
            "note": {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "created_time": note.created_time.isoformat() if note.created_time else None,
                "updated_time": note.updated_time.isoformat() if note.updated_time else None,
                "is_todo": note.is_todo
            }
        }
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        return {"error": str(e)}

@mcp.tool()
async def delete_note(note_id: str, permanent: bool = False) -> Dict[str, Any]:
    """Delete a note from Joplin.
    
    Args:
        note_id: ID of note to delete
        permanent: If True, permanently delete the note
    
    Returns:
        Dictionary containing the operation status
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        api.delete_note(note_id, permanent=permanent)
        return {
            "status": "success",
            "message": f"Note {note_id} {'permanently ' if permanent else ''}deleted"
        }
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        return {"error": str(e)}

@mcp.tool()
async def import_markdown(args: ImportMarkdownInput) -> Dict[str, Any]:
    """Import a markdown file as a new note.
    
    Args:
        args: Import parameters
            file_path: Path to the markdown file
    
    Returns:
        Dictionary containing the created note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    
    try:
        file_path = Path(args.file_path)
        md_content = MarkdownContent.from_file(file_path)
        
        note = api.create_note(
            title=md_content.title,
            body=md_content.content
        )
        
        return {
            "status": "success",
            "note": {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "created_time": note.created_time.isoformat() if note.created_time else None,
                "updated_time": note.updated_time.isoformat() if note.updated_time else None,
                "is_todo": note.is_todo
            },
            "imported_from": str(file_path)
        }
    except Exception as e:
        logger.error(f"Error importing markdown: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logging.info("Starting Joplin MCP Server...")
    mcp.run(transport='stdio')
