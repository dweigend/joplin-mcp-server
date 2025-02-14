"""
Utility functions for the Joplin API client.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class JoplinConfigError(Exception):
    """Raised when there is an error with the Joplin configuration."""

    def __init__(self, message: str, env_var: str | None = None):
        self.env_var = env_var
        super().__init__(message)

@dataclass
class MarkdownContent:
    """Represents parsed content from a Markdown file.
    
    Attributes:
        title: The title extracted from the first heading or filename
        content: The main content of the file
        source_path: Original path to the markdown file
        created_time: When the file was created
        modified_time: When the file was last modified
    """
    title: str
    content: str
    source_path: Path
    created_time: datetime
    modified_time: datetime

    @classmethod
    def from_file(cls, file_path: Path) -> 'MarkdownContent':
        """Create MarkdownContent from a file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            MarkdownContent instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        content = file_path.read_text(encoding='utf-8')
        if not content.strip():
            raise ValueError(f"File is empty: {file_path}")

        # Extract title from first heading or use filename
        lines = content.splitlines()
        title = file_path.stem

        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                content = '\n'.join(lines[1:]).strip()
                break

        return cls(
            title=title,
            content=content,
            source_path=file_path,
            created_time=datetime.fromtimestamp(file_path.stat().st_ctime),
            modified_time=datetime.fromtimestamp(file_path.stat().st_mtime)
        )

def get_token_from_env(env_var: str = "JOPLIN_TOKEN") -> str:
    """Read the Joplin API token from environment variable.
    
    First tries to load from .env file, then from environment.
    
    Args:
        env_var: Name of the environment variable containing the token
        
    Returns:
        The API token
        
    Raises:
        JoplinConfigError: If the token is not set or invalid
    """
    # Try to load from .env file
    load_dotenv()

    token = os.environ.get(env_var, "").strip()
    if not token:
        raise JoplinConfigError(
            f"No {env_var} found in environment variables or .env file",
            env_var=env_var
        )

    if len(token) < 32:
        raise JoplinConfigError(
            f"Invalid {env_var}: Token seems too short (< 32 chars)",
            env_var=env_var
        )

    return token

def format_timestamp(ts: datetime | None) -> str:
    """Format a timestamp for display.
    
    Args:
        ts: Timestamp to format
        
    Returns:
        Formatted string or 'N/A' if timestamp is None
    """
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"

def format_note_info(
    note_id: str,
    title: str,
    created_time: datetime | None = None,
    updated_time: datetime | None = None,
    is_todo: bool = False
) -> str:
    """Format note information for display.
    
    Args:
        note_id: The note's ID
        title: The note's title
        created_time: Creation timestamp
        updated_time: Last update timestamp
        is_todo: Whether this is a todo item
        
    Returns:
        Formatted string with note information
    """
    info = [
        f"ID: {note_id}",
        f"Title: {title}",
        f"Created: {format_timestamp(created_time)}",
    ]

    if updated_time:
        info.append(f"Updated: {format_timestamp(updated_time)}")

    if is_todo:
        info.append("Type: Todo")

    return "\n".join(info)

def read_markdown_file(file_path: str | Path) -> tuple[str, str]:
    """Read a markdown file and extract title and content.
    
    This is a convenience wrapper around MarkdownContent.from_file()
    that returns just the title and content.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Tuple of (title, content)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    content = MarkdownContent.from_file(file_path)
    return content.title, content.content

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename that is safe to use on all platforms
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    sanitized = filename.strip()

    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')

    # Ensure we still have a valid filename
    if not sanitized:
        sanitized = "untitled"

    return sanitized
