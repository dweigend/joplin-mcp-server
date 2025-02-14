"""Joplin API client implementation."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import requests

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar("T")

class OrderDirection(Enum):
    """Sort direction for API queries."""
    ASC = "ASC"
    DESC = "DESC"

@dataclass
class PaginatedResponse(Generic[T]):
    """Generic paginated response from the Joplin API.

    As per API documentation, all endpoints that return multiple results
    are paginated and return this structure.

    Reference: https://joplinapp.org/help/api/references/rest_api/#pagination
    """
    items: list[T]
    has_more: bool

@dataclass
class JoplinNote:
    """Represents a Joplin note with its attributes.

    Reference: https://joplinapp.org/help/api/references/rest_api/#notes

    Attributes:
        id: Unique identifier
        title: Note title
        body: Note content in Markdown format
        created_time: Creation timestamp
        updated_time: Last update timestamp
        is_conflict: Whether this note is in conflict
        latitude: Geographic latitude
        longitude: Geographic longitude
        altitude: Geographic altitude
        author: Note author
        source_url: Source URL
        is_todo: Whether this is a todo item
        todo_due: Todo due date
        todo_completed: Todo completion date
        source: Note source
        source_application: Source application
        application_data: Application-specific data
        order: Sort order
        user_created_time: User creation timestamp
        user_updated_time: User update timestamp
        encryption_cipher_text: Encrypted content
        encryption_applied: Whether encryption is applied
        markup_language: Markup language used
        is_shared: Whether note is shared
        share_id: Share identifier
        conflict_original_id: Original note ID if in conflict
        master_key_id: Master key identifier
        parent_id: Parent folder ID
    """

    id: str
    title: str
    body: str | None = None
    created_time: datetime | None = None
    updated_time: datetime | None = None
    is_conflict: bool = False
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    author: str | None = None
    source_url: str | None = None
    is_todo: bool = False
    todo_due: datetime | None = None
    todo_completed: datetime | None = None
    source: str | None = None
    source_application: str | None = None
    application_data: dict[str, Any] | None = None
    order: int | None = None
    user_created_time: datetime | None = None
    user_updated_time: datetime | None = None
    encryption_cipher_text: str | None = None
    encryption_applied: bool = False
    markup_language: int | None = None
    is_shared: bool = False
    share_id: str | None = None
    conflict_original_id: str | None = None
    master_key_id: str | None = None
    parent_id: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "JoplinNote":
        """Create a JoplinNote instance from API response data.

        Args:
            data: Raw API response dictionary

        Returns:
            JoplinNote instance

        Raises:
            ValueError: If required fields are missing
        """
        try:
            logger.debug(f"Creating JoplinNote from API data: {data}")

            # Ensure essential fields exist
            if "id" not in data or "title" not in data:
                msg = f"Missing essential fields (id/title) in API response: {data}"
                raise ValueError(msg)

            # Convert timestamps to datetime objects
            created_time = (
                datetime.fromtimestamp(data["created_time"] / 1000)
                if data.get("created_time")
                else None
            )
            updated_time = (
                datetime.fromtimestamp(data["updated_time"] / 1000)
                if data.get("updated_time")
                else None
            )
            user_created_time = (
                datetime.fromtimestamp(data["user_created_time"] / 1000)
                if data.get("user_created_time")
                else None
            )
            user_updated_time = (
                datetime.fromtimestamp(data["user_updated_time"] / 1000)
                if data.get("user_updated_time")
                else None
            )
            todo_due = (
                datetime.fromtimestamp(data["todo_due"] / 1000)
                if data.get("todo_due")
                else None
            )
            todo_completed = (
                datetime.fromtimestamp(data["todo_completed"] / 1000)
                if data.get("todo_completed")
                else None
            )

            return cls(
                id=data["id"],
                title=data["title"],
                body=data.get("body"),
                created_time=created_time,
                updated_time=updated_time,
                is_conflict=data.get("is_conflict", False),
                latitude=data.get("latitude"),
                longitude=data.get("longitude"),
                altitude=data.get("altitude"),
                author=data.get("author"),
                source_url=data.get("source_url"),
                is_todo=data.get("is_todo", False),
                todo_due=todo_due,
                todo_completed=todo_completed,
                source=data.get("source")
            )

        except Exception as e:
            logger.error(f"Error creating JoplinNote: {e}")
            raise

class JoplinAPI:
    """Client for the Joplin REST API.

    This class provides methods to interact with the Joplin API endpoints.
    All methods require a valid API token for authentication.

    Reference: https://joplinapp.org/help/api/references/rest_api/
    """

    def __init__(self, token: str, base_url: str = "http://localhost:41184"):
        """Initialize the API client.

        Args:
            token: API token for authentication
            base_url: Base URL for the Joplin API
        """
        self.token = token
        self.base_url = base_url.rstrip("/")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make an HTTP request to the Joplin API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            params: Query parameters
            json: JSON body data

        Returns:
            Response data as dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": self.token}

        # Add token to params if needed
        if params is None:
            params = {}
        if "token" not in params:
            params["token"] = self.token

        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_notes(
        self,
        page: int = 1,
        limit: int = 100,
        fields: list[str] | None = None,
        order_by: str = "updated_time",
        order_dir: OrderDirection = OrderDirection.DESC
    ) -> PaginatedResponse[JoplinNote]:
        """Get a list of notes.

        Args:
            page: Page number for pagination
            limit: Maximum number of items per page
            fields: List of fields to include in response
            order_by: Field to sort by
            order_dir: Sort direction

        Returns:
            PaginatedResponse containing list of JoplinNote objects
        """
        params = {
            "page": page,
            "limit": limit,
            "order_by": order_by,
            "order_dir": order_dir.value
        }

        if fields:
            params["fields"] = ",".join(fields)

        response = self._make_request("GET", "notes", params=params)

        return PaginatedResponse(
            items=[JoplinNote.from_api_response(item) for item in response["items"]],
            has_more=response["has_more"]
        )

    def get_note(self, note_id: str) -> JoplinNote:
        """Get a specific note by ID.

        Args:
            note_id: ID of the note to retrieve

        Returns:
            JoplinNote object

        Raises:
            requests.exceptions.RequestException: If note not found or other error
        """
        # Explizit alle wichtigen Felder anfordern, insbesondere den Body
        params = {
            "fields": "id,title,body,created_time,updated_time,is_todo"
        }
        response = self._make_request("GET", f"notes/{note_id}", params=params)
        return JoplinNote.from_api_response(response)

    def create_note(
        self,
        title: str,
        body: str | None = None,
        parent_id: str | None = None,
        is_todo: bool = False
    ) -> JoplinNote:
        """Create a new note.

        Args:
            title: Note title
            body: Note content in Markdown
            parent_id: ID of parent folder
            is_todo: Whether this is a todo item

        Returns:
            Created JoplinNote object
        """
        data = {
            "title": title,
            "is_todo": is_todo
        }

        if body is not None:
            data["body"] = body

        if parent_id is not None:
            data["parent_id"] = parent_id

        response = self._make_request("POST", "notes", json=data)
        return JoplinNote.from_api_response(response)

    def update_note(
        self,
        note_id: str,
        title: str | None = None,
        body: str | None = None,
        parent_id: str | None = None,
        is_todo: bool | None = None
    ) -> JoplinNote:
        """Update an existing note.

        Args:
            note_id: ID of note to update
            title: New title
            body: New content
            parent_id: New parent folder ID
            is_todo: New todo status

        Returns:
            Updated JoplinNote object
        """
        data = {}

        if title is not None:
            data["title"] = title

        if body is not None:
            data["body"] = body

        if parent_id is not None:
            data["parent_id"] = parent_id

        if is_todo is not None:
            data["is_todo"] = is_todo

        response = self._make_request("PUT", f"notes/{note_id}", json=data)
        return JoplinNote.from_api_response(response)

    def delete_note(self, note_id: str, permanent: bool = False) -> None:
        """Delete a note.

        Args:
            note_id: ID of note to delete
            permanent: If True, permanently delete the note
        """
        endpoint = f"notes/{note_id}"
        if permanent:
            endpoint += "?permanent=1"
        self._make_request("DELETE", endpoint)

    def search_notes(
        self,
        query: str,
        limit: int = 100
    ) -> PaginatedResponse[JoplinNote]:
        """Search for notes.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            PaginatedResponse containing matching JoplinNote objects
        """
        params = {
            "query": query,
            "limit": limit
        }
        response = self._make_request("GET", "search", params=params)
        return PaginatedResponse(
            items=[JoplinNote.from_api_response(item) for item in response["items"]],
            has_more=response["has_more"]
        )
