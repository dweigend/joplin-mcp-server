"""Joplin MCP Server implementation."""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.joplin.joplin_api import JoplinAPI, JoplinNotebook, JoplinNote, OrderDirection
from src.joplin.joplin_embeddings import (
    EmbeddingUnavailable,
    build_index,
    fuse,
    get_embedder,
    get_index,
    save_index,
    semantic_search,
    set_index,
    similar_notes,
)
from src.joplin.joplin_links import find_neighbours, get_link_graph, get_notebook_paths
from src.joplin.joplin_utils import (
    build_snippet, get_token_from_env, get_base_url_from_env, MarkdownContent,
)

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
    api = JoplinAPI(token=get_token_from_env(), base_url=get_base_url_from_env())
    logger.info(f"Successfully initialized Joplin API client (base_url={get_base_url_from_env()})")
except Exception as e:
    logger.error(f"Failed to initialize Joplin API client: {e}")
    api = None

# Tool parameters are declared as explicit primitive arguments (not a single
# pydantic-model argument). FastMCP in mcp>=1.x exposes a model-typed argument
# as a nested `{"args": {...}}` object, which broke MCP clients that pass the
# fields at the top level. Explicit args keep the tool schema flat and stable
# across mcp versions, so this works on both the host and in the devcontainer.


def serialize_notebook(notebook: JoplinNotebook) -> Dict[str, Any]:
    """Serialize a notebook tree into an MCP-friendly structure."""
    return {
        "id": notebook.id,
        "title": notebook.title,
        "parent_id": notebook.parent_id,
        "created_time": notebook.created_time.isoformat() if notebook.created_time else None,
        "updated_time": notebook.updated_time.isoformat() if notebook.updated_time else None,
        "user_created_time": notebook.user_created_time.isoformat() if notebook.user_created_time else None,
        "user_updated_time": notebook.user_updated_time.isoformat() if notebook.user_updated_time else None,
        "is_shared": notebook.is_shared,
        "share_id": notebook.share_id,
        "children": [serialize_notebook(child) for child in notebook.children or []],
    }


def iter_notebooks_with_paths(
    notebooks: List[JoplinNotebook],
    prefix: Optional[str] = None
) -> List[tuple[JoplinNotebook, str]]:
    """Flatten a notebook tree into notebook/path pairs."""
    flattened = []

    for notebook in notebooks:
        path = f"{prefix}/{notebook.title}" if prefix else notebook.title
        flattened.append((notebook, path))
        flattened.extend(iter_notebooks_with_paths(notebook.children or [], path))

    return flattened


def resolve_notebook_id(parent_id: Optional[str], notebook_name: Optional[str]) -> Optional[str]:
    """Resolve a destination notebook by ID or by exact title/path."""
    if parent_id and notebook_name:
        raise ValueError("Provide either parent_id or notebook_name, not both")

    if parent_id:
        return parent_id

    if not notebook_name:
        return None

    notebooks = api.list_notebooks()
    flattened = iter_notebooks_with_paths(notebooks)

    path_matches = [(notebook, path) for notebook, path in flattened if path == notebook_name]
    if len(path_matches) == 1:
        return path_matches[0][0].id
    if len(path_matches) > 1:
        matches = ", ".join(path for _, path in path_matches)
        raise ValueError(f"Notebook path '{notebook_name}' is ambiguous: {matches}")

    title_matches = [(notebook, path) for notebook, path in flattened if notebook.title == notebook_name]
    if len(title_matches) == 1:
        return title_matches[0][0].id
    if len(title_matches) > 1:
        matches = ", ".join(path for _, path in title_matches)
        raise ValueError(
            f"Notebook name '{notebook_name}' matches multiple notebooks. Use a full path instead: {matches}"
        )

    raise ValueError(f"Notebook '{notebook_name}' was not found")

# MCP Tools
SEARCH_FIELDS = ["id", "title", "body", "parent_id", "updated_time", "is_todo"]
SEARCH_SNIPPET_CHARS = 200
SEARCH_MODES = ("hybrid", "keyword", "semantic")
# Upper bounds on caller-supplied sizing, so one request can't ask for an
# unbounded result set or excerpt.
MAX_SEARCH_LIMIT = 200
MAX_SNIPPET_CHARS = 2000
# Fusion needs more candidates per side than it returns, or a note ranked well
# by one side but just outside the other's cut-off can never be promoted.
CANDIDATE_MULTIPLIER = 3


def _keyword_notes(query: str, limit: int):
    return api.search_notes(query=query, limit=limit, fields=SEARCH_FIELDS)


def _shape_hit(
    note_id: str,
    title: str,
    snippet: str,
    parent_id: str,
    updated: Optional[str],
    matched: Optional[str],
    score: Optional[float],
    is_todo: bool,
    notebook_paths: Dict[str, str],
) -> Dict[str, Any]:
    """One result, with empty fields left out rather than sent as nulls."""
    entry: Dict[str, Any] = {"id": note_id, "title": title}
    if snippet:
        entry["snippet"] = snippet
    notebook = notebook_paths.get(parent_id or "")
    if notebook:
        entry["notebook"] = notebook
    if updated:
        entry["updated_time"] = updated
    if matched:
        entry["matched"] = matched
    if score:
        entry["score"] = score
    if is_todo:
        entry["is_todo"] = True
    return entry


@mcp.tool()
async def search_notes(
    query: str,
    limit: int = 20,
    mode: str = "hybrid",
    keyword_weight: float = 0.3,
    snippet_chars: int = SEARCH_SNIPPET_CHARS,
) -> Dict[str, Any]:
    """Search for notes in Joplin by keyword, meaning, or both.

    Hybrid is the default and is usually what you want: keyword search alone
    misses notes that make the same point in different words, and meaning-based
    search alone is poor at exact strings such as ticket keys or identifiers,
    which embeddings barely represent. Blending covers both failure modes.

    Semantic modes need an index; call build_semantic_index first. Without one
    this falls back to keyword search and says so in the response.

    Returns a short excerpt of each match, normally enough to judge relevance.
    Call get_note for the full text of the ones worth reading.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 20)
        mode: "hybrid", "keyword" or "semantic" (default: "hybrid")
        keyword_weight: In hybrid mode, how much the keyword ranking counts,
            0 to 1. Raise it for identifier-style queries (default: 0.3)
        snippet_chars: Length of each excerpt; 0 omits excerpts (default: 200)

    Returns:
        Dictionary containing search results
    """
    if not api:
        return {"error": "Joplin API client not initialized"}
    if mode not in SEARCH_MODES:
        return {"error": f"mode must be one of {', '.join(SEARCH_MODES)}"}

    # Clamp caller-supplied knobs. limit feeds limit*CANDIDATE_MULTIPLIER into
    # the embedding search, and keyword_weight>1 makes (1-weight) negative in
    # the fusion, so an out-of-range value produces garbage ranking or a heavy
    # query rather than an error.
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    keyword_weight = max(0.0, min(float(keyword_weight), 1.0))
    snippet_chars = max(0, min(int(snippet_chars), MAX_SNIPPET_CHARS))

    # Keyword search hits the network and semantic search embeds the query and
    # multiplies the index matrix; both block. Run off the event loop.
    return await anyio.to_thread.run_sync(
        _search_notes_sync, query, limit, mode, keyword_weight, snippet_chars
    )


def _search_notes_sync(
    query: str, limit: int, mode: str, keyword_weight: float, snippet_chars: int
) -> Dict[str, Any]:
    try:
        notebook_paths = get_notebook_paths(api)
        notes: list[Dict[str, Any]] = []
        note_of_id: Dict[str, JoplinNote] = {}
        fell_back = None

        index = get_index() if mode in ("hybrid", "semantic") else None
        if mode in ("hybrid", "semantic") and (index is None or index.chunk_count == 0):
            fell_back = "No semantic index yet; ran keyword search. Call build_semantic_index."
            mode = "keyword"

        has_more = False

        if mode == "keyword":
            response_page = _keyword_notes(query, limit)
            has_more = response_page.has_more
            for note in response_page.items:
                notes.append(_shape_hit(
                    note.id, note.title,
                    build_snippet(note.body, query, snippet_chars) if snippet_chars else "",
                    note.parent_id or "",
                    note.updated_time.isoformat() if note.updated_time else None,
                    None, None, note.is_todo, notebook_paths,
                ))
        else:
            vector = get_embedder().encode([query])[0]
            candidates = limit * CANDIDATE_MULTIPLIER if mode == "hybrid" else limit
            hits = semantic_search(index, vector, candidates)

            if mode == "hybrid" and keyword_weight > 0:
                keyword_notes = _keyword_notes(query, candidates).items
                note_of_id = {note.id: note for note in keyword_notes}
                pool = len({h["id"] for h in hits} | {n.id for n in keyword_notes})
                hits = fuse(hits, [n.id for n in keyword_notes], keyword_weight, limit)
                has_more = pool > limit
            else:
                has_more = len(hits) > limit
                hits = hits[:limit]
                for hit in hits:
                    hit["matched"] = "semantic"

            for hit in hits:
                note = note_of_id.get(hit["id"])
                # Semantic hits describe themselves with the chunk that matched;
                # keyword-only hits have no chunk, so excerpt their body.
                snippet = hit.get("chunk") or (
                    build_snippet(note.body if note else "", query, snippet_chars)
                )
                if snippet_chars and len(snippet) > snippet_chars:
                    snippet = f"{snippet[:snippet_chars].rstrip()}…"
                notes.append(_shape_hit(
                    hit["id"],
                    hit["title"] or (note.title if note else ""),
                    snippet if snippet_chars else "",
                    hit.get("parent_id") or (note.parent_id if note else "") or "",
                    _iso_stamp(hit.get("updated")) or (
                        note.updated_time.isoformat() if note and note.updated_time else None
                    ),
                    hit.get("matched"),
                    hit.get("score") or None,
                    bool(note.is_todo) if note else False,
                    notebook_paths,
                ))

        response = {
            "status": "success",
            "mode": mode,
            "total": len(notes),
            "has_more": has_more,
            "notes": notes,
        }
        if fell_back:
            response["notice"] = fell_back
        return response
    except EmbeddingUnavailable as e:
        return {"error": f"Semantic search unavailable: {e}"}
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        return {"error": str(e)}


def _iso_stamp(epoch_seconds: Optional[int]) -> Optional[str]:
    if not epoch_seconds:
        return None
    return datetime.fromtimestamp(epoch_seconds).isoformat()


@mcp.tool()
async def build_semantic_index(rebuild: bool = False) -> Dict[str, Any]:
    """Build or update the local semantic search index.

    Embeds every note on this machine using a bundled model; nothing is sent
    anywhere. Notes whose timestamp has not moved keep their existing vectors,
    so a second call after a few edits is quick where the first is not.

    Args:
        rebuild: Re-embed everything instead of reusing unchanged notes

    Returns:
        Dictionary describing the resulting index
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    # Embedding every note is seconds of pure CPU; run it off the event loop so
    # it does not stall every other in-flight request.
    return await anyio.to_thread.run_sync(_build_semantic_index_sync, rebuild)


def _build_semantic_index_sync(rebuild: bool) -> Dict[str, Any]:
    try:
        previous = None if rebuild else get_index()
        index = build_index(api, get_embedder(), previous=previous)
        save_index(index)
        set_index(index)
        return {
            "status": "success",
            "notes": index.note_count,
            "chunks": index.chunk_count,
        }
    except EmbeddingUnavailable as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error building semantic index: {e}")
        return {"error": str(e)}


@mcp.tool()
async def find_similar_notes(
    note_id: str,
    limit: int = 10,
    min_score: float = 0.3,
) -> Dict[str, Any]:
    """Find notes whose content resembles a given note.

    Use after landing on a relevant note to find the rest of the thinking on
    the same subject, including notes that share no vocabulary with it.

    Args:
        note_id: Note to compare against
        limit: Maximum number of notes to return (default: 10)
        min_score: Cosine similarity floor, 0 to 1 (default: 0.3)

    Returns:
        Dictionary containing the similar notes found
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    min_score = max(0.0, min(float(min_score), 1.0))

    # The similarity matrix multiply plus per-chunk scan is CPU-bound; keep it
    # off the event loop.
    return await anyio.to_thread.run_sync(
        _find_similar_notes_sync, note_id, limit, min_score
    )


def _find_similar_notes_sync(
    note_id: str, limit: int, min_score: float
) -> Dict[str, Any]:
    try:
        index = get_index()
        if index is None or index.chunk_count == 0:
            return {"error": "No semantic index yet. Call build_semantic_index first."}

        notebook_paths = get_notebook_paths(api)
        hits = similar_notes(index, note_id, limit=limit, threshold=min_score)
        return {
            "status": "success",
            "total": len(hits),
            "notes": [
                _shape_hit(
                    hit["id"], hit["title"], hit["chunk"][:SEARCH_SNIPPET_CHARS],
                    hit.get("parent_id", ""), _iso_stamp(hit.get("updated")),
                    "semantic", hit["score"], False, notebook_paths,
                )
                for hit in hits
            ],
        }
    except KeyError:
        return {"error": f"Note {note_id} is not in the semantic index"}
    except Exception as e:
        logger.error(f"Error finding similar notes: {e}")
        return {"error": str(e)}


@mcp.tool()
async def find_linked_notes(
    note_id: str,
    direction: str = "both",
    depth: int = 1,
    limit: int = 50,
    refresh: bool = False,
    include_semantic: bool = False,
    semantic_fanout: int = 5,
    semantic_min_score: float = 0.45,
) -> Dict[str, Any]:
    """Find notes connected to a note by Joplin's internal links.

    Answers both "what does this note link to" and "what links back to this
    note". Joplin's own search cannot do the latter at all, so this is the way
    to find the notes that reference something without knowing their wording.

    Useful after a search has surfaced one good note, to pull in the notes
    around it. Returns identifiers, titles and notebooks only - call get_note
    on whichever results are worth reading, so unrelated note bodies stay out
    of the conversation.

    Set include_semantic to also walk similarity edges. Off by default because
    at depth 1 it returns what find_similar_notes already returns, only with
    more moving parts. It earns its keep at depth 2 or more, where it reaches
    notes related to what this note is related to - genuine transitive
    association that no single query expresses, at the cost of drifting off
    topic the further it goes.

    Args:
        note_id: ID of the note to start from
        direction: "out" for links this note makes, "in" for links back to it,
            "both" to ignore direction (default: "both")
        depth: How many hops to follow, 1-3 (default: 1)
        limit: Maximum number of linked notes to return (default: 50)
        refresh: Rebuild the link graph instead of using the cached one
        include_semantic: Also traverse similarity edges (default: False)
        semantic_fanout: Similarity edges per note when enabled (default: 5)
        semantic_min_score: Similarity floor for those edges (default: 0.45)

    Returns:
        Dictionary containing the origin note and the linked notes found
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    # depth and limit are clamped inside find_neighbours; bound the semantic
    # fan-out here so include_semantic can't request an oversized similarity
    # search per visited node.
    semantic_fanout = max(1, min(int(semantic_fanout), 20))
    semantic_min_score = max(0.0, min(float(semantic_min_score), 1.0))

    # Building the graph pulls every note body, and include_semantic runs a
    # similarity search per visited node — both blocking. Run off-thread.
    return await anyio.to_thread.run_sync(
        _find_linked_notes_sync,
        note_id, direction, depth, limit, refresh,
        include_semantic, semantic_fanout, semantic_min_score,
    )


def _find_linked_notes_sync(
    note_id: str, direction: str, depth: int, limit: int, refresh: bool,
    include_semantic: bool, semantic_fanout: int, semantic_min_score: float,
) -> Dict[str, Any]:
    try:
        graph = get_link_graph(api, refresh=refresh)
        provider = None

        if include_semantic:
            index = get_index()
            if index is None or index.chunk_count == 0:
                return {"error": "include_semantic needs an index. Call build_semantic_index."}

            def provider(current: str) -> Dict[str, float]:  # noqa: F811
                try:
                    hits = similar_notes(
                        index, current, limit=semantic_fanout, threshold=semantic_min_score,
                    )
                except KeyError:
                    return {}
                return {hit["id"]: hit["score"] for hit in hits}

        result = find_neighbours(
            graph, note_id=note_id, direction=direction, depth=depth, limit=limit,
            semantic_provider=provider,
        )
        return {"status": "success", **result}
    except KeyError:
        return {"error": f"Note {note_id} was not found"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error finding linked notes: {e}")
        return {"error": str(e)}


@mcp.tool()
async def list_notebooks() -> Dict[str, Any]:
    """List available Joplin notebooks.

    Returns:
        Dictionary containing the notebook tree
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    try:
        notebooks = api.list_notebooks()
        return {
            "status": "success",
            "total": len(notebooks),
            "notebooks": [serialize_notebook(notebook) for notebook in notebooks],
        }
    except Exception as e:
        logger.error(f"Error listing notebooks: {e}")
        return {"error": str(e)}


@mcp.tool()
async def create_notebook(
    title: str,
    parent_id: Optional[str] = None,
    parent_notebook_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new Joplin notebook.

    Args:
        title: Notebook title
        parent_id: Parent notebook ID (optional)
        parent_notebook_name: Parent notebook name or path (optional)

    Returns:
        Dictionary containing the created notebook data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    try:
        parent_id = resolve_notebook_id(parent_id, parent_notebook_name)
        notebook = api.create_notebook(title=title, parent_id=parent_id)
        return {
            "status": "success",
            "notebook": serialize_notebook(notebook),
        }
    except Exception as e:
        logger.error(f"Error creating notebook: {e}")
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
async def create_note(
    title: str,
    body: Optional[str] = None,
    parent_id: Optional[str] = None,
    notebook_name: Optional[str] = None,
    is_todo: bool = False,
) -> Dict[str, Any]:
    """Create a new note in Joplin.

    Args:
        title: Note title
        body: Note content in Markdown (optional)
        parent_id: ID of parent folder (optional)
        notebook_name: Notebook title or full path (optional)
        is_todo: Whether this is a todo item (optional)

    Returns:
        Dictionary containing the created note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    try:
        parent_id = resolve_notebook_id(parent_id, notebook_name)
        note = api.create_note(
            title=title,
            body=body,
            parent_id=parent_id,
            is_todo=is_todo
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
async def update_note(
    note_id: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    parent_id: Optional[str] = None,
    notebook_name: Optional[str] = None,
    is_todo: Optional[bool] = None,
) -> Dict[str, Any]:
    """Update an existing note in Joplin.

    Args:
        note_id: ID of note to update
        title: New title (optional)
        body: New content (optional)
        parent_id: New parent folder ID (optional)
        notebook_name: New notebook title or full path (optional)
        is_todo: New todo status (optional)

    Returns:
        Dictionary containing the updated note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    try:
        parent_id = resolve_notebook_id(parent_id, notebook_name)
        note = api.update_note(
            note_id=note_id,
            title=title,
            body=body,
            parent_id=parent_id,
            is_todo=is_todo
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
async def import_markdown(
    file_path: str,
    notebook_name: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Import a markdown file as a new note.

    Args:
        file_path: Path to the markdown file
        notebook_name: Destination notebook title or full path (optional)
        parent_id: Destination notebook ID (optional)

    Returns:
        Dictionary containing the created note data
    """
    if not api:
        return {"error": "Joplin API client not initialized"}

    try:
        file_path = Path(file_path)
        md_content = MarkdownContent.from_file(file_path)
        parent_id = resolve_notebook_id(parent_id, notebook_name)
        
        note = api.create_note(
            title=md_content.title,
            body=md_content.content,
            parent_id=parent_id
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
