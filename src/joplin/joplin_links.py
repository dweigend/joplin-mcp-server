"""Link graph over Joplin notes.

Joplin's Data API can tell you what a note contains but not what points at it,
and its search is literal, so backlinks are a capability that is missing
entirely rather than a ranking that could be improved. The only place the
information lives is the note bodies, so the graph is derived by scanning them.

The graph is cached: rebuilding means pulling every note body, which is wasted
work on a library that has not changed between two calls in the same session.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.joplin.joplin_api import JoplinAPI, JoplinNotebook, OrderDirection

logger = logging.getLogger(__name__)

# Joplin internal links are markdown links whose target is :/<32 hex>. Resource
# (attachment) links use the identical syntax, so a target is only kept once it
# resolves to a known note id - otherwise every embedded image reads as a link.
INTERNAL_LINK_PATTERN = re.compile(r"\]\(:/([0-9a-fA-F]{32})\)")

NOTE_FIELDS = ["id", "title", "body", "parent_id", "updated_time"]
PAGE_LIMIT = 100

# The fingerprint catches edits immediately. The TTL is the backstop for
# deletions, which move no surviving note's updated_time and so are invisible
# to the fingerprint.
CACHE_TTL_SECONDS = 300

DIRECTIONS = ("out", "in", "both")
MAX_DEPTH = 3
MAX_LIMIT = 200
# Cap on nodes whose neighbours get expanded. limit only truncates the output;
# without this the walk still explores the whole reachable component and, with
# include_semantic, runs one full similarity search per visited node. Bounding
# expansions bounds that cost regardless of library size.
MAX_EXPANSIONS = 500


@dataclass
class LinkGraph:
    """Note metadata plus the forward and reverse link maps."""

    titles: dict[str, str] = field(default_factory=dict)
    notebooks: dict[str, str] = field(default_factory=dict)
    outgoing: dict[str, set[str]] = field(default_factory=dict)
    incoming: dict[str, set[str]] = field(default_factory=dict)
    fingerprint: str = ""
    built_at: float = 0.0

    @property
    def note_count(self) -> int:
        return len(self.titles)

    @property
    def link_count(self) -> int:
        return sum(len(targets) for targets in self.outgoing.values())


_cache: LinkGraph | None = None
_notebook_cache: tuple[float, dict[str, str]] | None = None
# Serializes cache init/rebuild once the tool layer runs these off-thread, and
# prevents two callers from each doing a full-graph rebuild at once.
_lock = threading.RLock()


def flatten_notebook_paths(
    notebooks: Iterable[JoplinNotebook],
    prefix: str | None = None,
    _visited: set[str] | None = None,
) -> dict[str, str]:
    """Map each notebook id to its full slash-separated path."""
    paths: dict[str, str] = {}
    # Joplin enforces a tree, but a malformed API response with a parent cycle
    # would otherwise recurse until the stack is exhausted. Skip any id already
    # on the current path.
    visited = _visited if _visited is not None else set()

    for notebook in notebooks:
        if notebook.id in visited:
            continue
        visited.add(notebook.id)
        path = f"{prefix}/{notebook.title}" if prefix else notebook.title
        paths[notebook.id] = path
        paths.update(
            flatten_notebook_paths(notebook.children or [], path, visited)
        )

    return paths


def get_notebook_paths(api: JoplinAPI) -> dict[str, str]:
    """Notebook id to full path, cached briefly. Notebook trees change rarely."""
    global _notebook_cache

    with _lock:
        if _notebook_cache and (time.monotonic() - _notebook_cache[0]) < CACHE_TTL_SECONDS:
            return _notebook_cache[1]

        try:
            paths = flatten_notebook_paths(api.list_notebooks())
        except Exception as exc:  # noqa: BLE001 - paths are a nicety, not a requirement
            logger.warning(f"Could not resolve notebook paths: {exc}")
            return _notebook_cache[1] if _notebook_cache else {}

        _notebook_cache = (time.monotonic(), paths)
        return paths


def corpus_fingerprint(api: JoplinAPI) -> str:
    """Cheap change token: the id and timestamp of the most recently touched note."""
    response = api.get_notes(
        page=1,
        limit=1,
        # title is dead weight here but JoplinNote refuses to parse without it.
        fields=["id", "title", "updated_time"],
        order_by="updated_time",
        order_dir=OrderDirection.DESC,
    )
    if not response.items:
        return "empty"

    newest = response.items[0]
    stamp = newest.updated_time.isoformat() if newest.updated_time else ""
    return f"{newest.id}:{stamp}"


def build_link_graph(api: JoplinAPI) -> LinkGraph:
    """Scan every note body and assemble the forward and reverse link maps."""
    graph = LinkGraph()

    try:
        notebook_paths = flatten_notebook_paths(api.list_notebooks())
    except Exception as exc:  # noqa: BLE001 - paths are a nicety, not a requirement
        logger.warning(f"Could not resolve notebook paths, continuing without them: {exc}")
        notebook_paths = {}

    bodies: dict[str, str] = {}
    page = 1
    while True:
        response = api.get_notes(page=page, limit=PAGE_LIMIT, fields=NOTE_FIELDS)
        for note in response.items:
            graph.titles[note.id] = note.title or "(untitled)"
            graph.notebooks[note.id] = notebook_paths.get(note.parent_id or "", "")
            bodies[note.id] = note.body or ""
        if not response.has_more:
            break
        page += 1

    for note_id, body in bodies.items():
        targets = {
            target.lower()
            for target in INTERNAL_LINK_PATTERN.findall(body)
        }
        # Drop attachment ids and self-links; neither is a navigable edge.
        targets = {t for t in targets if t in graph.titles and t != note_id}
        if not targets:
            continue

        graph.outgoing.setdefault(note_id, set()).update(targets)
        for target in targets:
            graph.incoming.setdefault(target, set()).add(note_id)

    graph.built_at = time.monotonic()
    logger.info(f"Built link graph: {graph.note_count} notes, {graph.link_count} links")
    return graph


def get_link_graph(api: JoplinAPI, refresh: bool = False) -> LinkGraph:
    """Return the cached graph, rebuilding it when the library has moved on."""
    global _cache

    with _lock:
        if refresh or _cache is None:
            return _rebuild_locked(api)

        expired = (time.monotonic() - _cache.built_at) > CACHE_TTL_SECONDS
        try:
            changed = corpus_fingerprint(api) != _cache.fingerprint
        except Exception as exc:  # noqa: BLE001 - a stale graph beats a failed call
            logger.warning(f"Could not check for changes, serving cached graph: {exc}")
            changed = False

        if expired or changed:
            return _rebuild_locked(api)

        return _cache


def _rebuild_locked(api: JoplinAPI) -> LinkGraph:
    """Build a graph, fingerprint included, then publish it. Call under _lock.

    The fingerprint is set before the graph is assigned to _cache so no reader
    can observe a published graph whose fingerprint is still unset.
    """
    global _cache
    graph = build_link_graph(api)
    graph.fingerprint = corpus_fingerprint(api)
    _cache = graph
    return _cache


def reset_cache() -> None:
    """Drop the cached graph and notebook paths. Used by tests."""
    global _cache, _notebook_cache
    _cache = None
    _notebook_cache = None


def _neighbours_of(graph: LinkGraph, note_id: str, direction: str) -> dict[str, str]:
    """Adjacent note ids mapped to how they connect, from note_id's point of view."""
    outbound = graph.outgoing.get(note_id, set())
    inbound = graph.incoming.get(note_id, set())

    if direction == "out":
        return {target: "outbound" for target in outbound}
    if direction == "in":
        return {source: "inbound" for source in inbound}

    relations = {target: "outbound" for target in outbound}
    for source in inbound:
        relations[source] = "both" if source in relations else "inbound"
    return relations


def find_neighbours(
    graph: LinkGraph,
    note_id: str,
    direction: str = "both",
    depth: int = 1,
    limit: int = 50,
    semantic_provider: Callable[[str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Breadth-first walk out from a note along its links.

    Args:
        graph: The link graph to walk.
        note_id: Note to start from.
        direction: "out" for links the note makes, "in" for links back to it,
            "both" to treat the graph as undirected.
        depth: How many hops to follow.
        limit: Maximum neighbours to return.
        semantic_provider: Optional id -> {neighbour id: score} lookup whose
            results are walked as additional edges. At depth 1 this is the same
            answer a nearest-neighbour query gives; past depth 1 it is not, and
            that transitive reach is the reason to enable it.

    Returns:
        Dictionary with the origin note, the neighbours found and whether the
        result was truncated.

    Raises:
        KeyError: If note_id is not a known note.
        ValueError: If direction is not one of "out", "in" or "both".
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
    if note_id not in graph.titles:
        raise KeyError(note_id)

    depth = max(1, min(depth, MAX_DEPTH))
    limit = max(1, min(limit, MAX_LIMIT))

    seen = {note_id}
    frontier = [note_id]
    found: list[dict[str, Any]] = []
    expansions = 0
    capped = False

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for current in frontier:
            if expansions >= MAX_EXPANSIONS:
                capped = True
                break
            expansions += 1
            relations = _neighbours_of(graph, current, direction)
            scores: dict[str, float] = {}
            if semantic_provider is not None:
                for neighbour, score in semantic_provider(current).items():
                    scores[neighbour] = score
                    # An explicit link is the stronger claim; don't overwrite it.
                    relations.setdefault(neighbour, "semantic")

            for neighbour, relation in relations.items():
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                next_frontier.append(neighbour)
                entry = {
                    "id": neighbour,
                    "title": graph.titles.get(neighbour, "(unknown)"),
                    "notebook": graph.notebooks.get(neighbour, ""),
                    "relation": relation,
                    "depth": hop,
                    # Only meaningful past the first hop, where the connection
                    # is otherwise unexplained.
                    "via": graph.titles.get(current) if hop > 1 else None,
                }
                if relation == "semantic":
                    entry["score"] = round(scores[neighbour], 4)
                found.append(entry)
        if capped or not next_frontier:
            break
        frontier = next_frontier

    if capped:
        logger.info(
            "find_neighbours hit the %d-expansion cap for %s; results partial",
            MAX_EXPANSIONS, note_id,
        )

    found.sort(key=lambda item: (item["depth"], item["title"].lower()))

    return {
        "note": {
            "id": note_id,
            "title": graph.titles.get(note_id, "(unknown)"),
            "notebook": graph.notebooks.get(note_id, ""),
        },
        "total": len(found),
        "truncated": capped or len(found) > limit,
        "links": found[:limit],
    }
