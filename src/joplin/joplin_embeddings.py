"""Local embedding index over Joplin notes.

Runs the same quantized all-MiniLM-L6-v2 that the knowledge-graph plugin
bundles, through onnxruntime rather than torch, so the two agree on what
"similar" means and nothing has to be downloaded at run time. Everything is
local: the model is a file on disk and the index is a file next to it.

Retrieval is brute force. At a few thousand chunks a dot product against one
contiguous matrix is well under a millisecond, so an approximate index would
add moving parts and lose recall for no measurable gain.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Mirrors the plugin's own defaults (src/settings.ts) so both sides chunk the
# same way; a query embedded here must land in the same space as the index.
CHUNK_CHARS = 500
CHUNK_OVERLAP = 80
MAX_SEQUENCE_TOKENS = 256

# Rank-fusion constant. The paper's 60 is tuned for TREC runs of thousands of
# results; across a 15-item list it flattens the rank spread to 1.23x, which the
# weight ratio then swamps, so the lighter-weighted list can never interleave and
# hybrid silently degenerates into whichever side weighs more. A small constant
# keeps the spread wide enough for weight to mean what it says.
RRF_K = 1

DEFAULT_MODEL_DIRS = (
    "/usr/local/src/joplin-plugin-knowledge-graph/assets/models/Xenova/all-MiniLM-L6-v2",
)
INDEX_PATH = Path(
    os.environ.get("JOPLIN_MCP_INDEX", Path.home() / ".cache" / "joplin-mcp" / "index.npz")
)
META_PATH = INDEX_PATH.with_suffix(".meta.json")

NOTE_FIELDS = ["id", "title", "body", "parent_id", "updated_time"]
PAGE_LIMIT = 100


class EmbeddingUnavailable(RuntimeError):
    """Raised when the model or its runtime is not present."""


def model_dir() -> Path:
    """Locate the ONNX model directory."""
    configured = os.environ.get("JOPLIN_MCP_MODEL_DIR")
    candidates = [configured] if configured else list(DEFAULT_MODEL_DIRS)

    for candidate in candidates:
        path = Path(candidate)
        if (path / "onnx" / "model_quantized.onnx").exists() and (path / "tokenizer.json").exists():
            return path

    raise EmbeddingUnavailable(
        "Embedding model not found. Set JOPLIN_MCP_MODEL_DIR to a directory "
        "containing tokenizer.json and onnx/model_quantized.onnx (the "
        "knowledge-graph plugin creates one under assets/models/ when built)."
    )


class Embedder:
    """Mean-pooled MiniLM sentence embeddings via onnxruntime."""

    def __init__(self) -> None:
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise EmbeddingUnavailable(
                f"Semantic search needs onnxruntime and tokenizers: {exc}"
            ) from exc

        directory = model_dir()
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQUENCE_TOKENS)
        self.tokenizer.enable_padding(length=None)

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(
            str(directory / "onnx" / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {node.name for node in self.session.get_inputs()}
        logger.info(f"Loaded embedding model from {directory}")

    @property
    def dims(self) -> int:
        return 384

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed texts into L2-normalised row vectors."""
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)

        chunks = []
        for start in range(0, len(texts), batch_size):
            chunks.append(self._encode_batch(texts[start:start + batch_size]))
        return np.vstack(chunks)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]

        # Mean-pool over real tokens only; padding must not drag vectors around.
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-9)).astype(np.float32)


@dataclass
class VectorIndex:
    """Chunk vectors plus the note metadata needed to report a hit."""

    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 384), dtype=np.float32))
    chunk_note: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    chunk_texts: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    note_titles: list[str] = field(default_factory=list)
    note_parents: list[str] = field(default_factory=list)
    note_stamps: list[int] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.note_ids)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_texts)

    def note_index(self, note_id: str) -> int | None:
        try:
            return self.note_ids.index(note_id)
        except ValueError:
            return None


_embedder: Embedder | None = None
_index: VectorIndex | None = None
_index_loaded = False
# Guards the lazy init of the globals above. Once the tool layer offloads the
# heavy calls to a thread pool, these are touched from more than one thread, so
# the read-check-assign in each accessor needs to be atomic. RLock so a caller
# already holding it can re-enter without deadlock.
_lock = threading.RLock()


def get_embedder() -> Embedder:
    """The process-wide embedder, loaded on first use (about 5s)."""
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder()
        return _embedder


def get_index() -> VectorIndex | None:
    """The cached index, read from disk once. None when never built."""
    global _index, _index_loaded
    with _lock:
        if not _index_loaded:
            _index = load_index()
            _index_loaded = True
        return _index


def set_index(index: VectorIndex | None) -> None:
    """Install an index as the current one, e.g. straight after a build."""
    global _index, _index_loaded
    with _lock:
        _index = index
        _index_loaded = True


def reset_runtime() -> None:
    """Forget the loaded embedder and index. Used by tests."""
    global _embedder, _index, _index_loaded
    _embedder = None
    _index = None
    _index_loaded = False


def split_chunks(title: str, body: str) -> list[str]:
    """Split a note into overlapping chunks, each prefixed with the note title.

    The title is repeated into every chunk because chunks are retrieved on their
    own: a paragraph that never restates its subject is unfindable without it.
    """
    prefix = f"{title.strip()}\n\n" if title.strip() else ""
    budget = max(100, CHUNK_CHARS - len(prefix))
    overlap = max(0, min(CHUNK_OVERLAP, budget - 50))

    text = (body or "").strip()
    if not text:
        return [prefix.strip()] if prefix else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(prefix + buffer.strip())
        buffer = buffer[-overlap:] if overlap else ""

    for paragraph in paragraphs:
        while len(paragraph) > budget:
            room = budget - len(buffer)
            if room <= 0:
                flush()
                continue
            buffer += paragraph[:room]
            paragraph = paragraph[room:]
            flush()

        if len(buffer) + len(paragraph) + 2 > budget:
            flush()
        buffer = f"{buffer}\n\n{paragraph}".strip() if buffer.strip() else paragraph

    if buffer.strip():
        chunks.append(prefix + buffer.strip())

    return chunks or ([prefix.strip()] if prefix else [])


def _fetch_notes(api: Any) -> list[Any]:
    notes, page = [], 1
    while True:
        response = api.get_notes(page=page, limit=PAGE_LIMIT, fields=NOTE_FIELDS)
        notes.extend(response.items)
        if not response.has_more:
            break
        page += 1
    return notes


def _stamp(note: Any) -> int:
    return int(note.updated_time.timestamp()) if note.updated_time else 0


def build_index(api: Any, embedder: Embedder, previous: VectorIndex | None = None) -> VectorIndex:
    """Embed the library, reusing vectors for notes that have not changed."""
    notes = _fetch_notes(api)
    reusable: dict[str, tuple[list[str], np.ndarray]] = {}

    if previous is not None and previous.chunk_count:
        stamps = dict(zip(previous.note_ids, previous.note_stamps))
        for position, note_id in enumerate(previous.note_ids):
            rows = np.flatnonzero(previous.chunk_note == position)
            if rows.size:
                reusable[note_id] = (
                    [previous.chunk_texts[r] for r in rows],
                    previous.matrix[rows],
                )
        previous_stamps = stamps
    else:
        previous_stamps = {}

    index = VectorIndex(matrix=np.zeros((0, embedder.dims), dtype=np.float32))
    vector_blocks: list[np.ndarray] = []
    chunk_note: list[int] = []
    pending: list[tuple[int, list[str]]] = []
    reused_notes = 0

    for note in notes:
        chunks = split_chunks(note.title or "", note.body or "")
        if not chunks:
            continue

        position = len(index.note_ids)
        index.note_ids.append(note.id)
        index.note_titles.append(note.title or "(untitled)")
        index.note_parents.append(note.parent_id or "")
        index.note_stamps.append(_stamp(note))

        cached = reusable.get(note.id)
        unchanged = cached is not None and previous_stamps.get(note.id) == _stamp(note)
        if unchanged and len(cached[0]) == len(chunks):
            index.chunk_texts.extend(cached[0])
            vector_blocks.append(cached[1])
            chunk_note.extend([position] * len(cached[0]))
            reused_notes += 1
        else:
            pending.append((position, chunks))

    if pending:
        flat = [text for _, texts in pending for text in texts]
        logger.info(f"Embedding {len(flat)} chunks across {len(pending)} notes")
        fresh = embedder.encode(flat)
        offset = 0
        for position, texts in pending:
            index.chunk_texts.extend(texts)
            vector_blocks.append(fresh[offset:offset + len(texts)])
            chunk_note.extend([position] * len(texts))
            offset += len(texts)

    index.matrix = (
        np.vstack(vector_blocks) if vector_blocks
        else np.zeros((0, embedder.dims), dtype=np.float32)
    )
    index.chunk_note = np.array(chunk_note, dtype=np.int32)
    logger.info(
        f"Index: {index.note_count} notes, {index.chunk_count} chunks "
        f"({reused_notes} notes reused)"
    )
    return index


def save_index(index: VectorIndex, path: Path = INDEX_PATH) -> None:
    """Persist the index to disk.

    Written atomically: each file goes to a temp path and is os.replace()d into
    place, so a crash or a concurrent reader never sees a half-written file. The
    npz and its metadata are still two separate files, so load_index validates
    that the pair is mutually consistent rather than trusting it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_npz = path.with_name(path.name + ".tmp")
    # Pass a file handle, not the path: numpy appends ".npz" to a path that
    # lacks it, which would corrupt the temp-file name.
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(
            handle, matrix=index.matrix, chunk_note=index.chunk_note
        )
    os.replace(tmp_npz, path)

    tmp_meta = META_PATH.with_name(META_PATH.name + ".tmp")
    tmp_meta.write_text(
        json.dumps({
            "chunk_texts": index.chunk_texts,
            "note_ids": index.note_ids,
            "note_titles": index.note_titles,
            "note_parents": index.note_parents,
            "note_stamps": index.note_stamps,
        }),
        encoding="utf-8",
    )
    os.replace(tmp_meta, META_PATH)


def load_index(path: Path = INDEX_PATH) -> VectorIndex | None:
    """Load the index, or None when it has not been built or is inconsistent."""
    if not path.exists() or not META_PATH.exists():
        return None

    try:
        arrays = np.load(path)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        index = VectorIndex(
            matrix=arrays["matrix"],
            chunk_note=arrays["chunk_note"],
            chunk_texts=meta["chunk_texts"],
            note_ids=meta["note_ids"],
            note_titles=meta["note_titles"],
            note_parents=meta.get("note_parents") or [""] * len(meta["note_ids"]),
            note_stamps=meta["note_stamps"],
        )
    except Exception as exc:  # noqa: BLE001 - a corrupt index should rebuild, not crash
        logger.warning(f"Could not load index, treating as absent: {exc}")
        return None

    # A torn write (new npz + stale meta, each individually valid) would load
    # an index whose arrays disagree and then IndexError at query time. Reject
    # the mismatch here so it rebuilds instead.
    chunk_len = index.matrix.shape[0]
    if not (chunk_len == len(index.chunk_note) == len(index.chunk_texts)):
        logger.warning("Index chunk arrays disagree in length; rebuilding.")
        return None
    note_len = index.note_count
    if not (note_len == len(index.note_titles) == len(index.note_parents)
            == len(index.note_stamps)):
        logger.warning("Index note arrays disagree in length; rebuilding.")
        return None
    if index.chunk_count and (
        int(index.chunk_note.min()) < 0 or int(index.chunk_note.max()) >= note_len
    ):
        logger.warning("Index chunk_note references a missing note; rebuilding.")
        return None

    return index


def _best_chunk_per_note(index: VectorIndex, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score every note by its single best-matching chunk."""
    scores = index.matrix @ query
    best_score = np.full(index.note_count, -np.inf, dtype=np.float32)
    best_chunk = np.full(index.note_count, -1, dtype=np.int32)

    # A long note with one sharply relevant paragraph should beat a note that is
    # vaguely on topic throughout, so take the max rather than the mean.
    for chunk, note in enumerate(index.chunk_note):
        if scores[chunk] > best_score[note]:
            best_score[note] = scores[chunk]
            best_chunk[note] = chunk

    return best_score, best_chunk


def semantic_search(
    index: VectorIndex,
    query_vector: np.ndarray,
    limit: int,
    exclude: str | None = None,
) -> list[dict[str, Any]]:
    """Rank notes by cosine similarity to a query vector."""
    if index.chunk_count == 0:
        return []

    best_score, best_chunk = _best_chunk_per_note(index, query_vector)
    order = np.argsort(-best_score)

    hits = []
    for note in order:
        if best_chunk[note] < 0 or not np.isfinite(best_score[note]):
            continue
        if exclude is not None and index.note_ids[note] == exclude:
            continue
        hits.append({
            "id": index.note_ids[note],
            "title": index.note_titles[note],
            "parent_id": index.note_parents[note] if index.note_parents else "",
            "updated": index.note_stamps[note],
            "score": round(float(best_score[note]), 4),
            "chunk": index.chunk_texts[best_chunk[note]],
        })
        if len(hits) >= limit:
            break
    return hits


def fuse(
    semantic: list[dict[str, Any]],
    keyword_ids: Iterable[str],
    keyword_weight: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Blend two rankings with reciprocal rank fusion.

    RRF combines orderings without the two scoring scales needing to be
    comparable, which matters because BM25-ish keyword ranks and cosine
    similarities are not.
    """
    keyword_ids = list(keyword_ids)
    if keyword_weight <= 0 or not keyword_ids:
        return semantic[:limit]

    fused: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(semantic):
        fused[hit["id"]] = {
            **hit,
            "fused_score": (1 - keyword_weight) / (RRF_K + rank + 1),
            "matched": "semantic",
        }

    for rank, note_id in enumerate(keyword_ids):
        contribution = keyword_weight / (RRF_K + rank + 1)
        existing = fused.get(note_id)
        if existing:
            existing["fused_score"] += contribution
            existing["matched"] = "both"
            continue
        # Keyword-only hits have no vector score; carry them on fusion score
        # alone so exact strings the model cannot represent still surface.
        fused[note_id] = {
            "id": note_id,
            "title": "",
            "parent_id": "",
            "updated": 0,
            "score": 0.0,
            "chunk": "",
            "fused_score": contribution,
            "matched": "keyword",
        }

    ranked = sorted(fused.values(), key=lambda item: -item["fused_score"])
    return ranked[:limit]


def similar_notes(
    index: VectorIndex,
    note_id: str,
    limit: int,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Nearest neighbours of a note, scored by its best chunk against theirs.

    Raises:
        KeyError: If the note is not in the index.
    """
    position = index.note_index(note_id)
    if position is None:
        raise KeyError(note_id)

    rows = np.flatnonzero(index.chunk_note == position)
    if rows.size == 0:
        return []

    # Represent the note by the centroid of its chunks, renormalised.
    centroid = index.matrix[rows].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-9:
        return []

    hits = semantic_search(index, centroid / norm, limit, exclude=note_id)
    return [hit for hit in hits if hit["score"] >= threshold]
