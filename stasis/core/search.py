"""
Hybrid search engine for memory retrieval.

Combines BM25 keyword search with semantic vector search for accurate
and flexible memory retrieval.
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """A single search result with metadata."""
    content: str
    score: float
    line_start: int
    line_end: int
    timestamp: str
    source_file: str


class MemorySearch:
    """Hybrid search engine using BM25 + vector embeddings."""

    def __init__(self, workspace: Path):
        """
        Initialize search engine.

        Args:
            workspace: Path to workspace directory
        """
        self.workspace = workspace
        self.index_dir = workspace / '.stasis'
        self.db_path = self.index_dir / 'index.db'

        # ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # load embedding model (small and fast)
        print('[Stasis] Loading embedding model...')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print('[Stasis] Embedding model loaded')

        # initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with FTS5 and embeddings tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # FTS5 table for BM25 search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                source_file,
                line_start,
                line_end,
                timestamp,
                content_hash
            )
        ''')

        # embeddings table for vector search
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE,
                embedding BLOB,
                content TEXT,
                source_file TEXT,
                line_start INTEGER,
                line_end INTEGER,
                timestamp TEXT
            )
        ''')

        # metadata table for file hashes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                last_indexed TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def index_memory_file(self, force: bool = False) -> None:
        """
        Index MEMORY.md file with delta updates.

        Args:
            force: Force full re-index even if file hasn't changed
        """
        memory_path = self.workspace / 'MEMORY.md'
        if not memory_path.exists():
            print('[Stasis] MEMORY.md not found, skipping indexing')
            return

        # check if file changed
        current_hash = self._hash_file(memory_path)
        if not force and not self._file_changed(str(memory_path), current_hash):
            print('[Stasis] MEMORY.md unchanged, skipping indexing')
            return

        print('[Stasis] Indexing MEMORY.md...')

        # read and chunk file
        content = memory_path.read_text(encoding='utf-8')
        chunks = self._chunk_content(content, str(memory_path))

        # index chunks
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        indexed_count = 0
        for chunk in chunks:
            # check if chunk already indexed (by content hash)
            cursor.execute(
                'SELECT id FROM memory_embeddings WHERE content_hash = ?',
                (chunk['content_hash'],)
            )
            if cursor.fetchone() and not force:
                continue  # skip unchanged chunks

            # generate embedding
            embedding = self.model.encode(chunk['content'])
            embedding_blob = embedding.tobytes()

            # insert/update FTS5
            cursor.execute('''
                INSERT OR REPLACE INTO memory_fts (content, source_file, line_start, line_end, timestamp, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                chunk['content'],
                chunk['source_file'],
                chunk['line_start'],
                chunk['line_end'],
                chunk['timestamp'],
                chunk['content_hash']
            ))

            # insert/update embeddings
            cursor.execute('''
                INSERT OR REPLACE INTO memory_embeddings (content_hash, embedding, content, source_file, line_start, line_end, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk['content_hash'],
                embedding_blob,
                chunk['content'],
                chunk['source_file'],
                chunk['line_start'],
                chunk['line_end'],
                chunk['timestamp']
            ))

            indexed_count += 1

        # update file metadata
        cursor.execute('''
            INSERT OR REPLACE INTO file_metadata (file_path, file_hash, last_indexed)
            VALUES (?, ?, datetime('now'))
        ''', (str(memory_path), current_hash))

        conn.commit()
        conn.close()

        print(f'[Stasis] Indexed {indexed_count} chunks from MEMORY.md')

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search memory using hybrid BM25 + vector search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results ranked by relevance
        """
        # generate query embedding
        query_embedding = self.model.encode(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # BM25 search via FTS5
        cursor.execute('''
            SELECT content, source_file, line_start, line_end, timestamp, bm25(memory_fts) as bm25_score
            FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
        ''', (query, top_k * 2))  # get more for hybrid ranking

        bm25_results = {
            row[0]: {
                'content': row[0],
                'source_file': row[1],
                'line_start': row[2],
                'line_end': row[3],
                'timestamp': row[4],
                'bm25_score': abs(row[5])  # FTS5 returns negative scores
            }
            for row in cursor.fetchall()
        }

        # vector search
        cursor.execute('SELECT content_hash, embedding, content, source_file, line_start, line_end, timestamp FROM memory_embeddings')

        vector_results = {}
        for row in cursor.fetchall():
            content_hash, embedding_blob, content, source_file, line_start, line_end, timestamp = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            vector_results[content] = {
                'content': content,
                'source_file': source_file,
                'line_start': line_start,
                'line_end': line_end,
                'timestamp': timestamp,
                'vector_score': float(similarity)
            }

        conn.close()

        # hybrid ranking: 0.7 vector + 0.3 BM25
        combined = {}
        all_contents = set(bm25_results.keys()) | set(vector_results.keys())

        for content in all_contents:
            bm25_score = bm25_results.get(content, {}).get('bm25_score', 0)
            vector_score = vector_results.get(content, {}).get('vector_score', 0)

            # normalize BM25 scores (simple min-max)
            if bm25_results:
                max_bm25 = max(r['bm25_score'] for r in bm25_results.values())
                bm25_score = bm25_score / max_bm25 if max_bm25 > 0 else 0

            hybrid_score = 0.7 * vector_score + 0.3 * bm25_score

            # use data from whichever source has it
            data = bm25_results.get(content) or vector_results.get(content)
            combined[content] = {**data, 'hybrid_score': hybrid_score}

        # sort by hybrid score and return top k
        ranked = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]

        return [
            SearchResult(
                content=r['content'],
                score=r['hybrid_score'],
                line_start=r['line_start'],
                line_end=r['line_end'],
                timestamp=r['timestamp'],
                source_file=r['source_file']
            )
            for r in ranked
        ]

    def _chunk_content(self, content: str, source_file: str) -> List[dict]:
        """
        Chunk content into overlapping segments.

        Args:
            content: Text content to chunk
            source_file: Source file path

        Returns:
            List of chunk dictionaries with metadata
        """
        chunk_size = 1600  # chars
        overlap = 320  # chars
        chunks = []

        lines = content.split('\n')
        current_chunk = []
        current_length = 0
        line_start = 0
        current_timestamp = ''

        for i, line in enumerate(lines):
            # extract timestamp if present
            if line.startswith('[') and ']' in line:
                current_timestamp = line.split(']')[0][1:]

            current_chunk.append(line)
            current_length += len(line) + 1  # +1 for newline

            if current_length >= chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()

                chunks.append({
                    'content': chunk_text,
                    'source_file': source_file,
                    'line_start': line_start,
                    'line_end': i,
                    'timestamp': current_timestamp,
                    'content_hash': chunk_hash
                })

                # keep overlap for next chunk
                overlap_lines = []
                overlap_length = 0
                for line in reversed(current_chunk):
                    overlap_length += len(line) + 1
                    overlap_lines.insert(0, line)
                    if overlap_length >= overlap:
                        break

                current_chunk = overlap_lines
                current_length = overlap_length
                line_start = i - len(overlap_lines) + 1

        # add remaining content as final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()

            chunks.append({
                'content': chunk_text,
                'source_file': source_file,
                'line_start': line_start,
                'line_end': len(lines) - 1,
                'timestamp': current_timestamp,
                'content_hash': chunk_hash
            })

        return chunks

    def _hash_file(self, file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def _file_changed(self, file_path: str, current_hash: str) -> bool:
        """Check if file has changed since last index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT file_hash FROM file_metadata WHERE file_path = ?', (file_path,))
        row = cursor.fetchone()

        conn.close()

        if not row:
            return True  # never indexed

        return row[0] != current_hash
