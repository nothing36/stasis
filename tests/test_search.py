"""Tests for MemorySearch indexing behaviour."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _fake_embedding(_text: str) -> np.ndarray:
    """Return a deterministic unit-norm float32 vector (no real model needed)."""
    v = np.ones(384, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestIndexMemoryFileCleanup(unittest.TestCase):
    """Verify that re-indexing MEMORY.md removes stale chunks from both tables."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)

        patcher = patch('stasis.core.search.SentenceTransformer')
        self.mock_st_cls = patcher.start()
        self.addCleanup(patcher.stop)

        mock_model = MagicMock()
        mock_model.encode.side_effect = _fake_embedding
        self.mock_st_cls.return_value = mock_model

        from stasis.core.search import MemorySearch
        self.engine = MemorySearch(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def _row_counts(self, source_file: str):
        """Return (fts_count, embeddings_count) for the given source_file."""
        conn = sqlite3.connect(self.engine.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT count(*) FROM memory_fts WHERE source_file = ?',
            (source_file,)
        )
        fts_count = cursor.fetchone()[0]
        cursor.execute(
            'SELECT count(*) FROM memory_embeddings WHERE source_file = ?',
            (source_file,)
        )
        emb_count = cursor.fetchone()[0]
        conn.close()
        return fts_count, emb_count

    def test_stale_chunks_removed_on_reindex(self):
        memory_path = self.workspace / 'MEMORY.md'

        # Build initial content large enough to produce multiple chunks so that
        # appending shifts boundaries and guarantees new content hashes.
        initial_content = '\n'.join(
            [f'# Memory entry {i}\n' + ('word ' * 200) for i in range(5)]
        )
        memory_path.write_text(initial_content, encoding='utf-8')

        self.engine.index_memory_file()

        initial_fts, initial_emb = self._row_counts(str(memory_path))
        self.assertGreater(initial_fts, 0, 'Initial index should contain at least one chunk')
        self.assertEqual(initial_fts, initial_emb, 'FTS and embeddings tables should be in sync after initial index')

        # Append a new entry — shifts chunk boundaries, producing new hashes
        appended_content = initial_content + '\n# New appended memory entry\n' + ('word ' * 200)
        memory_path.write_text(appended_content, encoding='utf-8')

        self.engine.index_memory_file()

        expected_chunks = len(self.engine._chunk_content(appended_content, str(memory_path)))
        final_fts, final_emb = self._row_counts(str(memory_path))

        self.assertEqual(
            final_fts, expected_chunks,
            f'FTS table has {final_fts} rows but expected {expected_chunks} — stale chunks were not removed'
        )
        self.assertEqual(
            final_emb, expected_chunks,
            f'Embeddings table has {final_emb} rows but expected {expected_chunks} — stale chunks were not removed'
        )


if __name__ == '__main__':
    unittest.main()
