"""
Memory system for persistent context.

Manages markdown-based memory files that give Stasis long-term recall
across conversations. All memory is human-readable and editable.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class Memory:
    """Manages reading and writing memory files in the workspace."""

    def __init__(self, workspace: Path):
        """
        Initialize memory system.

        Args:
            workspace: Path to workspace directory containing memory files
        """
        self.workspace = workspace
        self.daily_dir = workspace / 'daily'

        # ensure directories exist
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def get_soul(self) -> str:
        """
        Read SOUL.md - personality and behavioral guidelines.

        Returns:
            Content of SOUL.md, or empty string if not found
        """
        return self._read_file('SOUL.md')

    def get_user(self) -> str:
        """
        Read USER.md - information about the user.

        Returns:
            Content of USER.md, or empty string if not found
        """
        return self._read_file('USER.md')

    def get_memory(self) -> str:
        """
        Read MEMORY.md - long-term facts and decisions.

        Returns:
            Content of MEMORY.md, or empty string if not found
        """
        return self._read_file('MEMORY.md')

    def get_daily(self, date: Optional[datetime] = None) -> str:
        """
        Read daily log for a specific date.

        Args:
            date: Date to read (defaults to today)

        Returns:
            Content of daily log, or empty string if not found
        """
        if date is None:
            date = datetime.now()

        filename = f'{date.strftime("%Y-%m-%d")}.md'
        return self._read_file(f'daily/{filename}')

    def get_recent_daily(self, days: int = 2) -> str:
        """
        Read recent daily logs (today + N previous days).

        Args:
            days: Number of days to include (default 2 = today + yesterday)

        Returns:
            Combined daily logs with date headers
        """
        logs = []
        now = datetime.now()

        for i in range(days):
            date = now - timedelta(days=i)
            content = self.get_daily(date)
            if content:
                date_str = date.strftime('%Y-%m-%d')
                logs.append(f'# {date_str}\n\n{content}')

        # reverse so oldest is first
        return '\n\n---\n\n'.join(reversed(logs))

    def append_memory(self, content: str) -> None:
        """
        Append content to MEMORY.md with timestamp.

        Args:
            content: Text to append
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = f'[{timestamp}]\n{content}'
        self._append_file('MEMORY.md', entry)
        print(f'[Stasis] Memory saved to MEMORY.md')

    def append_daily(self, content: str, date: Optional[datetime] = None) -> None:
        """
        Append content to daily log with timestamp.

        Args:
            content: Text to append
            date: Date for the log (defaults to today)
        """
        if date is None:
            date = datetime.now()

        timestamp = date.strftime('%H:%M')
        entry = f'[{timestamp}]\n{content}'

        filename = f'{date.strftime("%Y-%m-%d")}.md'
        self._append_file(f'daily/{filename}', entry)
        print(f'[Stasis] Daily log updated: {filename}')

    def build_context(self) -> str:
        """
        Build full memory context for system prompt.

        Combines SOUL, USER, MEMORY, and recent daily logs.

        Returns:
            Formatted context string
        """
        sections = []

        soul = self.get_soul()
        if soul:
            sections.append(f'# SOUL\n\n{soul}')

        user = self.get_user()
        if user:
            sections.append(f'# USER\n\n{user}')

        memory = self.get_memory()
        if memory:
            sections.append(f'# LONG-TERM MEMORY\n\n{memory}')

        recent = self.get_recent_daily(days=2)
        if recent:
            sections.append(f'# RECENT ACTIVITY\n\n{recent}')

        return '\n\n---\n\n'.join(sections)

    def _read_file(self, relative_path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            relative_path: Path relative to workspace

        Returns:
            File content, or empty string if not found
        """
        path = self.workspace / relative_path
        if path.exists():
            return path.read_text(encoding='utf-8').strip()
        return ''

    def _append_file(self, relative_path: str, content: str) -> None:
        """
        Append content to a file in the workspace.

        Creates the file if it doesn't exist.

        Args:
            relative_path: Path relative to workspace
            content: Text to append
        """
        path = self.workspace / relative_path

        # ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # append with newline separation
        with path.open('a', encoding='utf-8') as f:
            f.write(f'\n{content}\n')
