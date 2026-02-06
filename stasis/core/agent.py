"""
Agent logic for managing conversations with memory.

Handles message flow, memory extraction, and provider interaction.
"""

import re
from typing import List, Optional

from ..providers.base import Provider, Message
from .memory import Memory
from .prompt import build_system_prompt, build_checkin_prompt
from .search import MemorySearch, SearchResult


class Agent:
    """Main agent that orchestrates conversations with persistent memory."""

    def __init__(self, provider: Provider, memory: Memory, enable_search: bool = True):
        """
        Initialize the agent.

        Args:
            provider: LLM provider instance
            memory: Memory manager instance
            enable_search: Whether to enable memory search (default True)
        """
        self.provider = provider
        self.memory = memory
        self.conversation_history: List[Message] = []

        # initialize search engine
        self.search_enabled = enable_search
        if enable_search:
            self.search = MemorySearch(memory.workspace)
            # ensure index is up to date
            self.search.index_memory_file()
        else:
            self.search = None

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response with automatic memory saving.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response (with memory tags removed)
        """
        # add user message to history
        self.conversation_history.append(Message(role='user', content=user_message))

        # run search based on user's query to get relevant context
        search_results = None
        if self.search_enabled and self.search:
            print(f'[Stasis] Searching memory for: {user_message[:50]}...')
            search_results = self.search.search(user_message, top_k=5)
            if search_results:
                print(f'[Stasis] Found {len(search_results)} relevant memories')

        # build system prompt with search results (or full memory if no search)
        system_prompt = build_system_prompt(self.memory, search_results=search_results)

        # get model config for max tokens
        from ..config import settings
        max_tokens = settings.get_max_tokens()

        try:
            # get response from provider
            raw_response = self.provider.chat(
                messages=self.conversation_history,
                system=system_prompt,
                max_tokens=max_tokens,
            )

            # extract and save memories
            cleaned_response = self._process_memory_tags(raw_response)

            # add assistant response to history (use cleaned version)
            self.conversation_history.append(
                Message(role='assistant', content=cleaned_response)
            )

            return cleaned_response

        except Exception as e:
            error_msg = f'Failed to get response from {self.provider.get_provider_name()}: {e}'
            print(f'[Stasis] Error: {error_msg}')
            raise

    def checkin(self, checkin_type: str = 'daily') -> str:
        """
        Generate a proactive check-in message.

        Args:
            checkin_type: Type of check-in ('daily', 'evening', etc.)

        Returns:
            The check-in message
        """
        # build check-in specific prompt
        system_prompt = build_checkin_prompt(self.memory, checkin_type)

        # simple message to trigger check-in
        messages = [Message(role='user', content='Generate check-in message')]

        from ..config import settings
        max_tokens = settings.get_max_tokens()

        try:
            response = self.provider.chat(
                messages=messages,
                system=system_prompt,
                max_tokens=max_tokens,
            )

            # save that we did a check-in
            self.memory.append_daily(f'Sent {checkin_type} check-in to user')

            return response

        except Exception as e:
            error_msg = f'Failed to generate check-in: {e}'
            print(f'[Stasis] Error: {error_msg}')
            raise

    def clear_history(self) -> None:
        """Clear the conversation history (keeps memory intact)."""
        self.conversation_history = []
        print('[Stasis] Conversation history cleared')

    def _process_memory_tags(self, response: str) -> str:
        """
        Extract and save memory tags from response.

        Args:
            response: Raw response from LLM

        Returns:
            Response with memory tags removed
        """
        cleaned = response

        # extract and save <save_memory> tags
        memory_pattern = r'<save_memory>(.*?)</save_memory>'
        memory_matches = re.findall(memory_pattern, cleaned, re.DOTALL)
        for match in memory_matches:
            content = match.strip()
            if content:
                preview = content[:50] + '...' if len(content) > 50 else content
                print(f'[Stasis] Extracting memory: {preview}')
                self.memory.append_memory(content)

        # extract and save <save_daily> tags
        daily_pattern = r'<save_daily>(.*?)</save_daily>'
        daily_matches = re.findall(daily_pattern, cleaned, re.DOTALL)
        for match in daily_matches:
            content = match.strip()
            if content:
                preview = content[:50] + '...' if len(content) > 50 else content
                print(f'[Stasis] Extracting daily: {preview}')
                self.memory.append_daily(content)

        # remove all memory tags from response
        cleaned = re.sub(memory_pattern, '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(daily_pattern, '', cleaned, flags=re.DOTALL)

        # clean up any extra whitespace left behind
        cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned
