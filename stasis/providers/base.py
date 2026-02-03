"""
Base provider interface.

Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send messages to the LLM and get a response.

        Args:
            messages: Conversation history (user and assistant messages)
            system: System prompt to set context and behavior
            max_tokens: Maximum tokens in the response

        Returns:
            The assistant's response as a string

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider (for logging/debugging)."""
        pass
