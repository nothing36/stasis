"""
Anthropic Claude provider implementation.

Uses the official Anthropic Python SDK to interact with Claude models.
As of version 0.1.0 this is the main implementation.
"""

from typing import List

from anthropic import Anthropic

from .base import Provider, Message


class AnthropicProvider(Provider):
    """Provider for Anthropic Claude models."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., 'claude-sonnet-4-20250514')
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: List[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send messages to Claude and get a response.

        Args:
            messages: Conversation history
            system: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            Claude's response text

        Raises:
            Exception: If the API call fails
        """
        # convert our Message objects to Anthropic's format
        anthropic_messages = [
            {'role': msg.role, 'content': msg.content}
            for msg in messages
        ]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system if system else '',
                messages=anthropic_messages,
            )

            # extract text from response
            return response.content[0].text

        except Exception as e:
            print(f'[Stasis] Anthropic API error: {e}')
            raise

    def get_provider_name(self) -> str:
        """Return provider name for logging."""
        return f'anthropic/{self.model}'
