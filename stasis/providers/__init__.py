"""
Provider implementations for different LLM APIs.

Abstracts away provider-specific details so models can be swapped easily.
Still unsure if this will actually be used in practice...
"""

from .base import Provider, Message
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

__all__ = [
    'Provider',
    'Message',
    'AnthropicProvider',
    'OpenAIProvider',
    'OllamaProvider',
]