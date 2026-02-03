"""
Model definitions and capabilities

Defines supported models across different providers with their
context windows, token limits, and feature support.
"""

from dataclasses import dataclass
from typing import Literal


ProviderType = Literal['anthropic', 'openai', 'ollama']


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ProviderType
    max_tokens: int
    context_window: int
    supports_streaming: bool = True
    supports_tools: bool = False


# anthropic models
CLAUDE_OPUS_4 = ModelConfig(
    name='claude-opus-4-20250514',
    provider='anthropic',
    max_tokens=4096,
    context_window=200000,
    supports_tools=True,
)

CLAUDE_SONNET_4 = ModelConfig(
    name='claude-sonnet-4-20250514',
    provider='anthropic',
    max_tokens=4096,
    context_window=200000,
    supports_tools=True,
)

CLAUDE_HAIKU_4 = ModelConfig(
    name='claude-haiku-4-20250514',
    provider='anthropic',
    max_tokens=4096,
    context_window=200000,
    supports_tools=True,
)

# openai models
GPT_4_TURBO = ModelConfig(
    name='gpt-4-turbo-preview',
    provider='openai',
    max_tokens=4096,
    context_window=128000,
    supports_tools=True,
)

GPT_4 = ModelConfig(
    name='gpt-4',
    provider='openai',
    max_tokens=4096,
    context_window=8192,
    supports_tools=True,
)

GPT_3_5_TURBO = ModelConfig(
    name='gpt-3.5-turbo',
    provider='openai',
    max_tokens=4096,
    context_window=16385,
    supports_tools=True,
)

# ollama models (common ones)
# note: ollama models vary widely in capability
LLAMA_2 = ModelConfig(
    name='llama2',
    provider='ollama',
    max_tokens=2048,
    context_window=4096,
    supports_tools=False,
)

LLAMA_3 = ModelConfig(
    name='llama3',
    provider='ollama',
    max_tokens=2048,
    context_window=8192,
    supports_tools=False,
)

MISTRAL = ModelConfig(
    name='mistral',
    provider='ollama',
    max_tokens=2048,
    context_window=8192,
    supports_tools=False,
)

# registry of all known models
MODEL_REGISTRY = {
    # anthropic
    'claude-opus-4-20250514': CLAUDE_OPUS_4,
    'claude-sonnet-4-20250514': CLAUDE_SONNET_4,
    'claude-haiku-4-20250514': CLAUDE_HAIKU_4,

    # openai
    'gpt-4-turbo-preview': GPT_4_TURBO,
    'gpt-4': GPT_4,
    'gpt-3.5-turbo': GPT_3_5_TURBO,

    # ollama
    'llama2': LLAMA_2,
    'llama3': LLAMA_3,
    'mistral': MISTRAL,
}


def get_model_config(model_name: str, provider: ProviderType) -> ModelConfig:
    """
    Get configuration for a model by name.

    If model not in registry, returns a default config for that provider.
    This allows using custom/new models without updating the registry.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # default configs for unknown models
    print(f'[Stasis] Warning: model {model_name} not in registry, using defaults')

    if provider == 'anthropic':
        return ModelConfig(
            name=model_name,
            provider='anthropic',
            max_tokens=4096,
            context_window=200000,
            supports_tools=True,
        )
    elif provider == 'openai':
        return ModelConfig(
            name=model_name,
            provider='openai',
            max_tokens=4096,
            context_window=8192,
            supports_tools=True,
        )
    else:  # ollama
        return ModelConfig(
            name=model_name,
            provider='ollama',
            max_tokens=2048,
            context_window=4096,
            supports_tools=False,
        )
