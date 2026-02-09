"""
Configuration management

Loads settings from environment variables and validates them.
Ensures all required settings are present before the app starts.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .models import ProviderType, get_model_config, ModelConfig


# load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    # provider settings
    provider: ProviderType = Field(default='anthropic', validation_alias='STASIS_PROVIDER')

    # anthropic
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias='ANTHROPIC_API_KEY')
    anthropic_model: str = Field(default='claude-sonnet-4-20250514', validation_alias='ANTHROPIC_MODEL')

    # openai
    openai_api_key: Optional[str] = Field(default=None, validation_alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4-turbo-preview', validation_alias='OPENAI_MODEL')

    # ollama
    ollama_base_url: str = Field(default='http://localhost:11434', validation_alias='OLLAMA_BASE_URL')
    ollama_model: str = Field(default='llama2', validation_alias='OLLAMA_MODEL')

    # workspace
    workspace: Path = Field(default=Path('./workspace'), validation_alias='STASIS_WORKSPACE')

    # optional overrides
    max_tokens: Optional[int] = Field(default=None, validation_alias='STASIS_MAX_TOKENS')

    # session history - number of exchanges to retain across restarts
    history_depth: int = Field(default=5, validation_alias='STASIS_HISTORY_DEPTH')

    @field_validator('workspace', mode='before')
    @classmethod
    def expand_workspace_path(cls, v: str | Path) -> Path:
        """Expand ~ and relative paths in workspace setting."""
        path = Path(v).expanduser().resolve()
        return path

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensure provider is one of the supported types."""
        valid = ['anthropic', 'openai', 'ollama']
        if v not in valid:
            raise ValueError(f'provider must be one of {valid}, got: {v}')
        return v

    def validate_credentials(self) -> None:
        """
        Validate that required credentials are present for the active provider.

        Raises ValueError if credentials are missing.
        """
        if self.provider == 'anthropic':
            if not self.anthropic_api_key:
                raise ValueError('ANTHROPIC_API_KEY required when using anthropic provider')
        elif self.provider == 'openai':
            if not self.openai_api_key:
                raise ValueError('OPENAI_API_KEY required when using openai provider')
        # ollama doesn't require API keys for local usage

    def get_active_model(self) -> str:
        """Get the model name for the current provider."""
        if self.provider == 'anthropic':
            return self.anthropic_model
        elif self.provider == 'openai':
            return self.openai_model
        else:
            return self.ollama_model

    def get_model_config(self) -> ModelConfig:
        """Get the full model configuration for the active provider."""
        model_name = self.get_active_model()
        return get_model_config(model_name, self.provider)

    def get_max_tokens(self) -> int:
        """Get max tokens, using override if set, otherwise model default."""
        if self.max_tokens:
            return self.max_tokens
        return self.get_model_config().max_tokens


# global settings instance
settings = Settings()

# validate credentials at import time
try:
    settings.validate_credentials()
except ValueError as e:
    print(f'[Stasis] Configuration error: {e}')
    print(f'[Stasis] Please check your .env file and ensure required credentials are set')
    raise
