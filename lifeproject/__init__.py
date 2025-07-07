"""
lifeproject package

This package contains the core modules and logic for classifying life goals
using a predefined taxonomy and large language models (LLMs).
"""

# Expose key interfaces for external access
from .config import LLMConfigManager
from .llm import LLMConfig

__all__ = [
    "LLMConfig",
    "LLMConfigManager",
]

__version__ = "0.1.0"
__author__ = "Shiyu Dong"