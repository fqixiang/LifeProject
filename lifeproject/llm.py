from dataclasses import dataclass
from typing import Dict, Optional
from openai import AsyncOpenAI

__all__ = ["LLMConfig"]

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    headers: Optional[Dict[str, str]] = None
    client: Optional[AsyncOpenAI] = None
