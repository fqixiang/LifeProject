import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .llm import LLMConfig

load_dotenv()

class LLMConfigManager:
    """
    Factory for creating LLMConfig i nstances for different LLM providers.
    Currently only openai is supported.
    """

    @staticmethod
    def get_openai_config() -> LLMConfig:
        return LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),  # default to gpt-4o-mini
            temperature=0.0,  # suitable for classification tasks
        )

    @classmethod
    def get_config(cls, provider: str) -> LLMConfig:
        """
        Openai API is for production use.
        DeepSeek API is for me testing.
        """
        config_map = {
            "openai": cls.get_openai_config,
        }

        if provider not in config_map:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        config = config_map[provider]()
        config.client = AsyncOpenAI(
            base_url=config.base_url, api_key=config.api_key, default_headers=config.headers
        )
        return config