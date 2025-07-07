from typing import List
import json
from json import JSONDecodeError
from .llm import LLMConfig

__all__ = ["classify_text", "get_model_response"]

def clean_json_markdown(resp: str) -> str:
    """
    Remove markdown-style code block if present.
    """
    return resp.strip().removeprefix("```json").removesuffix("```").strip()

def classify_text(resp: str) -> List[str]:
    """
    Parse model response and return list of category codes.

    Args:
        resp (str): Model response in JSON format (possibly with markdown wrapping)

    Returns:
        List[str]: Category codes such as ["IR", "WEC"]

    Raises:
        ValueError: If the response format is invalid
    """
    try:
        clean_resp = clean_json_markdown(resp)
        data = json.loads(clean_resp)
        return data["categories"]
    except (KeyError, JSONDecodeError) as e:
        raise ValueError(f"Invalid classification response format: {resp}") from e
    

def build_system_prompt(categories: str) -> str:
    """
    Build the system prompt including classification rules and categories.

    Args:
        categories (str): JSON string of available category definitions

    Returns:
        str: Full prompt text to guide the LLM
    """
    return f"""You are a classification assistant. Your task is to assign one or more category codes to a given text.

Use only the valid codes provided below, formatted like this:
{{"categories": ["IR", "WEC"]}}

Here are the available categories (in JSON format):
{categories}
"""

async def get_model_response(client: LLMConfig, text: str, categories: str) -> str:
    """
    Send a classification prompt to the LLM and return its raw response.

    Args:
        client (LLMConfig): LLM client configuration
        text (str): The text that needs to be classified
        categories (str): JSON string of all category definitions

    Returns:
        str: Raw response from the LLM (expected to be JSON)
    """
    system_prompt = build_system_prompt(categories)
    try:
        response = await client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to classify:\n{text}"},
            ],
            max_tokens=300,
            temperature=client.temperature,
        )
        return response.choices[0].message.content or '{"categories": []}'
    except Exception as e:
        print(f"API error: {e}")
        return '{"categories": []}'
