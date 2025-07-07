from typing import Dict, List
import json

from .llm import LLMConfig
from .prompt_builder import build_prompts

__all__ = ["get_batched_model_response", "classify_text_batch"]

# Function for cleaning up the Markdown format in responses
def clean_json_markdown(resp: str) -> str:
    """
    Remove markdown-style code block if present.
    """
    return resp.strip().removeprefix("```json").removesuffix("```").strip()

# Function for parsing classification result 
def classify_text_batch(resp: str) -> Dict[str, List[str]]:
    """
    Parse LLM response containing multiple goals and return their category codes per goal.
    Supports 3 formats:
    - {"goal_1": {"categories": [...]}}
    - {"goal_1": [...]}
    - {"goal_1": "LP-01"}  (fallback: single code)
    """
    import json
    from json import JSONDecodeError

    try:
        clean_resp = resp.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean_resp)

        # Handle double stringified JSON
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if not isinstance(parsed, dict):
            raise ValueError("Parsed content is not a dict.")

        result = {}

        for goal_key, value in parsed.items():
            if isinstance(value, dict) and "categories" in value:
                result[goal_key] = value["categories"]
            elif isinstance(value, list):
                result[goal_key] = value
            elif isinstance(value, str):
                result[goal_key] = [value]  # fallback
            else:
                print(f"⚠️ Unexpected format for {goal_key}: {value}")

        return result

    except (KeyError, JSONDecodeError, TypeError, ValueError) as e:
        print("❌ Failed to parse response:", e)
        raise ValueError(f"❌ Invalid batched classification format:\n{resp}") from e

# Function for calling model
async def get_batched_model_response(client: LLMConfig, goals_dict: Dict[str, str], system_prompt: str) -> str:
    from .prompt_builder import build_prompts

    system_prompt, user_prompt = build_prompts(goals_dict, system_prompt)

    print("\n📥 SYSTEM PROMPT:\n", system_prompt[:500])
    print("\n🗣️ USER PROMPT:\n", user_prompt[:1000])

    try:
        response = await client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=client.temperature,
        )
        result = response.choices[0].message.content or "{}"
        print("\n🤖 LLM RAW RESPONSE:\n", result[:1000])
        return result
    except Exception as e:
        print(f"❌ API error: {e}")
        return "{}"