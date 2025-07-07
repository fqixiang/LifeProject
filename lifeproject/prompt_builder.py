def build_prompts(goals_dict: dict, system_prompt: str) -> tuple[str, str]:
    """
    Build the system and user prompts for a given dictionary of goals.

    Args:
        goals_dict (dict): Dictionary mapping column names to goal text.
        system_prompt (str): The system prompt read from system_prompt.txt.

    Returns:
        tuple[str, str]: A tuple with system_prompt and user_prompt.
    """
    goals_text = "\n".join([f"{k}: {v}" for k, v in goals_dict.items()])
    user_prompt = (
        "The following are the life goals expressed by a person:\n\n"
        f"{goals_text}\n\n"
        "Please classify each goal based on the category definitions in the system prompt. "
        "Return a JSON dictionary with each goal column name as key and the list of matching category codes as values."
    )
    return system_prompt, user_prompt
