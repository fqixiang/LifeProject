import asyncio
import os
import pandas as pd
import json
import re
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

from lifeproject import LLMConfigManager
from lifeproject.classifier import classify_text, get_model_response

print("🚀 Starting Life Goal Classification...")

# Loading environment variables
load_dotenv()

# Generating timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths
INPUT_PATH = "data/input_test.xlsx" # Change this to your input file path
OUTPUT_PATH = f"output/output_classified_{timestamp}.xlsx"
LOG_PATH = f"output/classification_log_{timestamp}.csv"
CATEGORIES_PATH = "categories.json"

# Reading classification standards
with open(CATEGORIES_PATH) as f:
    categories_json = json.load(f)
    categories_str = json.dumps(categories_json, indent=2)

# Constructing system prompt
system_prompt = f"""You are a classification assistant. Your task is to assign one or more category codes to a given text.

Use only the valid codes provided below, formatted like this:
{{"categories": ["IR", "WEC"]}}

Here are the available categories (in JSON format):
{categories_str}
"""

# Initializing LLM configuration
provider = os.getenv("LLM_PROVIDER", "openai")
config = LLMConfigManager.get_config(provider)

# Asynchronous classification function (with progress bar and logging)
async def classify_all_goals(df: pd.DataFrame, goal_columns: list, categories_str: str) -> pd.DataFrame:
    logs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Classifying"):
        for goal_col in goal_columns:
            goal_text = str(row[goal_col]).strip()
            log_entry = {
                "row_index": idx,
                "column_name": goal_col,
                "goal_text": goal_text,
                "gpt_response": "",
                "parsed_category": "",
                "error": ""
            }

            if goal_text and goal_text.lower() != "nan":
                try:
                    response = await get_model_response(config, goal_text, categories_str)
                    log_entry["gpt_response"] = response[:300]
                    categories = classify_text(response)
                    log_entry["parsed_category"] = ", ".join(categories)
                    df.at[idx, goal_col.replace("_content", "_category")] = log_entry["parsed_category"]
                except Exception as e:
                    log_entry["error"] = str(e)
                    log_entry["parsed_category"] = "Oth"
                    df.at[idx, goal_col.replace("_content", "_category")] = "Oth"
            else:
                log_entry["error"] = "Empty or missing text"
                df.at[idx, goal_col.replace("_content", "_category")] = ""

            logs.append(log_entry)

    # Saving logs
    log_df = pd.DataFrame(logs)
    os.makedirs("output", exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"📝 Log saved to {LOG_PATH}")

    return df

# Main program entry
if __name__ == "__main__":
    df = pd.read_excel(INPUT_PATH)

    # Automatically identify goal columns
    goal_columns = [col for col in df.columns if re.match(r"LPSgoal\d+_content", col)]
    goal_category_columns = [col.replace("_content", "_category") for col in goal_columns]

    # Adding empty category columns
    for col in goal_category_columns:
        if col not in df.columns:
            df[col] = ""

    # Running asynchronously
    df = asyncio.run(classify_all_goals(df, goal_columns, categories_str))

    # 🔄 Adjusting column order (category columns follow content columns)
    new_columns = []
    for col in df.columns:
        new_columns.append(col)
        if col in goal_columns:
            cat_col = col.replace("_content", "_category")
            if cat_col in df.columns:
                new_columns.append(cat_col)
    seen = set()
    ordered_columns = [c for c in new_columns if not (c in seen or seen.add(c))]
    df = df[ordered_columns]

    # Saving output
    os.makedirs("output", exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"✅ Classification complete. Results saved to: {OUTPUT_PATH}")
