import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# === 1. Timestamp and Path Config ===
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluate")
INPUT_MANUAL = os.path.join(EVAL_DIR, "Final_Data_Pilot_test_ea.xlsx")
OUTPUT_DIR = os.path.join(EVAL_DIR, "output")

# Output files with timestamp
DIFF_PATH = os.path.join(OUTPUT_DIR, f"differences_analysis_{TIMESTAMP}.xlsx")
PLOT_PATH = os.path.join(OUTPUT_DIR, f"accuracy_by_goal_{TIMESTAMP}.png")
LOG_PATH = os.path.join(OUTPUT_DIR, f"evaluate_accuracy_{TIMESTAMP}.log")

# Input LLM output from main.py
LLM_SOURCE_DIR = os.path.join(BASE_DIR, "output")

# === 2. Find Latest Output File ===
def find_latest_output_file(directory=LLM_SOURCE_DIR, prefix="output_classified_", ext=".xlsx"):
    print(f"🔍 Scanning directory: {directory}")
    try:
        all_files = os.listdir(directory)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Directory not found: {directory}")

    matched = [
        os.path.join(directory, f)
        for f in all_files
        if f.startswith(prefix) and f.endswith(ext)
    ]
    print(f"🧾 Matched files: {matched}")
    if not matched:
        raise FileNotFoundError("❌ No classified output files found.")
    return max(matched, key=os.path.getmtime)

# === 3. Helper Functions ===
def get_goal_column_pairs(df_llm, df_manual):
    auto_cols = [col for col in df_llm.columns if re.match(r"LPSgoal\d+_category", col)]
    pairs = []
    for auto_col in auto_cols:
        goal_num = re.search(r"LPSgoal(\d+)_category", auto_col).group(1)
        manual_col = f"LPSgoal{goal_num}_manual"
        if manual_col in df_manual.columns:
            pairs.append((auto_col, manual_col))
    return pairs

def parse_codes(code_str):
    if pd.isna(code_str):
        return set()
    return set([c.strip().upper() for c in str(code_str).split(",") if c.strip()])

def compare_exact_match(auto_set, manual_set):
    return auto_set == manual_set

# === 4. Evaluation ===
def evaluate_accuracy(df_llm, df_manual, column_pairs):
    results = []
    total_correct = 0
    total_total = 0
    mismatches = []

    for auto_col, manual_col in column_pairs:
        correct = 0
        total = 0

        for idx in tqdm(df_llm.index, desc=f"⏳ Evaluating {auto_col}"):
            auto_codes = parse_codes(df_llm.at[idx, auto_col])
            manual_codes = parse_codes(df_manual.at[idx, manual_col])

            if not manual_codes:
                continue

            total += 1
            if compare_exact_match(auto_codes, manual_codes):
                correct += 1
            else:
                mismatches.append({
                    "Row": idx,
                    "Goal": auto_col.replace("_category", ""),
                    "LLM Output": ", ".join(auto_codes),
                    "Manual Label": ", ".join(manual_codes),
                    "Text": df_llm.at[idx, auto_col.replace("_category", "_content")]
                })

        acc = correct / total if total > 0 else None
        results.append({
            "Goal": auto_col.replace("_category", ""),
            "Correct": correct,
            "Total": total,
            "Accuracy (%)": round(acc * 100, 2) if acc is not None else "N/A"
        })

        total_correct += correct
        total_total += total

    overall_acc = round(total_correct / total_total * 100, 2) if total_total > 0 else 0.0
    return results, overall_acc, mismatches

# === 5. Main Program ===
if __name__ == "__main__":
    print("📊 Evaluating LLM classification accuracy...")

    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        LLM_OUTPUT_PATH = find_latest_output_file()
        print(f"🗂 Using latest LLM output file: {LLM_OUTPUT_PATH}")
        df_llm = pd.read_excel(LLM_OUTPUT_PATH)
        df_manual = pd.read_excel(INPUT_MANUAL)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        exit(1)

    pairs = get_goal_column_pairs(df_llm, df_manual)
    if not pairs:
        print("❌ No matching goal columns found between LLM and manual data.")
        exit()

    results, overall, mismatches = evaluate_accuracy(df_llm, df_manual, pairs)

    print("\n✅ Accuracy Report:")
    for r in results:
        print(f"{r['Goal']}: {r['Accuracy (%)']}% ({r['Correct']}/{r['Total']})")
    print(f"\n🎯 Overall Accuracy: {overall}%")

    # Save mismatches if any
    if mismatches:
        pd.DataFrame(mismatches).to_excel(DIFF_PATH, index=False)
        print(f"\n❗ Mismatches saved to: {DIFF_PATH}")
    else:
        print("\n🎉 No mismatches found!")

    # Plot
    result_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Accuracy (%)", y="Goal", data=result_df, palette="Blues_d")
    plt.title("LLM Classification Accuracy by Goal", fontsize=14)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Goal")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"\n📊 Accuracy plot saved to: {PLOT_PATH}")

    # Log
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n🕓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"✅ Used LLM Output File: {LLM_OUTPUT_PATH}\n")
        f.write(f"📋 Manual Labels File: {INPUT_MANUAL}\n")
        f.write(f"🎯 Overall Accuracy: {overall}%\n")
        for r in results:
            f.write(f"  - {r['Goal']}: {r['Accuracy (%)']}% ({r['Correct']}/{r['Total']})\n")
        if mismatches:
            f.write(f"❗ Mismatches saved to: {DIFF_PATH} ({len(mismatches)} rows)\n")
        else:
            f.write("🎉 No mismatches found!\n")
    print(f"\n📝 Log saved to: {LOG_PATH}")

