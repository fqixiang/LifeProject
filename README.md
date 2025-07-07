# Life Goal Classification using LLMs

This project is designed to classify **life goals** based on a structured taxonomy using **large language models (LLMs)** via API calls. It reads life goal data from Excel files, classifies them into categories using an LLM (like GPT), and evaluates the classification accuracy against manually labeled data.

---

## 🗂️ Project Structure

```
FTOLP_LLM/
├── README.md
├── categories.json                                     # Life goal category taxonomy used by LLM
├── categories_prompt.txt                               # *Optional formatted prompt text for development(Not used)
├── data                                                # Input data folder
│   └── input_test.xlsx                                 # Excel file with life goals to classify
├── evaluate                                            # Manual labels and evaluation folder
│   ├── input_ea.xlsx                                   # Manually coded life goal categories
│   └── output                                          # Output of evaluation results folder
│       ├── accuracy_by_goal_20250630_122453.png        # Accuracy plots
│       ├── differences_analysis_20250630_122453.xlsx   # Mismatch result
│       └── evaluate_accuracy_20250630_122453.log       # Mismatch logs
├── evaluate_accuracy.py                                # Script to compare LLM output with manual labels
├── lifeproject                                         # Core Python package (classification logic & config)
│   ├── __init__.py                                     # Package initialization
│   ├── __pycache__                                     # (Generated) Cache directory folder
│   │   ├── __init__.cpython-313.pyc
│   │   ├── classifier.cpython-313.pyc
│   │   ├── config.cpython-313.pyc
│   │   └── llm.cpython-313.pyc
│   ├── classifier.py                                   # Main classification logic using LLM(Prompt)
│   ├── classifier_batched.py                           # *Main classification logic using LLM for main_batched.py(not run yet)
│   ├── prompt_builder.py                               # *System prompt for main_batched.py(not run yet)
│   ├── config.py                                       # LLM config management (loads .env)
│   └── llm.py                                          # LLM config dataclass and OpenAI interface
├── main.py                                             # Main script to classify life goals in an Excel file
├── main_batched.py                                     # *Main script to classify life goals in an Excel file using batched LLM requests(not run yet)
├── output                                              # Output data folder
│   ├── classification_log.csv                          # Test result log 1
│   ├── classification_log_20250630_103327.csv          # Test result log 2
│   ├── output_classified.xlsx                          # Test result 1
│   └── output_classified_20250630_103327.xlsx          # Test result 2
├── pyproject.toml                                      # Project metadata and configuration
├── requirements.in                                     # Editable dependency list
├── system_prompt.txt                                   # *System prompt template for guiding LLM(Not used)
├── uv.lock                                             # (Generated) Locked dependency versions
└── .env                                                # Environment variables (API key, model name)
```

---

## 🛠️ How to Run the Project

### 1. Set up the environment

Create a virtual environment and install dependencies:

```bash
uv venv
uv pip compile requirements.in --output uv.lock
uv pip sync uv.lock
```

Alternatively:
```bash
uv sync
```

> Make sure you have [`uv`](https://github.com/astral-sh/uv) installed beforehand, installation process is as follows:
    
Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux/MacOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
---

### 2. Add your API keys
Create a `.env` file in the project root directory based on this template:
```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
LLM_PROVIDER=openai
```

---

### 3. Run the classification

```bash
uv run python main_json.py
```

This will:

- Load input Excel file (`data/input_test.xlsx`)
- Classify goals using LLM based on `categories.json`
- Save results to `output/output_classified_<timestamp>.xlsx`
- Log outputs and errors to `output/`

---

### 4. Evaluate classification accuracy (optional)

```bash
uv run python evaluate_accuracy.py
```

This script compares the LLM output with manually labeled data (`evaluate/input_ea.xlsx`), generates a bar plot of accuracies, and outputs mismatch details.

---

## 📌 Notes

- All classification behavior is driven by the `categories.json` file.
- Output and log files are timestamped for traceability.
- Alternative approach 1: Batch goals by ID to reduce token usage. (Coding completed, but the code has not been executed yet.)
- Alternative approach 2: Batch goals by ID and combine with RAG to reduce token usage. (Not yet started.)

---

## 👩‍💻 Author

Shiyu Dong  
s.dong1@uu.nl
Utrecht University | SaSR & SoDa