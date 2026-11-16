# Data Science Chatbot

A local, terminal-based Data Science assistant powered by **Ollama** and **Qwen3 8B**. It accepts questions in **English or Spanish** and responds **in English**. After the initial installation and model download, the chatbot can run locally without an internet connection.

## Features

The main version provides two modes:

- **Text:** Answers in exactly three short English sentences (as instructed to the model).
- **Calc:** Provides short step-by-step numerical solutions, including formulas, substitutions, and calculations.

It includes Python calculation tools for Shannon entropy, Gini impurity, information gain, descriptive statistics, and polynomial equations. It uses NumPy and SymPy for calculations.

> **Note:** The alternate script included in this project uses `short` and `detail` commands instead of the Text/Calc menu. Both versions use the same dependencies and Ollama model.

## Requirements

- **Windows, macOS, or Linux** with a terminal.
- **Python 3.10 or later** (Python 3.12 recommended).
- **Ollama:** https://ollama.com/download
- Enough available memory and disk space to run the **qwen3:8b** model locally. Performance varies by hardware.

## Installation (Windows PowerShell)

### 1. Install Python and Ollama

Install Python from https://www.python.org/downloads/ and Ollama from https://ollama.com/download. During Python installation, enable **Add Python to PATH** if prompted.

Check the installations:

```powershell
python --version
ollama --version
```

### 2. Download the AI model

```powershell
ollama pull qwen3:8b
```

The download requires an internet connection only the first time. Check that the model is available:

```powershell
ollama list
```

### 3. Create a virtual environment (recommended)

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can skip it and use `.\.venv\Scripts\python.exe` instead of `python` in subsequent commands.

### 4. Install Python packages

```powershell
python -m pip install -r requirements.txt
```

Or install them directly:

```powershell
python -m pip install ollama numpy sympy
```

The built-in `math` and `json` modules require **no separate installation**.

### 5. Start Ollama

Make sure the Ollama app/server is running. If it is not running, start it in a separate terminal:

```powershell
ollama serve
```

If Ollama is already running, do not start a second server.

### 6. Run the chatbot

Save the main Python script as **`chatbot.py`** in the project folder, then run:

```powershell
python chatbot.py
```

If you saved the script under a different name, replace `chatbot.py` with that filename.

## How to use

For the main script, select a mode at startup:

```text
1. Text (exactly 3 short sentences)
2. Calc (brief step-by-step solution)
```

Useful commands:

| Command | Action |
| --- | --- |
| `menu` | Return to the mode selection menu |
| `clear` | Clear the conversation history |
| `exit` | Quit the chatbot |

Example questions:

```text
What is the difference between supervised and unsupervised learning?
Given class counts [6, 4], calculate entropy.
Calculate the mean and median of [4, 8, 6, 10, 12].
Solve 3x + 7 = 22.
```

For the **alternate script**, use `short`, `detail`, `clear`, or `exit` instead.

## Files

```text
project-folder/
├── chatbot.py         # Rename the main supplied Python script to this
├── requirements.txt   # Python dependencies
├── README.md          # Setup and usage instructions
└── .venv/             # Optional local virtual environment (do not commit)
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'ollama'`**: Run `python -m pip install -r requirements.txt` using the same Python interpreter that runs the script.

**`Failed to connect to Ollama`**: Open the Ollama app or run `ollama serve` in a separate terminal.

**`model 'qwen3:8b' not found`**: Run `ollama pull qwen3:8b`.

**`python` is not recognized**: Install Python and make sure it is on PATH. On Windows, you can also try `py` instead of `python`.

**Some statistical results are incomplete or use sample formulas**: The supplied main script's `descriptive_statistics` function calculates **sample** variance and standard deviation (`ddof=1`), and its display logic chooses one statistics branch at a time. Requests for *population* variance/standard deviation or several statistics in one question may therefore need code changes. Review answers before using them in coursework.

## Privacy and offline use

Questions and calculations are processed locally through Ollama when the local model is available and running. Initial downloads of Python packages, Ollama, and the model require internet access.
