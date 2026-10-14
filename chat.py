
import math
import json
import ollama
import numpy as np
import sympy as sp

# ==========================================
# CONFIGURATION
# ==========================================

MODEL = "qwen3:8b"
response_mode = "short"

SYSTEM_PROMPT = """
You are an offline Data Science AI assistant.

You specialize in:
- Statistics and probability
- Entropy, Gini and information gain
- Linear and logistic regression
- KNN, SVM, LDA and QDA
- Decision trees
- Data preprocessing
- Python programming
- Machine learning evaluation

RULES:
1. Always respond in English.
2. Use plain text suitable for a terminal.
3. Never use LaTeX or Markdown formatting.
4. Keep answers concise.
5. Show formulas and essential calculations.
6. Use available tools for numerical calculations.
7. Never invent tool results.
8. Keep full precision in intermediate calculations.
9. Round only final results to 3 decimals.
10. Clearly identify the final answer.
11. If a tool fails, explain the limitation.
12. Do not claim calculations were verified
    unless a tool actually verified them.


FORMATTING RULES:
- Never output LaTeX, Markdown, $$, or backslashes.
- Use clean, readable mathematical notation.
- Use section titles in uppercase.
- Separate formulas, substitutions, and results.
- Use H(S) = -sum[p(i) * log2(p(i))] for entropy.
- Use parentheses and brackets for clarity.
- Never display rounded intermediate values
  as if they were used to compute final results.
- Display final results rounded to 3 decimals.

"""

SHORT_PROMPT = """
Keep answers short and direct.
For simple questions, use 1-3 sentences.
For exercises, show essential steps and results.
"""

DETAIL_PROMPT = """
Explain the solution step by step.
Show formulas, substitutions and calculations.
Keep the explanation organized.
"""


messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


# ==========================================
# MATHEMATICAL TOOLS
# ==========================================

def entropy(counts: list[int]) -> dict:
    """
    Calculate Shannon entropy from class counts.
    Example: [6, 4]
    """
    if not counts or any(
        x < 0 or int(x) != x for x in counts
    ):
        raise ValueError("Counts must be nonnegative integers.")

    total = sum(counts)

    if total == 0:
        raise ValueError("Total count must be positive.")

    probabilities = [x / total for x in counts]

    result = -sum(
        p * math.log2(p)
        for p in probabilities if p > 0
    )

    return {
        "counts": counts,
        "probabilities": probabilities,
        "entropy_full_precision": result,
        "entropy_rounded": round(result, 3)
    }


def gini_index(counts: list[int]) -> dict:
    """
    Calculate Gini impurity.
    Example: [6, 4]
    """
    if not counts or any(
        x < 0 or int(x) != x for x in counts
    ):
        raise ValueError("Counts must be nonnegative integers.")

    total = sum(counts)

    if total == 0:
        raise ValueError("Total count must be positive.")

    probabilities = [x / total for x in counts]

    result = 1 - sum(p ** 2 for p in probabilities)

    return {
        "gini_full_precision": result,
        "gini_rounded": round(result, 3)
    }


def information_gain(
    parent: list[int],
    children: list[list[int]]
) -> dict:
    """
    Calculate entropy-based information gain.

    Example:
    parent = [6, 4]
    children = [[4, 1], [2, 3]]
    """
    if not children:
        raise ValueError("At least one child is required.")

    if any(len(child) != len(parent) for child in children):
        raise ValueError("Class dimensions must match.")

    for i in range(len(parent)):
        if sum(child[i] for child in children) != parent[i]:
            raise ValueError(
                "Children must sum to parent counts."
            )

    parent_result = entropy(parent)
    parent_h = parent_result["entropy_full_precision"]

    total = sum(parent)
    weighted = 0.0
    child_results = []

    for child in children:
        size = sum(child)

        if size == 0:
            child_h = 0.0
        else:
            child_h = entropy(child)["entropy_full_precision"]

        weight = size / total
        weighted += weight * child_h

        child_results.append({
            "counts": child,
            "weight": weight,
            "entropy": child_h
        })

    gain = parent_h - weighted

    return {
        "parent_entropy": parent_h,
        "children": child_results,
        "weighted_entropy": weighted,
        "information_gain": gain,
        "rounded_results": {
            "parent_entropy": round(parent_h, 3),
            "child_entropies": [
                round(c["entropy"], 3)
                for c in child_results
            ],
            "weighted_entropy": round(weighted, 3),
            "information_gain": round(gain, 3)
        }
    }


def descriptive_statistics(values: list[float]) -> dict:
    """
    Calculate descriptive statistics.
    Uses sample variance and standard deviation.
    """
    if not values:
        raise ValueError("Provide at least one value.")

    data = np.array(values, dtype=float)

    if not np.all(np.isfinite(data)):
        raise ValueError("All values must be finite.")

    result = {
        "count": len(data),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data))
    }

    if len(data) >= 2:
        result["sample_variance"] = float(
            np.var(data, ddof=1)
        )
        result["sample_std"] = float(
            np.std(data, ddof=1)
        )
    else:
        result["sample_variance"] = None
        result["sample_std"] = None

    return result


def solve_equation(
    coefficients: list[float],
    constant: float
) -> dict:
    """
    Solve a polynomial equation.

    Coefficients are ordered highest degree first.

    Example:
    [3, 7] and 22 means 3*x + 7 = 22.
    """
    if len(coefficients) < 2:
        raise ValueError("Provide polynomial coefficients.")

    x = sp.Symbol("x")

    degree = len(coefficients) - 1

    expression = sum(
        sp.Float(coef) * x ** (degree - i)
        for i, coef in enumerate(coefficients)
    )

    solutions = sp.solve(
        sp.Eq(expression, sp.Float(constant)),
        x
    )

    return {
        "equation": str(sp.Eq(expression, constant)),
        "solutions": [str(s) for s in solutions]
    }


# ==========================================
# TOOL REGISTRY
# ==========================================

AVAILABLE_TOOLS = {
    "entropy": entropy,
    "gini_index": gini_index,
    "information_gain": information_gain,
    "descriptive_statistics": descriptive_statistics,
    "solve_equation": solve_equation
}


# ==========================================
# CONVERSATION FUNCTIONS
# ==========================================

def reset_conversation():
    messages.clear()
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })


def process_question(user_input):
    mode_prompt = (
        SHORT_PROMPT
        if response_mode == "short"
        else DETAIL_PROMPT
    )

    messages.append({
        "role": "user",
        "content": user_input
    })

    chat_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n" + mode_prompt
        }
    ] + messages[1:]

    # Tool-calling loop
    for iteration in range(5):

        response = ollama.chat(
            model=MODEL,
            messages=chat_messages,
            tools=list(AVAILABLE_TOOLS.values()),
            stream=False,
            think=False,
            options={
                "num_predict": 1000,
                "temperature": 0.1,
                "num_ctx": 4096
            }
        )

        assistant_message = response.message
        chat_messages.append(assistant_message)

        tool_calls = assistant_message.tool_calls or []

        # No tools requested: final answer
        if not tool_calls:
            answer = assistant_message.content or ""

            if not answer.strip():
                answer = "No answer was generated."

            print("\nAI:\n" + answer)

            messages.append({
                "role": "assistant",
                "content": answer
            })

            return

        # Execute requested tools
        for call in tool_calls:

            function_name = call.function.name
            arguments = call.function.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n[Calculating: {function_name}]")

            try:
                function = AVAILABLE_TOOLS.get(function_name)

                if function is None:
                    raise ValueError(
                        f"Unknown tool: {function_name}"
                    )

                result = function(**arguments)

                tool_output = json.dumps(
                    result,
                    default=str
                )

            except Exception as error:
                tool_output = json.dumps({
                    "error": str(error)
                })

            chat_messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": tool_output
            })

    print("\nAI: Calculation limit reached.")
    messages.pop()


# ==========================================
# MAIN CHATBOT
# ==========================================

def main():
    global response_mode

    print("=" * 50)
    print("          DATA SCIENCE CHATBOT")
    print("=" * 50)
    print("Model:", MODEL)
    print("Mode:", response_mode)
    print()
    print("Commands:")
    print("  short  - Short answers")
    print("  detail - Detailed explanations")
    print("  clear  - Clear conversation")
    print("  exit   - Quit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command == "exit":
                print("\nGoodbye!")
                break

            if command == "clear":
                reset_conversation()
                print("\nConversation cleared.")
                continue

            if command == "short":
                response_mode = "short"
                print("\nMode changed to SHORT.")
                continue

            if command == "detail":
                response_mode = "detail"
                print("\nMode changed to DETAIL.")
                continue

            process_question(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        except Exception as error:
            if messages and messages[-1]["role"] == "user":
                messages.pop()

            print("\nError:", error)


# ==========================================
# START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
