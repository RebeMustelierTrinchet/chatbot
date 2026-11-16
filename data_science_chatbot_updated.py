
import math
import json
import ollama
import numpy as np
import sympy as sp

# ==========================================
# CONFIGURATION
# ==========================================

MODEL = "qwen3:8b"
response_mode = "text"

SYSTEM_PROMPT = """
You are an offline Data Science assistant.
You understand questions in Spanish and English, but ALWAYS answer in English.
Use simple plain text suitable for a terminal, without Markdown or LaTeX.
Be accurate, direct, and extremely concise.
Never invent facts, computations, or tool results.
Round final numerical results to 3 decimal places when appropriate.
"""

TEXT_PROMPT = """
MODE: TEXT.
Answer the user's actual question in EXACTLY 3 short English sentences.
No lists, headings, formulas, extra sentences, or introductory filler.
Answer questions written in either Spanish or English in English only.
If you cannot answer, explain briefly within the 3-sentence limit.
"""

CALC_PROMPT = """
MODE: CALC.
Solve the user's requested math, statistics, or numerical problem.
Answer in English even when the question is in Spanish.
Show the shortest correct step-by-step solution:
1. Formula or equation.
2. Substitute the given values.
3. Compute and state the final answer.
If more steps are strictly necessary, include them, but keep every line brief.
Use plain text and show essential calculations only.
Keep full precision internally and round final results to 3 decimals when appropriate.
Use a suitable available calculation tool when it can verify the result.
If a tool cannot solve the problem, do not pretend it did.
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
    mode_prompt = TEXT_PROMPT if response_mode == "text" else CALC_PROMPT

    messages.append({"role": "user", "content": user_input})
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + mode_prompt}
    ] + messages[1:]

    # Text uses no calculation tools; Calc can call the functions above.
    for iteration in range(5):
        kwargs = {
            "model": MODEL,
            "messages": chat_messages,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 170 if response_mode == "text" else 500,
                "temperature": 0.0,
                "num_ctx": 4096
            }
        }
        if response_mode == "calc":
            kwargs["tools"] = list(AVAILABLE_TOOLS.values())

        response = ollama.chat(**kwargs)
        assistant_message = response.message
        chat_messages.append(assistant_message)
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            answer = (assistant_message.content or "").strip()
            if not answer:
                answer = "No answer was generated. Please try again."
            print("\nAI:\n" + answer)
            messages.append({"role": "assistant", "content": answer})
            return

        for call in tool_calls:
            function_name = call.function.name
            arguments = call.function.arguments
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                function = AVAILABLE_TOOLS.get(function_name)
                if function is None:
                    raise ValueError(f"Unknown tool: {function_name}")
                result = function(**arguments)
                tool_output = json.dumps(result, default=str)
            except Exception as error:
                tool_output = json.dumps({"error": str(error)})

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

def choose_mode():
    global response_mode
    while True:
        print("\nChoose a mode:")
        print("  1. Text (exactly 3 short sentences)")
        print("  2. Calc (brief step-by-step solution)")
        selection = input("Option (1/2): ").strip().lower()
        if selection in ("1", "text"):
            response_mode = "text"
            break
        if selection in ("2", "calc"):
            response_mode = "calc"
            break
        print("Enter 1 or 2.")
    reset_conversation()
    print("Mode:", response_mode.upper())


def main():
    print("=" * 44)
    print("           DATA SCIENCE CHATBOT")
    print("=" * 44)
    print("Model:", MODEL)
    print("Language: English output (Spanish/English input)")
    print("Commands: menu, clear, exit")
    choose_mode()

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            command = user_input.lower()
            if command in ("exit", "salir"):
                print("\nGoodbye!")
                break
            if command in ("menu", "mode", "modo"):
                choose_mode()
                continue
            if command == "clear":
                reset_conversation()
                print("\nConversation cleared.")
                continue
            process_question(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as error:
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            print("\nError:", error)


if __name__ == "__main__":
    main()
