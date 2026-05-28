"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
API_KEY = ""
MAX_HISTORY_TURNS = 3
MAX_BATCH_WORKERS = 4

_OPENAI_CLIENT = None
_OPENAI_CLIENT_FACTORY = None


def _get_openai_client() -> Any:
    from openai import OpenAI

    global _OPENAI_CLIENT, _OPENAI_CLIENT_FACTORY
    if _OPENAI_CLIENT is None or _OPENAI_CLIENT_FACTORY is not OpenAI:
        _OPENAI_CLIENT = OpenAI(api_key=API_KEY)
        _OPENAI_CLIENT_FACTORY = OpenAI
    return _OPENAI_CLIENT


def _estimate_output_cost(response_text: str, model: str) -> float:
    estimated_tokens = len(response_text.split()) / 0.75
    return estimated_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS[model]


def _truncate(value: Any, max_length: int = 40) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key="")
    """
    client = _get_openai_client()
    start_time = time.perf_counter()
    response = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
        max_retries=2,
        base_delay=0.2,
    )
    latency = time.perf_counter() - start_time
    response_text = (response.choices[0].message.content or "").strip()
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using gpt-4o-mini and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both gpt-4o and gpt-4o-mini with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpt4o_future = executor.submit(call_openai, prompt=prompt, model=OPENAI_MODEL)
        mini_future = executor.submit(call_openai_mini, prompt=prompt)
        gpt4o_response, gpt4o_latency = gpt4o_future.result()
        mini_response, mini_latency = mini_future.result()

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": _estimate_output_cost(gpt4o_response, OPENAI_MODEL),
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns: history = history[-3:]
    """
    history: list[dict[str, str]] = []
    client = _get_openai_client()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history[-MAX_HISTORY_TURNS * 2:],
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stream=True,
        )

        print("Assistant: ", end="", flush=True)
        assistant_chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            assistant_chunks.append(delta)
        print()

        history.append({"role": "assistant", "content": "".join(assistant_chunks)})
        history = history[-MAX_HISTORY_TURNS * 2:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    if not prompts:
        return []

    def compare_one(prompt: str) -> dict:
        result = dict(compare_models(prompt))
        result["prompt"] = prompt
        return result

    worker_count = min(MAX_BATCH_WORKERS, len(prompts))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(compare_one, prompts))


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    columns = [
        ("Prompt", 40),
        ("GPT-4o Response", 40),
        ("Mini Response", 40),
        ("GPT-4o Latency", 16),
        ("Mini Latency", 14),
    ]
    header = " | ".join(title.ljust(width) for title, width in columns)
    separator = "-+-".join("-" * width for _, width in columns)
    rows = [header, separator]

    for result in results:
        row_values = [
            _truncate(result.get("prompt", "")),
            _truncate(result.get("gpt4o_response", "")),
            _truncate(result.get("mini_response", "")),
            f"{float(result.get('gpt4o_latency', 0.0)):.3f}s",
            f"{float(result.get('mini_latency', 0.0)):.3f}s",
        ]
        rows.append(
            " | ".join(
                value.ljust(width)
                for value, (_, width) in zip(row_values, columns)
            )
        )

    return "\n".join(rows)


if "-" in __name__:
    _PATCHABLE_MODULE_NAME = "day01_lab_solution"
    sys.modules.setdefault(_PATCHABLE_MODULE_NAME, sys.modules[__name__])
    for _fn in (
        call_openai,
        call_openai_mini,
        compare_models,
        streaming_chatbot,
        retry_with_backoff,
        batch_compare,
        format_comparison_table,
    ):
        _fn.__module__ = _PATCHABLE_MODULE_NAME


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
