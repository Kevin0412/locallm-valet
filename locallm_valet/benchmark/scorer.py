# -*- coding: utf-8 -*-
"""Scorer — extract answer from model response and compare to ground truth."""

from __future__ import annotations

import logging
import re
from typing import Optional

from .schema import BenchmarkResult

logger = logging.getLogger("locallm_valet.benchmark.scorer")


def score_result(result: BenchmarkResult) -> None:
    """Score a single BenchmarkResult in-place, setting extracted_answer and is_correct.

    Supports multiple strategies per question category:
    - Multiple-choice (A/B/C/D): extract the letter
    - Short answer (fact/reasoning/math): exact match or numeric tolerance
    - Chinese (zh): match keywords or exact answer
    - Instruction following: exact match (case-insensitive for text)
    - Coding: keyword or exact answer
    """
    text = (result.raw_response or "").strip()
    gt = result.item.ground_truth.strip()

    if not text:
        result.is_correct = False
        result.score_detail = "empty response"
        return

    cat = result.item.category

    # Code benchmarks (HumanEval / MBPP): execute the generated code against
    # the hidden tests in an isolated subprocess — pass@1 style.
    # NOTE: pass the RAW response (not .strip()'d) — the first line of a
    # function body is indented and must keep its whitespace.
    if cat == "coding" and result.item.meta:
        _score_code(result.raw_response or "", result)
        return

    # Route to appropriate scorer based on whether the item has choices (MCQ) or not
    if result.item.choices:
        _score_mcq(text, gt, result)
    elif cat == "math":
        _score_math_short(text, gt, result)
    elif cat == "instruction":
        _score_instr(text, gt, result)
    elif cat == "chinese":
        _score_chinese(text, gt, result)
    else:
        # Fallback: exact match (strip whitespace, case-insensitive)
        _score_exact(text, gt, result)


# ---------------------------------------------------------------------------
# MCQ: extract letter and match
# ---------------------------------------------------------------------------

def _score_mcq(text: str, gt: str, result: BenchmarkResult) -> None:
    """Extract answer letter from MCQ response."""
    extracted = _extract_mcq_answer(text)
    result.extracted_answer = extracted

    if extracted is None:
        result.is_correct = False
        result.score_detail = "no answer letter found"
        return

    # Check for known failure patterns: the model repeated the question or said "cannot determine"
    lowered = text.lower()
    failure_keywords = [
        "cannot determine", "cannot be determined", "not provided",
        "insufficient information", "i don't know", "i'm not sure",
        "none of the above",
    ]
    # Only flag as fail if it's actually wrong AND the model is uncertain
    if extracted != gt and any(kw in lowered for kw in failure_keywords):
        result.is_correct = False
        result.score_detail = f"extracted={extracted}, uncertain response"
        return

    result.is_correct = (extracted == gt)
    result.score_detail = f"extracted={extracted}, ground_truth={gt}"


def _extract_mcq_answer(text: str) -> Optional[str]:
    """Extract answer letter from model response.

    Strategies in order:
    1. "Answer: X" / "answer: X" / "answer is X" markers
    2. \\boxed{X}
    3. **X** (bold letter at end)
    4. Last standalone A/B/C/D on its own line near the end
    5. Whole response is just a single letter
    """
    text = text.strip()

    # Strategy 1: explicit answer markers
    patterns = [
        r'(?im)(?:correct\s+)?answer\s*(?:is|:)?\s*\*{0,2}\s*([A-Da-d])\b',
        r'(?im)(?:option|select|choose)\s*(?:is|:)?\s*\*{0,2}\s*([A-Da-d])\b',
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text))
        if matches:
            return matches[-1].group(1).upper()

    # Strategy 2: \\boxed{X}
    boxed = re.findall(r'\\boxed\{([A-Da-d])\}', text)
    if boxed:
        return boxed[-1].upper()

    # Strategy 3: bold letter
    bold = re.findall(r'\*\*([A-Da-d])\*\*', text)
    if bold:
        return bold[-1].upper()

    # Strategy 4: last standalone letter on its own line near the end
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        # "A", "A.", "A)" etc
        m = re.match(r'^\s*([A-Da-d])\s*[\.\)]?\s*$', line)
        if m:
            return m.group(1).upper()

    # Strategy 5: whole response is just a single letter
    if re.match(r'^[A-Da-d]$', text):
        return text.upper()

    return None


# ---------------------------------------------------------------------------
# Short-answer math scorer
# ---------------------------------------------------------------------------

def _score_math_short(text: str, gt: str, result: BenchmarkResult) -> None:
    """Score short-answer math: extract the numeric answer from text."""
    # Try to find a number in the response
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if not numbers:
        result.is_correct = False
        result.score_detail = "no number found in response"
        return

    # Use the last number (model's final answer)
    extracted = numbers[-1]
    result.extracted_answer = extracted

    # Compare as number (handle both int and float)
    try:
        ext_val = float(extracted)
        gt_val = float(gt)
    except ValueError:
        result.is_correct = (extracted.strip() == gt.strip())
        result.score_detail = f"extracted={extracted}, ground_truth={gt}"
        return

    result.is_correct = abs(ext_val - gt_val) < 0.01
    result.score_detail = f"extracted={extracted} (numeric), ground_truth={gt}"


# ---------------------------------------------------------------------------
# Instruction following scorer
# ---------------------------------------------------------------------------

def _score_instr(text: str, gt: str, result: BenchmarkResult) -> None:
    """Score instruction following: exact match (for short answers) or keyword."""
    text_clean = text.strip().strip('"\'.,!?')

    # Exact match (case-insensitive for 'hello', 'bonjour', 'the sky is blue')
    if gt.lower() in ("hello", "bonjour", "the sky is blue", "42"):
        result.extracted_answer = text_clean
        result.is_correct = (text_clean.lower() == gt.lower())
        result.score_detail = f"extracted='{text_clean}', ground_truth='{gt}'"
        return

    # For 'def' (keyword), check if the keyword appears
    result.extracted_answer = text_clean
    result.is_correct = (gt.lower() in text_clean.lower())
    result.score_detail = f"extracted='{text_clean}', ground_truth='{gt}'"


# ---------------------------------------------------------------------------
# Chinese scorer
# ---------------------------------------------------------------------------

def _score_chinese(text: str, gt: str, result: BenchmarkResult) -> None:
    """Score Chinese questions: the MCQ path handles choice items; for free-text
    questions (e.g. zh_07), check if the expected answer string appears."""
    if result.item.choices:
        _score_mcq(text, gt, result)
        return

    # Free-text: check if the expected answer string appears in the response
    text_clean = text.strip().strip('"\'，。！？')
    result.extracted_answer = text_clean

    # Contains check (for short expected strings like "H2O")
    if gt in text_clean or gt in text:
        result.is_correct = True
        result.score_detail = f"ground_truth '{gt}' found in response"
    else:
        result.is_correct = False
        result.score_detail = f"ground_truth '{gt}' not found in response: {text_clean[:80]}"


# ---------------------------------------------------------------------------
# Generic exact-match fallback
# ---------------------------------------------------------------------------

def _score_exact(text: str, gt: str, result: BenchmarkResult) -> None:
    """Exact match fallback (case-insensitive, stripped)."""
    result.extracted_answer = text.strip()[:120]
    result.is_correct = (text.strip().lower() == gt.strip().lower())
    result.score_detail = f"exact match: '{text.strip()[:50]}' vs '{gt}'"


# ---------------------------------------------------------------------------
# Code execution scorer (HumanEval / MBPP) — pass@1 via subprocess
# ---------------------------------------------------------------------------

def _score_code(text: str, result: BenchmarkResult) -> None:
    """Run the model-generated code against the item's hidden tests in an
    isolated subprocess. pass@1: the single generated attempt must pass all
    tests. Never eval() in-process — the model output is untrusted."""
    import subprocess
    import sys
    import textwrap

    meta = result.item.meta
    generated = _extract_code(text)

    if not generated:
        result.is_correct = False
        result.score_detail = "no code block found in response"
        return

    try:
        if "entry_point" in meta:  # HumanEval
            entry = meta["entry_point"]
            # The model may return the full function (with its own `def` +
            # imports) or only the body. If it already defines the entry
            # point, use the model's code as-is; otherwise prepend the prompt
            # (which contains the def header) to the generated body.
            if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", generated, re.MULTILINE):
                full = generated
            else:
                full = meta["prompt"] + "\n" + generated
            test_code = meta["test"].replace("candidate", entry)
            harness = f"{full}\n\n{test_code}\n\ncheck({entry})"
        else:  # MBPP: generated solution + assert list
            setup = meta.get("test_setup", "") or ""
            asserts = "\n".join(meta.get("test_list", []))
            harness = f"{setup}\n\n{generated}\n\n{asserts}"

        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0:
            result.is_correct = True
            result.score_detail = "all tests passed"
        else:
            result.is_correct = False
            tail = (proc.stderr or "").strip().splitlines()
            result.score_detail = "tests failed: " + (tail[-1][:160] if tail else "rc!=0")
    except subprocess.TimeoutExpired:
        result.is_correct = False
        result.score_detail = "test execution timed out (>20s)"
    except Exception as exc:  # noqa: BLE001
        result.is_correct = False
        result.score_detail = f"harness error: {exc}"


def _extract_code(text: str) -> str:
    """Extract Python code from the model response.

    - Markdown fences (```python ... ```) → inner code.
    - Otherwise the response is used as-is (stripped of leading/trailing
      blank lines only, NEVER leading whitespace): for HumanEval the model
      returns the function BODY whose first line is indented (the prompt
      already contains ``def ...:``).
    """
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return text.strip("\n")