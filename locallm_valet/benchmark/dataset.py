# -*- coding: utf-8 -*-
"""Built-in benchmark dataset — self-contained, no HF download required.

~80 curated items covering fact, reasoning, math, zh, instruction, coding.
Each item is a simple question with a gold-standard answer that small local
models should be capable of. Use different model quantisations on the same
items to detect capability degradation.
"""

from __future__ import annotations

from typing import Optional

from .schema import BenchmarkItem


# ---------------------------------------------------------------------------
# Item builders — keep the dataset inline so it's zero-dependency
# ---------------------------------------------------------------------------

def _mcq(item_id: str, category: str, question: str,
         options: list[str], answer_letter: str, language: str = "en") -> BenchmarkItem:
    """Build a multiple-choice item. Options already prefixed 'A. ' / 'B. ' / …"""
    choices_str = "\n".join(options)
    prompt = f"{question}\n\n{choices_str}\n\nAnswer with the letter (A/B/C/D) only."
    return BenchmarkItem(
        item_id=item_id, category=category,
        question=prompt, ground_truth=answer_letter.upper(),
        choices=options, language=language,
    )


def _qa(item_id: str, category: str, question: str,
        ground_truth: str, language: str = "en") -> BenchmarkItem:
    """Build a short-answer item."""
    return BenchmarkItem(
        item_id=item_id, category=category,
        question=question, ground_truth=ground_truth,
        language=language,
    )


# ---------------------------------------------------------------------------
# The full built-in dataset
# ---------------------------------------------------------------------------

def get_builtin_dataset() -> list[BenchmarkItem]:
    """Return ~80 curated benchmark items, zero download needed."""

    items: list[BenchmarkItem] = []

    # === Factual knowledge (22 items, en) ===
    items.append(_mcq("fact_01", "fact",
        "What is the capital of France?",
        ["A. Berlin", "B. Madrid", "C. Paris", "D. Rome"], "C"))
    items.append(_mcq("fact_02", "fact",
        "Which planet is known as the Red Planet?",
        ["A. Venus", "B. Mars", "C. Jupiter", "D. Saturn"], "B"))
    items.append(_mcq("fact_03", "fact",
        "Who wrote Romeo and Juliet?",
        ["A. Charles Dickens", "B. William Shakespeare", "C. Mark Twain", "D. Jane Austen"], "B"))
    items.append(_mcq("fact_04", "fact",
        "What is the chemical symbol for water?",
        ["A. H2O", "B. CO2", "C. NaCl", "D. O2"], "A"))
    items.append(_mcq("fact_05", "fact",
        "What is the largest ocean on Earth?",
        ["A. Atlantic", "B. Indian", "C. Arctic", "D. Pacific"], "D"))
    items.append(_mcq("fact_06", "fact",
        "In what year did World War II end?",
        ["A. 1943", "B. 1944", "C. 1945", "D. 1946"], "C"))
    items.append(_mcq("fact_07", "fact",
        "What is the freezing point of water in Celsius?",
        ["A. 0°C", "B. 32°C", "C. 100°C", "D. -10°C"], "A"))
    items.append(_mcq("fact_08", "fact",
        "Which gas do plants absorb from the atmosphere?",
        ["A. Oxygen", "B. Nitrogen", "C. Carbon dioxide", "D. Hydrogen"], "C"))
    items.append(_mcq("fact_09", "fact",
        "What is the tallest mountain on Earth?",
        ["A. K2", "B. Mount Everest", "C. Mount Fuji", "D. Denali"], "B"))
    items.append(_mcq("fact_10", "fact",
        "Which element has atomic number 1?",
        ["A. Helium", "B. Lithium", "C. Hydrogen", "D. Carbon"], "C"))
    items.append(_mcq("fact_11", "fact",
        "What is the currency of Japan?",
        ["A. Yuan", "B. Won", "C. Yen", "D. Dollar"], "C"))
    items.append(_mcq("fact_12", "fact",
        "Which country invented paper?",
        ["A. India", "B. Egypt", "C. China", "D. Greece"], "C"))
    items.append(_mcq("fact_13", "fact",
        "What is the speed of light approximately?",
        ["A. 3×10⁶ m/s", "B. 3×10⁸ m/s", "C. 3×10¹⁰ m/s", "D. 3×10⁵ m/s"], "B"))
    items.append(_mcq("fact_14", "fact",
        "Which organ pumps blood in the human body?",
        ["A. Lung", "B. Liver", "C. Heart", "D. Kidney"], "C"))
    items.append(_mcq("fact_15", "fact",
        "Who developed the theory of general relativity?",
        ["A. Newton", "B. Einstein", "C. Galileo", "D. Hawking"], "B"))

    # === Reasoning (16 items, en) ===
    items.append(_mcq("reason_01", "reasoning",
        "All cats are mammals. Tiger is a cat. Is Tiger a mammal?",
        ["A. Yes", "B. No", "C. Cannot be determined", "D. Only if tiger is a pet"], "A"))
    items.append(_mcq("reason_02", "reasoning",
        "A is taller than B. B is taller than C. Who is the shortest?",
        ["A. A", "B. B", "C. C", "D. Cannot be determined"], "C"))
    items.append(_mcq("reason_03", "reasoning",
        "If you flip a fair coin three times, what is the probability of getting heads all three times?",
        ["A. 1/2", "B. 1/4", "C. 1/8", "D. 1/3"], "C"))
    items.append(_mcq("reason_04", "reasoning",
        "John is older than Mary. Mary is older than Tom. Who is the youngest?",
        ["A. John", "B. Mary", "C. Tom", "D. Cannot be determined"], "C"))
    items.append(_mcq("reason_05", "reasoning",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        ["A. $0.10", "B. $0.05", "C. $0.15", "D. $0.20"], "B"))
    items.append(_mcq("reason_06", "reasoning",
        "Which number comes next in the sequence: 2, 4, 6, 8, ?",
        ["A. 9", "B. 10", "C. 11", "D. 12"], "B"))
    items.append(_mcq("reason_07", "reasoning",
        "If all bloops are groops, and some groops are traangs, which of the following must be true?",
        ["A. Some bloops are traangs", "B. All groops are bloops",
         "C. No bloops are traangs", "D. All bloops are groops"], "D"))
    items.append(_qa("reason_08", "reasoning",
        "If you have three apples and eat two, how many apples do you have left? Answer with just the number.",
        "1"))

    # === Math (14 items, en) ===
    items.append(_mcq("math_01", "math",
        "What is 12 × 15?",
        ["A. 150", "B. 160", "C. 170", "D. 180"], "D"))
    items.append(_mcq("math_02", "math",
        "What is 144 ÷ 12?",
        ["A. 10", "B. 11", "C. 12", "D. 13"], "C"))
    items.append(_mcq("math_03", "math",
        "What is 25% of 200?",
        ["A. 25", "B. 40", "C. 50", "D. 75"], "C"))
    items.append(_mcq("math_04", "math",
        "Solve for x: 2x + 6 = 14",
        ["A. x=3", "B. x=4", "C. x=5", "D. x=6"], "B"))
    items.append(_mcq("math_05", "math",
        "What is the area of a square with side 7?",
        ["A. 14", "B. 28", "C. 42", "D. 49"], "D"))
    items.append(_mcq("math_06", "math",
        "What is 3² + 4²?",
        ["A. 12", "B. 25", "C. 7", "D. 14"], "B"))
    items.append(_qa("math_07", "math",
        "Calculate: 47 + 35. Answer with just the number.",
        "82"))
    items.append(_qa("math_08", "math",
        "If a train travels at 60 km/h for 2 hours, how far does it travel? Answer with just the number.",
        "120"))

    # === Chinese capability (14 items, zh) ===
    items.append(_mcq("zh_01", "chinese",
        "鲁迅的原名是什么？",
        ["A. 周作人", "B. 周树人", "C. 胡适", "D. 茅盾"], "B", "zh"))
    items.append(_mcq("zh_02", "chinese",
        "中国的首都是哪个城市？",
        ["A. 上海", "B. 广州", "C. 北京", "D. 深圳"], "C", "zh"))
    items.append(_mcq("zh_03", "chinese",
        "一年有多少个月？",
        ["A. 10个月", "B. 11个月", "C. 12个月", "D. 13个月"], "C", "zh"))
    items.append(_mcq("zh_04", "chinese",
        "太阳从哪个方向升起？",
        ["A. 西方", "B. 南方", "C. 北方", "D. 东方"], "D", "zh"))
    items.append(_mcq("zh_05", "chinese",
        "“举头望明月”的下一句是什么？",
        ["A. 低头思故乡", "B. 疑是地上霜", "C. 春风又绿江南岸", "D. 处处闻啼鸟"], "A", "zh"))
    items.append(_mcq("zh_06", "chinese",
        "中国最长的河流是哪条？",
        ["A. 黄河", "B. 长江", "C. 珠江", "D. 淮河"], "B", "zh"))
    items.append(_qa("zh_07", "chinese",
        "请用中文回答：水的化学式是什么？只回答化学式即可。",
        "H2O", "zh"))
    items.append(_mcq("zh_08", "chinese",
        "中华人民共和国的国庆日是几月几日？",
        ["A. 7月1日", "B. 8月1日", "C. 10月1日", "D. 1月1日"], "C", "zh"))

    # === Instruction following (10 items, en) ===
    items.append(_qa("instr_01", "instruction",
        "Reply with exactly one word: Hello.",
        "Hello"))
    items.append(_qa("instr_02", "instruction",
        "List ONLY the number 42 in your response, nothing else.",
        "42"))
    items.append(_qa("instr_03", "instruction",
        "Translate 'good morning' to French. Respond with only the translation.",
        "bonjour"))
    items.append(_qa("instr_04", "instruction",
        "Repeat exactly: The sky is blue.",
        "The sky is blue."))
    items.append(_mcq("instr_05", "instruction",
        "Choose the correct option.",
        ["A. This is correct", "B. This is wrong", "C. Not sure", "D. None of the above"], "A"))

    # === Coding knowledge (10 items, en) ===
    items.append(_mcq("code_01", "coding",
        "What does the Python function len() return?",
        ["A. The length of an iterable", "B. The square root of a number",
         "C. The type of an object", "D. The maximum value"], "A"))
    items.append(_mcq("code_02", "coding",
        "Which data structure uses FIFO ordering?",
        ["A. Stack", "B. Queue", "C. Tree", "D. Graph"], "B"))
    items.append(_mcq("code_03", "coding",
        "What is the time complexity of binary search?",
        ["A. O(n)", "B. O(log n)", "C. O(n²)", "D. O(n log n)"], "B"))
    items.append(_qa("code_04", "coding",
        "In Python, what keyword is used to define a function? Respond with just the keyword.",
        "def"))

    return items


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_BUILTIN_DATASETS = {
    "builtin": get_builtin_dataset,
}

# Aliases
_BUILTIN_DATASETS["builtin_qa"] = get_builtin_dataset
_BUILTIN_DATASETS["default"] = get_builtin_dataset


def get_dataset(name: str = "builtin") -> list[BenchmarkItem]:
    """Return a dataset by name. Raises KeyError if unknown."""
    builder = _BUILTIN_DATASETS.get(name)
    if builder is None:
        raise KeyError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return builder()


def list_datasets() -> list[str]:
    """List available dataset names."""
    return sorted(set(_BUILTIN_DATASETS.keys()))