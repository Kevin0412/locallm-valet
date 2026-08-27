"""Code-execution scorer tests — verify the harness imports (typing/stdlib) and
pass@1 evaluation behave correctly."""

from locallm_valet.benchmark.schema import BenchmarkItem, BenchmarkResult
from locallm_valet.benchmark.scorer import _score_code


def _item(entry, test):
    return BenchmarkItem(item_id=f"he_{entry}", category="coding", question="Q",
                         ground_truth="", meta={"entry_point": entry, "test": test})


VAL_BODY_TPL = (
    "def {fn}(numbers: List[float], threshold: float) -> bool:\n"
    "    if len(numbers) < 2:\n        return False\n"
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)

# The hidden test defines `check(candidate)`; the harness appends `check(entry)`.
TEST_OK = (
    "def check(candidate):\n"
    "    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n"
    "    assert candidate([1.0, 1.5, 3.0], 0.6) is True\n"
)


def test_code_harness_typing_annotation_resolves():
    """A `List[...]` type hint in the signature must not NameError the harness."""
    item = _item("has_close_elements", TEST_OK)
    r = BenchmarkResult(item=item, model_name="m")
    r.raw_response = VAL_BODY_TPL.format(fn="has_close_elements")
    _score_code(r.raw_response, r)
    assert r.is_correct is True
    assert r.score_detail == "all tests passed"


def test_code_harness_rejects_wrong_answer():
    """An implementation that fails the hidden test must be scored False."""
    item = _item("has_close_elements", TEST_OK)
    r = BenchmarkResult(item=item, model_name="m")
    # Always returns False — fails the second assert (which wants True).
    r.raw_response = ("def has_close_elements(numbers, threshold) -> bool:\n"
                      "    if len(numbers) < 2:\n        return False\n"
                      "    return False\n")
    _score_code(r.raw_response, r)
    assert r.is_correct is False
    assert r.score_detail.startswith("tests failed")


def test_code_harness_mbpp_style():
    """MBPP harness (setup + solution + assert list) executes and scores."""
    item = BenchmarkItem(
        item_id="mbpp_1", category="coding", question="Q", ground_truth="",
        meta={"test_setup": "def add(a, b):\n    return a + b\n",
              "test_list": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]},
    )
    r = BenchmarkResult(item=item, model_name="m")
    r.raw_response = "def add(a, b):\n    return a + b\n"
    _score_code(r.raw_response, r)
    assert r.is_correct is True


# ---------------------------------------------------------------------------
# BFCL v3 AST key-argument scorer
# ---------------------------------------------------------------------------

from locallm_valet.benchmark.scorer import score_result


def _bfcl_item(subject, functions):
    return BenchmarkItem(item_id="bfcl_1", category=subject, question="Q",
                         ground_truth="(AST)",
                         meta={"functions": functions, "check": "ast"})


def _tc(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


_TRI_FUNCS = [{
    "name": "calculate_triangle_area", "description": "d",
    "parameters": {"type": "dict", "properties": {
        "base": {"type": "integer"}, "height": {"type": "integer"},
        "unit": {"type": "string"}},
        "required": ["base", "height"]},
}]


def test_bfcl_ast_simple_valid():
    r = BenchmarkResult(item=_bfcl_item("bfcl_simple", _TRI_FUNCS), model_name="m")
    r.tool_calls = [_tc("calculate_triangle_area", {"base": 10, "height": 5})]
    score_result(r)
    assert r.is_correct is True


def test_bfcl_ast_simple_missing_required():
    r = BenchmarkResult(item=_bfcl_item("bfcl_simple", _TRI_FUNCS), model_name="m")
    r.tool_calls = [_tc("calculate_triangle_area", {"base": 10})]  # missing height
    score_result(r)
    assert r.is_correct is False


def test_bfcl_ast_simple_wrong_type():
    r = BenchmarkResult(item=_bfcl_item("bfcl_simple", _TRI_FUNCS), model_name="m")
    r.tool_calls = [_tc("calculate_triangle_area", {"base": "abc", "height": 5})]
    score_result(r)
    assert r.is_correct is False
    assert "expected integer" in r.score_detail


def test_bfcl_ast_simple_requires_one_call():
    r = BenchmarkResult(item=_bfcl_item("bfcl_simple", _TRI_FUNCS), model_name="m")
    r.tool_calls = [
        _tc("calculate_triangle_area", {"base": 1, "height": 2}),
        _tc("calculate_triangle_area", {"base": 3, "height": 4}),
    ]
    score_result(r)
    assert r.is_correct is False


def test_bfcl_ast_unknown_function():
    r = BenchmarkResult(item=_bfcl_item("bfcl_simple", _TRI_FUNCS), model_name="m")
    r.tool_calls = [_tc("not_a_real_fn", {"x": 1})]
    score_result(r)
    assert r.is_correct is False
    assert "unknown function" in r.score_detail
