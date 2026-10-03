"""Typed decision API using the provisioned scorers and published checkpoints."""

import json
import threading

import math


class InputError(ValueError):
    """The caller's typed decision request is invalid or unsupported.

    Callers can fix these; the message is safe to return to them.
    """


class UpstreamError(RuntimeError):
    """The provisioned scorer answered in a shape we cannot trust.

    Not the caller's fault: the service must fail loudly instead of reporting
    a probability distribution that does not add up.
    """


def checked_result(result, model, kind, options):
    if not isinstance(result, dict) or result.get("model") != model:
        raise UpstreamError("Unexpected upstream model identity")
    answer = result.get("answers", {}).get("answer", {})
    if answer.get("type") != kind:
        raise UpstreamError("Unexpected upstream answer type")
    if kind == "noul":
        values = {"true": answer.get("noul")}
    else:
        values = answer.get("probabilities")
        if not isinstance(values, dict) or set(values) != set(options):
            raise UpstreamError("Unexpected upstream probability labels")
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values.values()
    ):
        raise UpstreamError("Invalid upstream probabilities")
    if kind != "noul":
        if abs(sum(values.values()) - 1) > 1e-5:
            raise UpstreamError("Upstream probability mass does not sum to one")
        if kind == "choice":
            choice = answer.get("choice")
            if choice not in values or max(values.values()) - values[choice] > 1e-5:
                raise UpstreamError(
                    "Upstream choice differs from the probability maximum"
                )
        else:
            score = answer.get("score")
            expected = sum(
                int(key) * probability for key, probability in values.items()
            )
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or abs(score - expected) > 1e-4
            ):
                raise UpstreamError(
                    "Upstream Score differs from its probability distribution"
                )
    usage = result.get("usage", {})
    if not isinstance(usage, dict) or any(
        type(usage.get(key, 0)) is not int or usage.get(key, 0) < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise UpstreamError("Invalid upstream token usage")
    return usage


def _text(value, name):
    if isinstance(value, str):
        if not value.strip():
            raise InputError(f"{name} must be nonempty")
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise InputError(f"{name} must be a string, object or array")


class DecisionService:
    def __init__(self, model, upstream_model, preflight, predict, max_options):
        self.model = model
        self.upstream_model = upstream_model
        self.preflight = preflight
        self.predict = predict
        self.max_options = max_options
        self.lock = threading.Lock()

    def evaluate(self, payload):
        if not isinstance(payload, dict) or payload.get("model") != self.model:
            raise InputError("Request model must match the registered model name")
        if payload.get("stream"):
            raise InputError("Typed decisions do not support streaming")
        state = _text(payload.get("state"), "state")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
            raise InputError("questions must contain 1 to 32 named questions")
        compiled = []
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict):
                raise InputError("Questions require nonempty names and typed objects")
            instructions = _text(question.get("instructions"), "instructions")
            kind, criteria = question.get("type"), question.get("criteria")
            if kind == "noul":
                if criteria is None:
                    criteria = {"false": "No", "true": "Yes"}
                if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                    raise InputError("Noul criteria must contain false and true")
                options = {
                    key: _text(criteria[key], "criteria") for key in ("false", "true")
                }
            elif kind == "choice":
                if (
                    not isinstance(criteria, dict)
                    or not 2 <= len(criteria) <= self.max_options
                ):
                    raise InputError(
                        f"Choice requires 2 to {self.max_options} options"
                    )
                if any(not isinstance(key, str) or not key for key in criteria):
                    raise InputError("Choice option names must be nonempty strings")
                options = {
                    key: key if value is None else _text(value, "criteria")
                    for key, value in criteria.items()
                }
            elif kind == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise InputError("Score requires 2 to 10 ordered levels")
                options = {
                    str(index): _text(value, "criteria")
                    for index, value in enumerate(criteria)
                }
            else:
                raise InputError("Question type must be choice, noul or score")
            criteria = list(options.values()) if kind == "score" else options
            body = {
                "model": self.upstream_model,
                "state": state,
                "questions": {
                    "answer": {
                        "type": kind,
                        "instructions": instructions,
                        "criteria": criteria,
                    }
                },
            }
            compiled.append((name, kind, options, body))
        answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
        with self.lock:
            # Reject any invalid/oversized question before evaluating the first one.
            for _, _, _, body in compiled:
                try:
                    check = self.preflight(body)
                except ValueError as error:
                    # The upstream compiler says the input cannot be scored
                    # (usually the token limit): that is a caller problem.
                    raise InputError(
                        str(error) or "Upstream compiler rejects input length or shape"
                    ) from error
                if not check.get("supported"):
                    raise InputError("Upstream compiler rejects input length or shape")
            for name, kind, options, body in compiled:
                result = self.predict(body)
                checked_usage = checked_result(
                    result, self.upstream_model, kind, options
                )
                answers[name] = result["answers"]["answer"]
                for key in usage:
                    usage[key] += checked_usage.get(key, 0)
        return {"model": self.model, "answers": answers, "usage": usage}
