import json

import pytest

from src.search.llm_evolution.candidate_parser import (
    CandidateValidationError,
    parse_candidates,
)
from src.search.llm_evolution.sandbox import compile_expression


def valid_candidate(policy_id="safe"):
    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "name": policy_id,
        "initial_kernel_size": 2,
        "initial_score": {"feature": "fisher_score"},
        "add_score": {
            "op": "weighted_sum",
            "terms": [
                {"weight": 1.0, "expr": {"feature": "mutual_information"}},
                {"weight": 0.5, "expr": {"feature": "lp_activation"}},
            ],
        },
        "keep_score": {
            "op": "add",
            "args": [{"feature": "is_selected"}, {"feature": "abs_coefficient"}],
        },
        "target_kernel_size": {
            "op": "clip",
            "value": {
                "op": "add",
                "args": [{"search": "kernel_size"}, 1],
            },
            "lower": {"search": "feature_budget"},
            "upper": {"search": "total_features"},
        },
        "metadata": {},
    }


def test_valid_policy_compiles_deterministically():
    first = parse_candidates(json.dumps(valid_candidate()))[0]
    second = parse_candidates(json.dumps(valid_candidate()))[0]
    assert first.policy_hash == second.policy_hash


@pytest.mark.parametrize(
    "expression, message",
    [
        ({"op": "import", "args": []}, "unsafe or unknown"),
        ({"feature": "heldout_accuracy"}, "unknown feature signal"),
        ({"op": "add", "args": [{"value": 1}, {"filesystem": "/tmp"}]}, "must contain"),
    ],
)
def test_sandbox_rejects_unsafe_operations_and_unknown_signals(expression, message):
    with pytest.raises(ValueError, match=message):
        compile_expression(expression)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_nonfinite_json_numbers(constant):
    text = json.dumps(valid_candidate()).replace("2", constant, 1)
    with pytest.raises(CandidateValidationError, match="non-finite"):
        parse_candidates(text)


def test_parser_rejects_invalid_kernel_size_behavior():
    candidate = valid_candidate()
    candidate["target_kernel_size"] = {"value": 1}
    with pytest.raises(CandidateValidationError, match="target_kernel_size"):
        parse_candidates(json.dumps(candidate))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("initial_kernel_size", True, "positive integer"),
        ("initial_kernel_size", 2.5, "positive integer"),
        ("policy_id", 7, "must be strings"),
    ],
)
def test_parser_rejects_wrong_json_scalar_types(field, value, message):
    candidate = valid_candidate()
    candidate[field] = value
    with pytest.raises(CandidateValidationError, match=message):
        parse_candidates(json.dumps(candidate))
