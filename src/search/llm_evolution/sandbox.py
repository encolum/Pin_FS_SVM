"""Bounded interpreter for the VeraPin JSON expression DSL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from ..states import FeatureState, SearchState


ALLOWED_OPERATIONS = {
    "weighted_sum",
    "add",
    "subtract",
    "multiply",
    "safe_divide",
    "min",
    "max",
    "clip",
    "conditional",
}

FEATURE_SIGNALS = {
    "index",
    "in_kernel",
    "is_selected",
    "abs_coefficient",
    "fisher_score",
    "mutual_information",
    "mean_abs_correlation",
    "max_abs_correlation",
    "lp_activation",
    "lp_abs_coefficient",
    "slack_association",
    "selection_frequency",
    "inactive_iterations",
    "kernel_age",
    "l1_abs_coefficient",
    "pin_abs_coefficient",
    "support_redundancy",
}

SEARCH_SIGNALS = {
    "iteration",
    "current_objective",
    "best_objective",
    "current_gap",
    "best_bound",
    "kernel_size",
    "feature_budget",
    "total_features",
    "stagnation_iterations",
    "elapsed_seconds",
    "remaining_seconds",
    "C",
    "tau",
    "improved_last_iteration",
}


@dataclass(frozen=True)
class CompiledExpression:
    expression: dict[str, Any]
    evaluator: Callable[[FeatureState | None, SearchState], float]

    def __call__(self, feature: FeatureState | None, search: SearchState) -> float:
        value = float(self.evaluator(feature, search))
        if not math.isfinite(value):
            raise ValueError("policy expression evaluated to NaN or infinity")
        return value


def compile_expression(
    expression: dict[str, Any],
    *,
    allow_feature_signals: bool = True,
    max_depth: int = 16,
    max_nodes: int = 256,
) -> CompiledExpression:
    """Validate and compile a bounded expression without executing generated code."""
    counter = [0]
    evaluator = _compile_node(
        expression,
        allow_feature_signals=allow_feature_signals,
        depth=0,
        max_depth=int(max_depth),
        max_nodes=int(max_nodes),
        counter=counter,
    )
    return CompiledExpression(expression=expression, evaluator=evaluator)


def _compile_node(
    node: Any,
    *,
    allow_feature_signals: bool,
    depth: int,
    max_depth: int,
    max_nodes: int,
    counter: list[int],
) -> Callable[[FeatureState | None, SearchState], float]:
    counter[0] += 1
    if counter[0] > max_nodes:
        raise ValueError(f"policy expression exceeds {max_nodes} nodes")
    if depth > max_depth:
        raise ValueError(f"policy expression exceeds depth {max_depth}")
    if isinstance(node, bool) or not isinstance(node, (dict, int, float)):
        raise ValueError("DSL nodes must be finite numbers or expression objects")
    if isinstance(node, (int, float)):
        value = _finite_number(node)
        return lambda feature, search, value=value: value
    if not isinstance(node, dict):
        raise ValueError("invalid DSL node")

    if "op" not in node and "value" in node:
        _require_keys(node, required={"value"})
        value = _finite_number(node["value"])
        return lambda feature, search, value=value: value
    if "op" not in node and "feature" in node:
        _require_keys(node, required={"feature"})
        signal = str(node["feature"])
        if not allow_feature_signals:
            raise ValueError("feature signals are not allowed in this expression")
        if signal not in FEATURE_SIGNALS:
            raise ValueError(f"unknown feature signal: {signal}")

        def feature_signal(feature: FeatureState | None, search: SearchState) -> float:
            if feature is None:
                raise ValueError(f"feature signal {signal!r} used without a feature")
            value = getattr(feature, signal)
            return float(value) if value is not None else 0.0

        return feature_signal
    if "op" not in node and "search" in node:
        _require_keys(node, required={"search"})
        signal = str(node["search"])
        if signal not in SEARCH_SIGNALS:
            raise ValueError(f"unknown search signal: {signal}")

        def search_signal(feature: FeatureState | None, search: SearchState) -> float:
            value = getattr(search, signal)
            return float(value) if value is not None else 0.0

        return search_signal
    if "op" not in node:
        raise ValueError("expression object must contain value, feature, search, or op")
    operation = str(node["op"])
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"unsafe or unknown DSL operation: {operation}")

    compile_child = lambda child: _compile_node(
        child,
        allow_feature_signals=allow_feature_signals,
        depth=depth + 1,
        max_depth=max_depth,
        max_nodes=max_nodes,
        counter=counter,
    )
    if operation == "weighted_sum":
        _require_keys(node, required={"op", "terms"})
        terms = node["terms"]
        if not isinstance(terms, list) or not terms:
            raise ValueError("weighted_sum.terms must be a non-empty list")
        compiled_terms = []
        for term in terms:
            if not isinstance(term, dict):
                raise ValueError("each weighted_sum term must be an object")
            _require_keys(term, required={"weight", "expr"})
            compiled_terms.append((_finite_number(term["weight"]), compile_child(term["expr"])))
        return lambda feature, search: sum(
            weight * expression(feature, search) for weight, expression in compiled_terms
        )

    if operation in {"add", "subtract", "multiply", "min", "max", "safe_divide"}:
        allowed = {"op", "args"} | ({"epsilon"} if operation == "safe_divide" else set())
        _require_keys(node, required={"op", "args"}, allowed=allowed)
        args = node["args"]
        if not isinstance(args, list) or not args:
            raise ValueError(f"{operation}.args must be a non-empty list")
        if operation in {"subtract", "safe_divide"} and len(args) != 2:
            raise ValueError(f"{operation} requires exactly two arguments")
        compiled = [compile_child(argument) for argument in args]
        if operation == "add":
            return lambda feature, search: sum(item(feature, search) for item in compiled)
        if operation == "subtract":
            return lambda feature, search: compiled[0](feature, search) - compiled[1](feature, search)
        if operation == "multiply":
            return lambda feature, search: math.prod(item(feature, search) for item in compiled)
        if operation == "min":
            return lambda feature, search: min(item(feature, search) for item in compiled)
        if operation == "max":
            return lambda feature, search: max(item(feature, search) for item in compiled)
        epsilon = _finite_number(node.get("epsilon", 1e-9))
        if epsilon <= 0:
            raise ValueError("safe_divide.epsilon must be positive")

        def safe_divide(feature: FeatureState | None, search: SearchState) -> float:
            numerator = compiled[0](feature, search)
            denominator = compiled[1](feature, search)
            if abs(denominator) < epsilon:
                denominator = epsilon if denominator >= 0 else -epsilon
            return numerator / denominator

        return safe_divide

    if operation == "clip":
        _require_keys(node, required={"op", "value", "lower", "upper"})
        value_fn = compile_child(node["value"])
        lower_fn = compile_child(node["lower"])
        upper_fn = compile_child(node["upper"])

        def clip(feature: FeatureState | None, search: SearchState) -> float:
            lower = lower_fn(feature, search)
            upper = upper_fn(feature, search)
            if lower > upper:
                raise ValueError("clip lower bound exceeds upper bound")
            return min(upper, max(lower, value_fn(feature, search)))

        return clip

    _require_keys(node, required={"op", "condition", "if_true", "if_false"})
    condition = compile_child(node["condition"])
    if_true = compile_child(node["if_true"])
    if_false = compile_child(node["if_false"])
    return lambda feature, search: (
        if_true(feature, search)
        if condition(feature, search) > 0
        else if_false(feature, search)
    )


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean literals are not numeric DSL values")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("DSL numeric values must be finite")
    return number


def _require_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str] | None = None,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"DSL node is missing fields: {missing}")
    allowed = required if allowed is None else allowed
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"DSL node contains unknown fields: {unknown}")
