"""Explicit execution gates; code readiness is not scientific approval."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from src.utils.serialization import read_json


def check_execution_readiness(config, command):
    execution = config.get("execution", {})
    maximum = execution.get("max_instances_per_run")
    if maximum is not None and len(config["instances"]) > maximum:
        raise ValueError(f"pilot permits at most {maximum} instance(s); select explicitly with --instance")
    if execution.get("require_hardness_evidence"):
        reports = execution.get("hardness_reports", [])
        hard = set()
        for report in reports:
            content = read_json(Path(report))
            hard.update(content.get("nontrivial_instance_ids", []))
        if len(hard) < 2:
            raise ValueError("ADKS/evolution requires hardness reports identifying at least two nontrivial instances")
    if command == "evolve-verapin" and execution.get("require_evolution_readiness"):
        required = ("adks_baseline_frozen", "hardness_groups_defined", "solver_budgets_fixed", "signals_profiled")
        if any(execution.get(name) is not True for name in required):
            raise ValueError("evolution readiness decisions have not all been explicitly approved")
    if command == "evaluate-verapin" and execution.get("require_frozen_method") and execution.get("solver_method_frozen") is not True:
        raise ValueError("final classification requires a reviewed frozen solver method")


def cplex_environment_report(*, probe_size_limit=False):
    """Optional two-second, 1001-variable license probe; never benchmark data."""
    try:
        import cplex
        result = {"installed": True, "version": version("cplex"), "probe_performed": probe_size_limit}
    except (ImportError, PackageNotFoundError) as exc:
        return {"installed": False, "error": str(exc), "probe_performed": False}
    if not probe_size_limit:
        return result
    model = cplex.Cplex()
    try:
        for setter in (model.set_log_stream, model.set_results_stream, model.set_warning_stream, model.set_error_stream):
            setter(None)
        model.parameters.timelimit.set(2)
        model.parameters.threads.set(1)
        model.variables.add(lb=[0.] * 1001, ub=[1.] * 1001)
        model.solve()
        result.update(large_model_probe_passed=True, status=model.solution.get_status_string())
    except cplex.exceptions.CplexSolverError as exc:
        result.update(large_model_probe_passed=False, error=str(exc),
                      community_limit="1016" in str(exc))
    finally:
        model.end()
    return result
