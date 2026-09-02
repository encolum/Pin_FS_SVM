import json
import os

import pytest
import yaml

import main
from src.data import benchmark_adapter
from src.data.benchmark_loaders import DEFAULT_BENCHMARK_ROOT
from src.data.benchmark_registry import read_benchmark_registry
from src.data.benchmark_validation import write_validation_manifest


def test_cli_real_six_benchmarks_no_experiment(tmp_path, capsys, monkeypatch):
    from src.experiments import verapin
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not train, evolve, or fit preprocessing")
    for name in ("run_hardness_benchmark", "run_static_kernel_search", "run_adks",
                 "run_verapin_evolution", "run_verapin_final"):
        monkeypatch.setattr(verapin, name, forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(MaxAbsScaler, "fit", forbidden)
    output = tmp_path / "solver_ready_validation.json"
    assert main.main(["validate-benchmarks", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(report) == 6
    assert all(row["status"] == "passed" for row in report)
    text = capsys.readouterr().out
    assert "6/6 passed" in text
    for label in ("positive=", "negative=", "storage=", "density=", "partition_policy=",
                  "label_mapping=", "preprocessing=", "Warning:"):
        assert label in text
    assert {path.name for path in tmp_path.iterdir()} == {output.name}


def test_cli_official_holdout_report(tmp_path, capsys):
    registry, _ = read_benchmark_registry()
    for name in ("hill_valley", "madelon"):
        registry[name]["source_partition_policy"] = "official_holdout"
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry))
    output = tmp_path / "report.json"
    assert main.main(["validate-benchmarks", "--registry", str(path), "--output", str(output)]) == 0
    by_name = {row["dataset"]: row for row in json.loads(output.read_text())}
    assert by_name["hill_valley"]["samples"] == 606
    assert by_name["hill_valley"]["holdout"]["samples"] == 606
    assert by_name["madelon"]["samples"] == 2000
    assert by_name["madelon"]["holdout"]["samples"] == 600
    assert "Separate validation holdout" in capsys.readouterr().out


def test_cli_failed_dataset_is_nonzero_and_reported(tmp_path, monkeypatch, capsys):
    record = {"dataset": "gina", "status": "failed", "errors": ["checksum mismatch"]}
    monkeypatch.setattr(benchmark_adapter, "audit_solver_ready_benchmarks", lambda **kwargs: [record])
    output = tmp_path / "failed.json"
    assert main.main(["validate-benchmarks", "--output", str(output)]) == 1
    assert json.loads(output.read_text()) == [record]
    assert "checksum mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["missing", "invalid"])
def test_cli_registry_error_has_no_traceback(tmp_path, capsys, mode):
    path = tmp_path / "registry.yaml"
    if mode == "invalid":
        path.write_text("not: a registry\n")
    with pytest.raises(SystemExit) as exc:
        main.main(["validate-benchmarks", "--registry", str(path)])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_json_registry_cannot_be_overwritten_even_via_hardlink(tmp_path):
    registry, _ = read_benchmark_registry()
    path = tmp_path / "registry.json"
    path.write_text(yaml.safe_dump(registry))
    original = path.read_bytes()
    alias = tmp_path / "alias.json"
    os.link(path, alias)
    for output in (path, alias):
        with pytest.raises(SystemExit) as exc:
            main.main(["validate-benchmarks", "--registry", str(path), "--output", str(output), "--overwrite"])
        assert exc.value.code == 2
        assert path.read_bytes() == original


def test_separate_solver_report_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "validation.json"
    report = [{"dataset": "hill_valley", "status": "passed"}]
    write_validation_manifest(report, path)
    with pytest.raises(FileExistsError):
        write_validation_manifest([], path)
    assert json.loads(path.read_text()) == report
    write_validation_manifest([], path, overwrite=True)
    assert json.loads(path.read_text()) == []


def test_solver_report_cannot_overwrite_original_manifest():
    path = DEFAULT_BENCHMARK_ROOT / "manifest.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="original input"):
        write_validation_manifest([], path, overwrite=True)
    assert path.read_bytes() == original


def test_raw_cli_remains_separate(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("raw validation must not call solver adapter")
    monkeypatch.setattr(benchmark_adapter, "audit_solver_ready_benchmarks", forbidden)
    assert main.main(["validate-datasets"]) == 0
