import pytest
import yaml

import main
from src.data import benchmark_data
from src.data.benchmark_data import read_benchmark_registry


def test_cli_real_six_benchmarks_no_experiment(capsys, monkeypatch):
    from src.experiments import verapin
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not train, evolve, or fit preprocessing")
    for name in ("run_hardness_benchmark", "run_static_kernel_search", "run_adks",
                 "run_verapin_evolution", "run_verapin_final"):
        monkeypatch.setattr(verapin, name, forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(MaxAbsScaler, "fit", forbidden)
    assert main.main(["validate-benchmarks"]) == 0
    text = capsys.readouterr().out
    assert "6/6 passed" in text
    for label in ("positive=", "negative=", "storage=", "density=", "partition_policy=",
                  "label_mapping=", "preprocessing=", "Warning:"):
        assert label in text


def test_cli_official_holdout_report(tmp_path, capsys):
    registry, _ = read_benchmark_registry()
    for name in ("hill_valley", "madelon"):
        registry[name]["source_partition_policy"] = "official_holdout"
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry))
    assert main.main(["validate-benchmarks", "--registry", str(path)]) == 0
    text = capsys.readouterr().out
    assert "hill_valley: shape=(606, 100)" in text
    assert "Separate test holdout: 606 rows" in text
    assert "madelon: shape=(2000, 500)" in text
    assert "Separate validation holdout: 600 rows" in text


def test_cli_failed_dataset_is_nonzero_and_reported(monkeypatch, capsys):
    record = {"dataset": "gina", "status": "failed", "errors": ["checksum mismatch"]}
    monkeypatch.setattr(benchmark_data, "audit_solver_ready_benchmarks", lambda **kwargs: [record])
    assert main.main(["validate-benchmarks"]) == 1
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


def test_raw_cli_remains_separate(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("raw validation must not call solver adapter")
    monkeypatch.setattr(benchmark_data, "audit_solver_ready_benchmarks", forbidden)
    assert main.main(["validate-datasets"]) == 0
