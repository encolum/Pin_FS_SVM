import argparse

import main
import pytest


def test_cli_exposes_only_active_verapin_pipeline():
    parser = main.build_parser()
    action = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(action.choices) == {
        "validate-datasets", "validate-benchmarks", "hardness",
        "adks", "evolve-verapin", "evaluate-verapin", "verify-policy",
        "replay-evolution",
    }


@pytest.mark.parametrize("command", ["hardness", "adks"])
def test_experiment_commands_require_an_explicit_config(command):
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([command])
