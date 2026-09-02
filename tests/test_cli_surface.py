import argparse

import main


def test_cli_exposes_only_active_verapin_pipeline():
    parser = main.build_parser()
    action = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(action.choices) == {
        "validate-datasets", "validate-benchmarks", "hardness", "kernel-search",
        "adks", "evolve-verapin", "evaluate-verapin", "verify-policy",
        "replay-evolution",
    }
