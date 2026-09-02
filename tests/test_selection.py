from src.experiments.selection import selection_tie_key


def test_tie_key_prefers_fewer_features_then_smaller_budget_then_parameters():
    assert selection_tie_key({"B": 3, "C": 1.0}, 2.0) < selection_tie_key(
        {"B": 1, "C": 1.0}, 3.0
    )
    assert selection_tie_key({"B": 1, "C": 1.0}, 2.0) < selection_tie_key(
        {"B": 3, "C": 1.0}, 2.0
    )
    assert selection_tie_key({"C": 2.0}, 2.0) < selection_tie_key(
        {"C": 10.0}, 2.0
    )
