import copy

import pytest

from src.utils.config import load_config
from src.experiments.verapin import validate_verapin_config


def test_evolution_config_rejects_any_heldout_test_instance():
    config = load_config("configs/verapin_evolution.yaml")
    # Resolve only enough to ensure the split guard itself remains independently testable.
    extra = copy.deepcopy(config["instances"][0])
    extra["id"] = "forbidden-test"
    extra["research_split"] = "test"
    config["instances"].append(extra)
    with pytest.raises(ValueError, match="held-out test"):
        validate_verapin_config(config, command="evolve-verapin")


def test_final_config_keeps_outer_classification_choices_author_gated():
    config = load_config("configs/verapin_final.yaml")
    with pytest.raises(ValueError, match="unresolved author decisions") as error:
        validate_verapin_config(config, command="evaluate-verapin")
    assert config["classification"]["outer_folds"] == 5
    assert config["classification"]["inner_folds"] == 3
    assert "classification.parameter_grid" in str(error.value)
    assert "classification.outer_seed" in str(error.value)


@pytest.mark.parametrize("kind", ["dataset", "legacy_dataset"])
def test_legacy_manuscript_instance_kinds_are_rejected(kind):
    config = load_config("configs/hardness_real_pilot.yaml")
    config["instances"][0]["kind"] = kind
    with pytest.raises(ValueError, match="must be synthetic or benchmark"):
        validate_verapin_config(config, command="hardness")
