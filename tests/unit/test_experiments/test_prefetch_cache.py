"""The cache prefetch for offline cluster jobs finds every hub model Octo loads by name."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "setup" / "prefetch_research_cache.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prefetch_research_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_models_come_from_the_checkpoint_configuration():
    config = {
        "text_processor": {"kwargs": {"tokenizer_name": "t5-base"}},
        "model": {
            "task_tokenizers": {
                "language": {"kwargs": {"encoder": "t5-base"}},
                "goal": {"kwargs": {}},
            }
        },
    }
    assert load_script().hub_model_names(config) == ["t5-base"]
    config["model"]["task_tokenizers"]["language"]["kwargs"]["encoder"] = "t5-large"
    assert load_script().hub_model_names(config) == ["t5-base", "t5-large"]
