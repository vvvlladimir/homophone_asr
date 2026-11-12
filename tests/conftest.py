import pytest

from homophone_asr.config import load_config, mock_overrides


@pytest.fixture
def cfg(tmp_path):
    c = load_config(overrides=mock_overrides())
    c["paths"]["cache_dir"] = str(tmp_path / "cache")
    c["paths"]["results_dir"] = str(tmp_path / "results")
    c["paths"]["refs_dir"] = str(tmp_path / "refs")
    c["tts"]["common_voice"]["enabled"] = False
    return c
