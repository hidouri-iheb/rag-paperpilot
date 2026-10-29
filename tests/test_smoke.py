"""Smoke test: the package imports and settings load. No network, no API key needed."""

import paperpilot
from paperpilot.config import Settings


def test_package_imports():
    assert paperpilot.__name__ == "paperpilot"


def test_settings_load_without_env_file():
    settings = Settings(_env_file=None)
    assert settings.final_top_k > 0