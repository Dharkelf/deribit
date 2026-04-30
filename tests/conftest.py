"""Shared pytest fixtures."""

import pytest
import yaml


@pytest.fixture
def config() -> dict:
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)
