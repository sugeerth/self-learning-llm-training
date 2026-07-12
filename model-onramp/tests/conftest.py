import pytest


@pytest.fixture(autouse=True)
def onramp_home(tmp_path, monkeypatch):
    """Every test gets a fresh, isolated state directory."""
    monkeypatch.setenv("ONRAMP_HOME", str(tmp_path / "onramp-home"))
    return tmp_path / "onramp-home"


@pytest.fixture
def registry():
    from onramp.registry import get_registry

    return get_registry()
