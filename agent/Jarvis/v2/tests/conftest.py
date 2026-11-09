"""Run the same behavioral contracts against s15 and the modular runtime."""

import atexit
import importlib
import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


V2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_DIR.parents[2]
S15_SOURCE = V2_DIR.parents[1] / "learn-claude-code/s15_integrated_harness/code.py"
PACKAGE = "agent.Jarvis.v2"


@pytest.fixture(params=["s15", "v2"])
def harness(request, monkeypatch, tmp_path):
    """Use fresh module state, fake credentials, and a temporary workspace."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_ID", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("FALLBACK_MODEL_ID", "test-fallback")
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
    client = Mock()
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=client))
    previous_signal = signal.getsignal(signal.SIGTERM)
    before = set(sys.modules)
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(PACKAGE + "."):
            monkeypatch.delitem(sys.modules, name)

    if request.param == "s15":
        spec = importlib.util.spec_from_file_location("s15_contract_test", S15_SOURCE)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules = [module]
    else:
        importlib.import_module(PACKAGE + ".code")
        modules = [
            module for name, module in sys.modules.items()
            if name.startswith(PACKAGE + ".")
        ]

    # Functions and shared containers retain their real defining modules.
    api = SimpleNamespace()
    for module in modules:
        vars(api).update({name: value for name, value in vars(module).items()
                          if not name.startswith("__")})
    api.model_client = client
    api.variant = request.param
    api.workspace = tmp_path
    try:
        yield api
    finally:
        api._stop_all_shell_processes()
        atexit.unregister(api._stop_all_shell_processes)
        signal.signal(signal.SIGTERM, previous_signal)
        for name in set(sys.modules) - before:
            if name.startswith((PACKAGE + ".", "integrated_memory_")):
                sys.modules.pop(name, None)


@pytest.fixture
def model_turn(harness, monkeypatch):
    """Keep memory calls deterministic so tests focus on model/tool turns."""
    monkeypatch.setattr(harness.MEMORY_RUNTIME, "read_memory_index", lambda: "")
    monkeypatch.setattr(harness.MEMORY_RUNTIME, "load_memories", lambda messages: "")
    monkeypatch.setattr(harness.MEMORY_RUNTIME, "extract_memories", lambda messages: False)
    return harness
