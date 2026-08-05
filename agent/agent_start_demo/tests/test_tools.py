from pathlib import Path

import pytest

from agent_blueprint.tools import Workspace, safe_calculate


def test_safe_calculator() -> None:
    assert safe_calculate("(17 + 5) * 3") == 66


def test_calculator_rejects_calls() -> None:
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo bad')")


def test_workspace_blocks_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    with pytest.raises(PermissionError):
        workspace.resolve("../secret.txt")
