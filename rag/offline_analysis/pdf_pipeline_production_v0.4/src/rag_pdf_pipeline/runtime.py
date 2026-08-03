from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

PACKAGE_VERSION = "0.4.0"


def package_root() -> Path:
    return Path(__file__).resolve().parent


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    root = package_root()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def runtime_info() -> dict[str, str]:
    return {
        "pipeline_version": PACKAGE_VERSION,
        "code_fingerprint": code_fingerprint(),
        "package_root": str(package_root()),
        "module_file": str(Path(__file__).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
    }


def main() -> int:
    print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
