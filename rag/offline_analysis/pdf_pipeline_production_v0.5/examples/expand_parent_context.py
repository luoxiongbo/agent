from __future__ import annotations

import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "用法：python examples/expand_parent_context.py OUTPUT_DIR CHILD_ID"
        )

    output = Path(sys.argv[1])
    child_id = sys.argv[2]
    children = {item["child_id"]: item for item in load_jsonl(output / "children.jsonl")}
    parents = {item["parent_id"]: item for item in load_jsonl(output / "parents.jsonl")}

    child = children[child_id]
    parent = parents[child["parent_id"]]

    print("=== 检索命中的 Child ===")
    print(child["content"])
    print("\n=== 展开的 Parent ===")
    print(parent["content"])


if __name__ == "__main__":
    main()
