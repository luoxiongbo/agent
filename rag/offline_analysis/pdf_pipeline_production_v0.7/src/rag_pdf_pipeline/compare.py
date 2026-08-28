from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON object")
        records.append(value)
    return records


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(record)
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("source", None)
        metadata.pop("package_root", None)
        metadata.pop("pipeline_version", None)
        metadata.pop("code_fingerprint", None)
    return value


def compare_sets(
    old_records: Sequence[dict[str, Any]],
    new_records: Sequence[dict[str, Any]],
    id_field: str,
) -> dict[str, Any]:
    old = {str(item[id_field]): item for item in old_records}
    new = {str(item[id_field]): item for item in new_records}
    common = sorted(set(old) & set(new))
    normalized_same = [
        record_id
        for record_id in common
        if normalize_record(old[record_id]) == normalize_record(new[record_id])
    ]
    changed = [record_id for record_id in common if record_id not in set(normalized_same)]
    return {
        "old_count": len(old),
        "new_count": len(new),
        "same_id_set": set(old) == set(new),
        "common_count": len(common),
        "normalized_identical_count": len(normalized_same),
        "normalized_identical_ratio": round(len(normalized_same) / max(len(common), 1), 6),
        "added_ids": sorted(set(new) - set(old))[:20],
        "removed_ids": sorted(set(old) - set(new))[:20],
        "changed_ids": changed[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较两次 Parent/Child JSONL 输出")
    parser.add_argument("--old-parents", required=True)
    parser.add_argument("--old-children", required=True)
    parser.add_argument("--new-parents", required=True)
    parser.add_argument("--new-children", required=True)
    parser.add_argument("-o", "--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = {
        "parents": compare_sets(
            load_jsonl(args.old_parents), load_jsonl(args.new_parents), "parent_id"
        ),
        "children": compare_sets(
            load_jsonl(args.old_children), load_jsonl(args.new_children), "child_id"
        ),
    }
    report["behaviorally_identical"] = (
        report["parents"]["normalized_identical_ratio"] == 1.0
        and report["children"]["normalized_identical_ratio"] == 1.0
        and report["parents"]["same_id_set"]
        and report["children"]["same_id_set"]
    )
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
