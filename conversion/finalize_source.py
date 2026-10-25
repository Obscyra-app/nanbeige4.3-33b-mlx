"""Merge individually verified download receipts and rehash the complete source."""
import argparse
import json
from pathlib import Path

from prepare import REPO, REVISION, atomic_json, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipts", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    entries = {}
    for path in a.receipts:
        receipt = json.loads(path.read_text())
        assert receipt["source"] == REPO and receipt["revision"] == REVISION
        for record in receipt["files"]:
            if record["name"] in entries:
                assert entries[record["name"]]["sha256"] == record["sha256"]
            entries[record["name"]] = record
    index = json.loads((a.source / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values()) | {"README.md", "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"}
    assert set(entries) == expected, (sorted(expected - entries.keys()), sorted(entries.keys() - expected))
    for name, record in sorted(entries.items()):
        path = a.source / name
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
    result = {"source": REPO, "revision": REVISION, "complete": True, "metadata_only": False,
              "total_bytes": sum(r["bytes"] for r in entries.values()), "files": list(entries.values()),
              "verification": "Every receipt entry verified individually against pinned upstream; full local source rehashed after receipt merge"}
    atomic_json(a.output, result)
    print(json.dumps({"status": "pass", "files": len(entries), "bytes": result["total_bytes"]}))


if __name__ == "__main__":
    main()
