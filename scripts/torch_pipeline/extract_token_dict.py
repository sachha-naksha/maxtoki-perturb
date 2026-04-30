"""Extract token_dictionary.json from a BioNeMo MaxToki distcp context/io.json.

The fiddle-config dump stores the vocabulary as the largest tuple of
(Index(index=N), token_name) pairs. We invert that to a {name: id} dict
and write it out as the JSON file `bionemo.maxtoki.predict` expects.

Run:
    python -m scripts.torch_pipeline.extract_token_dict \\
        /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/io.json \\
        /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_INDEX_RE = re.compile(r"Index\(index=(\d+)\)")


def _is_str_tuple(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    t = obj.get("type", {})
    if isinstance(t, dict) and t.get("name") != "tuple":
        return False
    items = obj.get("items", [])
    return bool(items) and all(
        isinstance(it, list) and len(it) == 2
        and isinstance(it[1], str)
        and _INDEX_RE.match(str(it[0]))
        for it in items
    )


def extract(io_path: Path) -> dict[str, int]:
    raw = json.loads(io_path.read_text())
    objects = raw.get("objects", raw)

    candidates: list[tuple[str, list]] = []
    for key, obj in objects.items():
        if _is_str_tuple(obj):
            candidates.append((key, obj["items"]))
    if not candidates:
        sys.exit(f"no string-valued indexed tuple in {io_path}")

    # Vocab is the largest such tuple
    candidates.sort(key=lambda x: -len(x[1]))
    key, items = candidates[0]
    print(f"[extract] using {key!r} ({len(items)} entries)")

    id_to_token: dict[int, str] = {}
    for idx_str, name in items:
        m = _INDEX_RE.match(idx_str)
        if m:
            id_to_token[int(m.group(1))] = name

    return {name: tid for tid, name in id_to_token.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("io_json", type=Path)
    p.add_argument("out_path", type=Path)
    args = p.parse_args()

    td = extract(args.io_json)
    args.out_path.write_text(json.dumps(td, indent=2))

    print(f"[extract] wrote {args.out_path}: {len(td)} tokens")
    for s in ("<pad>", "<mask>", "<bos>", "<eos>", "<boq>", "<eoq>"):
        print(f"  {s}: {td.get(s)}")
    nums = [k for k in td if k.lstrip("-").isdigit()]
    print(f"  numeric tokens: {len(nums)}"
          f" (range {min((int(n) for n in nums), default=None)} -> "
          f"{max((int(n) for n in nums), default=None)})")


if __name__ == "__main__":
    main()