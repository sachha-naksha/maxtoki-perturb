"""Extract token_dictionary.json from a BioNeMo MaxToki distcp context/io.json.

Read the tokenizer's actual token_dictionary values from the Fiddle dump.
Serialized tuple indices describe dictionary insertion order, not token IDs.
Write the stored {name: id} mapping as the JSON file prediction expects.

Run:
    python -m scripts.torch_pipeline.extract_token_dict \\
        /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/io.json \\
        /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def extract(io_path: Path) -> dict[str, int]:
    raw = json.loads(io_path.read_text())
    objects = raw.get("objects", raw)

    def resolve(value):
        seen = set()
        while isinstance(value, dict) and value.get("type") == "ref":
            key = value["key"]
            if key in seen:
                raise ValueError(f"Cyclic reference {key!r} in {io_path}")
            seen.add(key)
            value = objects[key]
        if isinstance(value, dict) and value.get("type") == "leaf":
            return value["value"]
        return value

    # Follow the tokenizer's named attribute, not an unrelated tuple of keys.
    candidates = []
    for obj in objects.values():
        for attr, value in obj.get("items", []):
            if attr == "Attr(name='token_dictionary')":
                candidates.append(resolve(value))
    if not candidates:
        raise ValueError(f"No token_dictionary attribute in {io_path}")

    mappings = []
    for obj in candidates:
        if obj.get("type", {}).get("name") != "dict":
            raise ValueError(f"token_dictionary is not a serialized dict in {io_path}")
        mapping = {}
        for key, value in obj["items"]:
            if not key.startswith("Key(key=") or not key.endswith(")"):
                raise ValueError(f"Invalid dictionary key {key!r} in {io_path}")
            token = ast.literal_eval(key[len("Key(key="):-1])
            token_id = resolve(value)
            if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
                raise ValueError(f"Invalid token mapping {token!r}: {token_id!r}")
            if token in mapping:
                raise ValueError(f"Duplicate token {token!r} in {io_path}")
            mapping[token] = token_id
        if not mapping or len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Empty vocabulary or duplicate token IDs in {io_path}")
        mappings.append(mapping)
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError(f"Conflicting token_dictionary mappings in {io_path}")
    return mappings[0]


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