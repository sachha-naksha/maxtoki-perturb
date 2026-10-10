"""Regression coverage for Fiddle dictionary values versus key-tuple positions."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "torch_pipeline"))
from extract_token_dict import extract, main


def _dump(tmp_path, entries):
    # BOS was inserted last, but its trained token ID remains 2.
    raw = {"objects": {
        "tokenizer": {"items": [["Attr(name='token_dictionary')",
                                 {"type": "ref", "key": "vocab"}]]},
        "vocab": {"type": {"name": "dict"}, "items": [
            [f"Key(key={token!r})", {"type": "leaf", "value": token_id}]
            for token, token_id in entries
        ]},
        "key_tuple": {"type": {"name": "tuple"}, "items": [
            [f"Index(index={i})", token] for i, (token, _) in enumerate(entries)
        ]},
    }}
    path = tmp_path / "io.json"
    path.write_text(json.dumps(raw))
    return path, raw


ENTRIES = [("<pad>", 0), ("<mask>", 1), ("<eos>", 3),
           ("ENSG00000004799", 63), ("-1", 21774),
           ("<boq>", 23275), ("<eoq>", 23276), ("<bos>", 2)]


def test_preserves_stored_ids_instead_of_tuple_positions(tmp_path):
    path, _ = _dump(tmp_path, ENTRIES)
    assert extract(path) == dict(ENTRIES)


def test_cli_writes_actual_mapping(tmp_path, monkeypatch):
    path, _ = _dump(tmp_path, ENTRIES)
    out = tmp_path / "token_dictionary.json"
    monkeypatch.setattr("sys.argv", ["extract_token_dict", str(path), str(out)])
    main()
    assert json.loads(out.read_text()) == dict(ENTRIES)


def test_resolves_referenced_values_and_quoted_keys(tmp_path):
    path, raw = _dump(tmp_path, [("gene's token", 42)])
    raw["objects"]["vocab"]["items"][0][1] = {"type": "ref", "key": "id"}
    raw["objects"]["id"] = {"type": "leaf", "value": 42}
    path.write_text(json.dumps(raw))
    assert extract(path) == {"gene's token": 42}


@pytest.mark.parametrize("entries", [[], [("a", 1), ("b", 1)],
                                    [("a", 1), ("a", 2)], [("a", -1)],
                                    [("a", True)], [("a", "1")]])
def test_rejects_invalid_mapping(tmp_path, entries):
    path, _ = _dump(tmp_path, entries)
    with pytest.raises(ValueError):
        extract(path)


def test_does_not_fall_back_to_tuple_positions(tmp_path):
    path, raw = _dump(tmp_path, ENTRIES)
    del raw["objects"]["tokenizer"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="No token_dictionary"):
        extract(path)


def test_rejects_conflicting_vocabularies(tmp_path):
    path, raw = _dump(tmp_path, ENTRIES)
    raw["objects"]["other_tokenizer"] = {"items": [
        ["Attr(name='token_dictionary')", {"type": "ref", "key": "other_vocab"}]]}
    raw["objects"]["other_vocab"] = {"type": {"name": "dict"}, "items": [
        ["Key(key='<pad>')", 7]]}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Conflicting"):
        extract(path)
