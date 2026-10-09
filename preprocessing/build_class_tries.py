"""Precompute per-class token tries -> dbpedia/class_tries.pkl.

Inputs: dbpedia/class_entities.json (from extract_entities.py),
        tbox_reasoner/tbox_rules.json (class_subsumptions + effective ranges)
Output: {class_iri: trie over Qwen token ids} with ?uri/?x in every trie,
        None as terminal key, merged all-entities trie under "__ALL__"
        (subject slot, any entity legal). Every tbox-reachable class is
        present; entity-less classes share one variables-only trie.

Tokenizer-dependent -- rebuild if the model changes.
"""

import json
import os
import pickle
import time
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = os.getenv("TRIE_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B")
DATA_DIR = Path(__file__).parent.parent / "dbpedia"
VARIABLES = ["?uri", "?x"]
TRIE_END = None  # terminal marker key (must match the runtime matcher)
ALL_KEY = "__ALL__"  # reserved key for the merged all-entities trie

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def build_trie(strings):
    """Build a dict-of-dicts token trie over the given strings."""
    root = {}
    for ids in tokenizer(strings, add_special_tokens=False).input_ids:
        node = root
        for tok in ids:
            node = node.setdefault(tok, {})
        node[TRIE_END] = None
    return root


def merge_tries(a, b):
    """Union trie b into a without mutating b (copy-on-write along shared
    paths, so per-class tries stay intact)."""
    for k, v in b.items():
        if k in a and isinstance(v, dict) and isinstance(a[k], dict):
            a[k] = dict(a[k])
            merge_tries(a[k], v)
        else:
            a[k] = v


def main():
    with open(DATA_DIR / "class_entities.json", encoding="utf-8") as f:
        class_entities = json.load(f)

    tries = {}
    total_entities = 0
    start = time.time()
    for i, (class_iri, entities) in enumerate(sorted(class_entities.items()), 1):
        tries[class_iri] = build_trie(entities + VARIABLES)
        total_entities += len(entities)
        if i % 50 == 0:
            print(f"{i}/{len(class_entities)} classes, "
                  f"{total_entities:,} entities, {time.time() - start:.0f}s",
                  flush=True)

    # tbox classes with no entities share one variables-only trie
    # (single shared object -- pickled once; tries are never mutated)
    tbox = json.loads(
        (DATA_DIR.parent / "tbox_reasoner" / "tbox_rules.json").read_text(encoding="utf-8")
    )
    reachable = set()
    for parent, descendants in tbox["class_subsumptions"].items():
        reachable.add(parent)
        reachable.update(descendants)
    for ranges in tbox["effective_property_range_map"].values():
        reachable.update(ranges)
    missing = sorted(reachable - tries.keys())
    vars_trie = build_trie(VARIABLES)
    for class_iri in missing:
        tries[class_iri] = vars_trie

    # merged trie over every entity (+ variables, already in each class trie);
    # the subject slot accepts any entity, so it walks this one
    all_trie = {}
    for class_iri in sorted(tries):
        merge_tries(all_trie, tries[class_iri])
    tries[ALL_KEY] = all_trie

    out_path = DATA_DIR / "class_tries.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(tries, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(tries)} tries ({len(class_entities)} classes, "
          f"{len(missing)} variables-only, 1 merged; {total_entities:,} "
          f"entities, {size_mb:.0f} MB) -> {out_path} "
          f"in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
