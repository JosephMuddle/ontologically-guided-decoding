"""Precompute per-class token tries and pickle them to dbpedia/class_tries.pkl.

Inputs: dbpedia/class_entities.json (written by extract_entities.py) and
        tbox_reasoner/tbox_rules.json (class_subsumptions + effective ranges)
Output: dbpedia/class_tries.pkl -- {class_iri: dict-of-dicts trie over Qwen
        token ids}, with variables ?uri/?x inserted into every trie (object
        slots may be variables) and None as the terminal marker key, plus a
        merged all-entities trie under the reserved key "__ALL__" (walked for
        the subject slot, where any entity is legal). Every
        class reachable from the tbox (subsumption parents/descendants, range
        values) is present; classes without entities share one variables-only
        trie, so the runtime never has to build a trie itself.

The tries are tokenizer-dependent: rebuild this file if the model changes.
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
    """Union trie b into trie a without mutating b: shared subtrees are
    copied along the merged path (copy-on-write), so the per-class tries
    stay intact. Terminal markers (TRIE_END) are just keys."""
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

    # every class the runtime can ask for must be present: classes mentioned
    # in the tbox but having no entities share one variables-only trie
    # (single shared object -- pickle stores it once; tries are never mutated)
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

    # one merged trie over every entity (+ variables, present in each class
    # trie): the subject slot accepts any entity, so it walks this trie
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
