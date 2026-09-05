"""End-to-end surface-level evaluation over the LC-QuAD test split.

For every test question, generate a SPARQL query with the type-constrained
generator, canonicalize both the produced query and the gold query, and count
exact string matches. Two metrics are reported: strict exact match, and match
"modulo namespace twins", where predicates whitelisted in both the ontology/
and property/ namespaces (e.g. architect) are compared namespace-neutrally.
Every question is decoded once per rung of the ablation ladder -- the full
constraint stack, the same minus the ontological boosts, structure-only, and
the raw fine-tuned weights with no constraints at all -- so one run produces
every column the comparison needs. output.json carries the gold query and all
four outputs, canonical form only: canonicalisation is whitespace-level and
loses nothing the evaluation uses, and keeping one spelling per query stops the
raw and canonical copies drifting apart. Records are rewritten every CHECKPOINT
questions so a crash does not lose the run, and re-read on startup: a run cut
short by a Colab timeout resumes at the question after the last record instead
of starting over. Delete output.json to force a fresh run.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

# --beams is parsed before the generation module is imported, because that
# module reads BEAM_WIDTH from the environment at import time -- and importing
# it loads the 3 GB checkpoint, which --help should not have to wait for
if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="LC-QuAD test-split evaluation.")
    _ap.add_argument("--beams", type=int, default=None,
                     help="beam width: the whole-query beam search in every "
                          "constrained rung, and num_beams for the unconstrained "
                          "baseline (default 4; 1 is greedy)")
    _beams = _ap.parse_args().beams
    if _beams is not None:
        os.environ["BEAM_WIDTH"] = str(_beams)

from type_constrained_generation import (BEAM_WIDTH, generate, generate_grammar_only,
                                         generate_no_boosts, generate_unconstrained)

DATA_FILE = Path(__file__).parent / "lcquad_data" / "test-data.json"
WHITELIST_FILE = Path(__file__).parent / "lcquad_data" / "predicates.txt"
OUT_FILE = Path(__file__).parent / "output.json"
CHECKPOINT = 10  # questions between progress prints / output.json rewrites

# The ablation ladder, strongest first. Each entry is (record prefix, decoder);
# the full system keeps the legacy "generated" prefix so existing tooling still
# finds it. generate() minus generate_no_boosts() isolates the ontological
# boosts; generate_no_boosts() minus generate_grammar_only() isolates the KB
# vocabulary (entity tries + relation whitelist) from bare structure.
SYSTEMS = (
    ("generated", generate),
    ("no_boosts", generate_no_boosts),
    ("grammar_only", generate_grammar_only),
    ("unconstrained", generate_unconstrained),
)


def canonicalize(q):
    # _scratch_trie.canonicalize plus one extra step: drop a trailing dot
    # before the closing brace. Legal SPARQL and present in ~28% of gold
    # queries, but the generator's grammar can never emit it, so without
    # this those queries could never exact-match.
    q = " ".join(q.split())
    q = q.replace("COUNT( ?uri )", "COUNT(?uri)")
    q = q.replace("{", "{ ").replace("}", " }")
    q = re.sub(r"\s*\.\s*(?![^<]*>)", " . ", q)
    return " ".join(q.split()).replace(" . }", " }")


def _load_twin_names():
    # predicate local-names whitelisted in BOTH the ontology/ and property/
    # namespaces -- the pairs a strict exact match cannot tell apart
    ont, prop = set(), set()
    for line in WHITELIST_FILE.read_text(encoding="utf-8").splitlines():
        iri = line.strip().rstrip(",")
        if "/ontology/" in iri:
            ont.add(iri.rsplit("/", 1)[1])
        elif "/property/" in iri:
            prop.add(iri.rsplit("/", 1)[1])
    return ont & prop


TWIN_RE = re.compile(
    r"<http://dbpedia\.org/(?:ontology|property)/("
    + "|".join(sorted((re.escape(n) for n in _load_twin_names()), key=len, reverse=True))
    + r")>"
)


def canonicalize_twins(q):
    # canonicalize, then rewrite every twin predicate to a namespace-neutral
    # IRI, so gold <.../property/architect> and generated <.../ontology/architect>
    # compare equal. Generation itself is unaffected.
    return TWIN_RE.sub(r"<dbpedia-twin/\1>", canonicalize(q))


def report(hits, twin_hits, n):
    """One line of exact-match rates, in ladder order."""
    parts = [f"{name} {hits[name]}/{n} ({hits[name] / n:.1%})" for name, _ in SYSTEMS]
    parts.insert(1, f"mod-twins {twin_hits}/{n} ({twin_hits / n:.1%})")
    return "  ".join(parts)


def main():
    print(f"beam width: {BEAM_WIDTH}", flush=True)
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    # Resume: an existing output.json is taken as the first N answers and the run
    # continues at question N+1. A Colab session is routinely shorter than a full
    # four-rung run, so this is the difference between losing a run and extending
    # it across sessions.
    results = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else []
    done = len(results)
    if done:
        # a mismatch means this output.json belongs to a different dataset or a
        # reordered one; continuing would silently interleave two runs
        assert results[-1]["question"] == data[done - 1]["corrected_question"], (
            f"{OUT_FILE.name} does not line up with {DATA_FILE.name} at record {done}"
        )
        print(f"resuming after {done} records, {len(data) - done} questions left", flush=True)
    if done >= len(data):
        print("nothing to do: output.json already covers every question")
        return

    # seed the counters from what is already on disk, so the reported rates cover
    # the whole file rather than just this session
    hits = {name: sum(r[f"{name}_match"] for r in results) for name, _ in SYSTEMS}
    twin_hits = sum(r["match_modulo_twins"] for r in results)
    start = time.perf_counter()
    for i, item in enumerate(data[done:], done + 1):
        question = item["corrected_question"]
        gold_c = canonicalize(item["sparql_query"])
        gold_twins = canonicalize_twins(item["sparql_query"])
        record = {"question": question, "gold_canonical": gold_c}
        for name, decode in SYSTEMS:
            try:
                produced = decode(question)
            except Exception as e:  # one bad question must not kill a long run
                produced = f"ERROR: {e}"
            produced_c = canonicalize(produced)
            record[f"{name}_canonical"] = produced_c
            record[f"{name}_match"] = produced_c == gold_c
            hits[name] += record[f"{name}_match"]
            if name == "generated":
                # the twin-neutral variant is only reported for the full system
                record["match_modulo_twins"] = canonicalize_twins(produced) == gold_twins
                twin_hits += record["match_modulo_twins"]
        results.append(record)
        if i % CHECKPOINT == 0:
            OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
            per = (time.perf_counter() - start) / (i - done)
            print(f"{i}/{len(data)}  {report(hits, twin_hits, i)}  "
                  f"{per:.2f}s per question", flush=True)

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    n = len(data)
    print(f"done: {report(hits, twin_hits, n)}  "
          f"({elapsed / (n - done):.2f}s per question over the {n - done} done here, "
          f"{elapsed / 60:.0f} min this session)")


if __name__ == "__main__":
    main()
