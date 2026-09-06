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
of starting over. --systems narrows the run to particular rungs and --redo
clears them first, so one rung can be regenerated in place after a decoder
change without touching the other three. Delete output.json to start over.
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
ARGS = None
if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="LC-QuAD test-split evaluation.")
    _ap.add_argument("--beams", type=int, default=None,
                     help="beam width: the whole-query beam search in every "
                          "constrained rung, and num_beams for the unconstrained "
                          "baseline (default 4; 1 is greedy)")
    _ap.add_argument("--systems", nargs="+", metavar="NAME", default=None,
                     help="only run these rungs (generated, no_boosts, grammar_only, "
                          "unconstrained). Rungs already stored in output.json are "
                          "kept as they are; default is all four")
    _ap.add_argument("--redo", action="store_true",
                     help="clear the selected rungs from output.json first, so they "
                          "are regenerated even where an answer is already stored")
    ARGS = _ap.parse_args()
    if ARGS.beams is not None:
        os.environ["BEAM_WIDTH"] = str(ARGS.beams)

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


def report(results):
    """Match rates per rung over whatever the file currently holds. Each rung is
    counted over the records that actually have it, so the line stays honest
    when only some rungs have been run."""
    parts = []
    for name, _ in SYSTEMS:
        have = [r for r in results if f"{name}_match" in r]
        if have:
            hits = sum(r[f"{name}_match"] for r in have)
            parts.append(f"{name} {hits}/{len(have)} ({hits / len(have):.1%})")
    twins = [r for r in results if "match_modulo_twins" in r]
    if twins:
        hits = sum(r["match_modulo_twins"] for r in twins)
        parts.insert(1, f"mod-twins {hits}/{len(twins)} ({hits / len(twins):.1%})")
    return "  ".join(parts)


def main():
    print(f"beam width: {BEAM_WIDTH}", flush=True)
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    names = [n for n, _ in SYSTEMS]
    chosen = (ARGS.systems if ARGS and ARGS.systems else names)
    unknown = [n for n in chosen if n not in names]
    assert not unknown, f"unknown rung(s) {unknown}; pick from {names}"
    run_systems = [(n, d) for n, d in SYSTEMS if n in chosen]

    # Resume: whatever is already in output.json stands. A record is finished
    # when it holds an answer for every rung being run, so re-running one rung
    # over a complete file needs --redo to clear that rung first.
    results = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else []
    if results:
        # a mismatch means this output.json belongs to a different dataset or a
        # reordered one; continuing would silently interleave two runs
        last = min(len(results), len(data))
        assert results[last - 1]["question"] == data[last - 1]["corrected_question"], (
            f"{OUT_FILE.name} does not line up with {DATA_FILE.name} at record {last}"
        )

    if ARGS and ARGS.redo:
        cleared = 0
        for rec in results:
            for name, _ in run_systems:
                cleared += rec.pop(f"{name}_canonical", None) is not None
                rec.pop(f"{name}_match", None)
                if name == "generated":
                    rec.pop("match_modulo_twins", None)
        print(f"--redo: cleared {cleared} stored answers for {', '.join(chosen)}", flush=True)

    todo = [i for i in range(len(data))
            if i >= len(results)
            or any(f"{n}_canonical" not in results[i] for n, _ in run_systems)]
    print(f"running [{', '.join(chosen)}] on {len(todo)} of {len(data)} questions", flush=True)
    if not todo:
        print("nothing to do -- every record already has these rungs (use --redo to force)")
        return

    start = time.perf_counter()
    for k, i in enumerate(todo, 1):
        item = data[i]
        if i >= len(results):
            results.append({"question": item["corrected_question"],
                            "gold_canonical": canonicalize(item["sparql_query"])})
        record = results[i]
        gold_c = record["gold_canonical"]
        for name, decode in run_systems:
            if f"{name}_canonical" in record:
                continue  # kept from an earlier run
            try:
                produced = decode(record["question"])
            except Exception as e:  # one bad question must not kill a long run
                produced = f"ERROR: {e}"
            produced_c = canonicalize(produced)
            record[f"{name}_canonical"] = produced_c
            record[f"{name}_match"] = produced_c == gold_c
            if name == "generated":
                # the twin-neutral variant is only reported for the full system
                record["match_modulo_twins"] = (canonicalize_twins(produced)
                                                == canonicalize_twins(item["sparql_query"]))
        if k % CHECKPOINT == 0:
            OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
            per = (time.perf_counter() - start) / k
            print(f"{k}/{len(todo)}  {report(results)}  {per:.2f}s per question", flush=True)

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"done: {report(results)}  "
          f"({elapsed / len(todo):.2f}s per question over the {len(todo)} done here, "
          f"{elapsed / 60:.0f} min this session)")


if __name__ == "__main__":
    main()
