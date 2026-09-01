"""End-to-end surface-level evaluation over the LC-QuAD test split.

For every test question, generate a SPARQL query with the type-constrained
generator, canonicalize both the produced query and the gold query, and count
exact string matches. Two metrics are reported: strict exact match, and match
"modulo namespace twins", where predicates whitelisted in both the ontology/
and property/ namespaces (e.g. architect) are compared namespace-neutrally.
Every (question, generated, gold, match, ...) record is written to
output.json -- rewritten every CHECKPOINT questions so a crash does not lose
the run.
"""
import json
import re
import time
from pathlib import Path

from type_constrained_generation import generate

DATA_FILE = Path(__file__).parent / "lcquad_data" / "test-data.json"
WHITELIST_FILE = Path(__file__).parent / "lcquad_data" / "predicates.txt"
OUT_FILE = Path(__file__).parent / "output.json"
CHECKPOINT = 10  # questions between progress prints / output.json rewrites


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


def main():
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    results = []
    matches = 0
    twin_matches = 0
    start = time.perf_counter()
    for i, item in enumerate(data, 1):
        question = item["corrected_question"]
        try:
            produced = generate(question)
        except Exception as e:  # one bad question must not kill a long run
            produced = f"ERROR: {e}"
        gen_c = canonicalize(produced)
        gold_c = canonicalize(item["sparql_query"])
        match = gen_c == gold_c
        twin_match = canonicalize_twins(produced) == canonicalize_twins(item["sparql_query"])
        matches += match
        twin_matches += twin_match
        results.append({
            "question": question,
            "generated_query": produced,
            "gold_query": item["sparql_query"],
            "generated_canonical": gen_c,
            "gold_canonical": gold_c,
            "match": match,
            "match_modulo_twins": twin_match,
        })
        if i % CHECKPOINT == 0:
            OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - start
            print(f"{i}/{len(data)}  exact {matches}/{i} ({matches / i:.1%})  "
                  f"mod-twins {twin_matches}/{i} ({twin_matches / i:.1%})  "
                  f"{elapsed / i:.2f}s per query", flush=True)

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"done: exact match {matches}/{len(data)} ({matches / len(data):.1%}), "
          f"modulo twins {twin_matches}/{len(data)} ({twin_matches / len(data):.1%})  "
          f"({elapsed / len(data):.2f}s per query, {elapsed / 60:.0f} min total)")


if __name__ == "__main__":
    main()
