"""Counterfactual 1-hop questions from LC-QuAD

Step 1  regex-select test queries whose WHERE holds exactly one triple
        -> "lcquad 1 hop.json".
Step 2  swap the subject for another test-data entity in the predicate's
        domain; keep it only if the local endpoint holds no triple for it.
Step 3  rewrite the subject's mention in intermediary_question with the new
        entity's name; other mentions keep their wording, brackets dropped.
Step 4  write {question, sparql_query} pairs -> "unanswerable 1 hops.json".

Domain test: a subject satisfies a domain when one of its types IS that class
or a subclass; owl:Thing constrains nothing, so replacements must rest on an
explicit domain (the owl:Thing fallback would admit anything).

Step 2 needs the local Fuseki endpoint running (Fuseki cell of
evaluations and results/dbpedia_endpoint_local.ipynb).

    python unanswerable_questions/build_unanswerables.py
"""
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parent
TEST_DATA = PROJECT / "lcquad_data" / "test-data.json"
TBOX_RULES = PROJECT / "tbox_reasoner" / "tbox_rules.json"
ONE_HOP_FILE = HERE / "lcquad 1 hop.json"
COUNTERFACTUAL_FILE = HERE / "unanswerable 1 hops.json"
ENDPOINT = "http://localhost:3030/dbpedia/sparql"
COUNTERFACTUALS_WANTED = 100
SEED = 0

OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"

# one triple-term: bracketed IRI or variable
TERM = r"(<[^>]*>|\?\w+)"
# a whole single-triple query, SELECT ?uri or SELECT COUNT(?uri). ASK excluded:
# SPARKLE's decoder only checks that an ASK query's entity and relation exist,
# so an ASK counterfactual cannot tell us apart from it
ONE_TRIPLE_QUERY = re.compile(
    r"^\s*SELECT\s+DISTINCT\s+(?:COUNT\(\s*\?uri\s*\)|\?uri)\s+WHERE\s*\{\s*"
    + TERM + r"\s+" + TERM + r"\s+" + TERM + r"\s*\.?\s*\}\s*$",
    re.IGNORECASE,
)
# every triple of any query, for the replacement pool
ANY_TRIPLE = re.compile(TERM + r"\s+" + TERM + r"\s+" + TERM)
BRACKETED = re.compile(r"<([^<>]*)>")


def sparql(query):
    url = ENDPOINT + "?query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def is_entity(term):
    #/resource/ is entities
    return term.startswith("<") and "/resource/" in term


def entity_name(iri):
    """normalise the names"""
    local = urllib.parse.unquote(iri[1:-1].split("/resource/", 1)[1])
    return " ".join(local.replace("_", " ").split())


def ascii_fold(text):
    """LC-QuAD questions drop non-ASCII"""
    return " ".join(text.encode("ascii", "ignore").decode().split()).lower()


def locate_subject(question, subject):
    """(start, end) of the subject's mention in the question, or None.

    No match -> sample is skipped."""
    if not is_entity(subject):
        return None
    name = ascii_fold(entity_name(subject))
    for m in BRACKETED.finditer(question):
        if ascii_fold(m.group(1)) == name:
            return m.span()
    bare = re.search(r"\s+".join(map(re.escape, name.split())), question, re.IGNORECASE)
    return bare.span() if bare else None


def rewrite(question, span, new_subject):
    """Step 3: swap in the new entity's name"""
    a, b = span
    return (question[:a] + entity_name(new_subject) + question[b:]).replace("<", "").replace(">", "")


def main():
    # ---- step 1 ----
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))
    one_hop = [s for s in test_data if ONE_TRIPLE_QUERY.match(s["sparql_query"])]
    ONE_HOP_FILE.write_text(json.dumps(one_hop, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"step 1: {len(one_hop)} single-triple samples -> {ONE_HOP_FILE.name}")

    # ---- step 2 ----
    # replacement pool: every entity in the test data
    entities = set()
    for sample in test_data:
        for s, p, o in (m.groups() for m in ANY_TRIPLE.finditer(sample["sparql_query"])):
            entities.update(t for t in (s, o) if is_entity(t))
    entities = sorted(entities)

    tbox = json.loads(TBOX_RULES.read_text(encoding="utf-8"))
    DOMAIN = tbox["effective_property_domain_map"]
    SUBCLASSES = tbox["class_subsumptions"]
    ancestors = {}
    for parent, descendants in SUBCLASSES.items():
        for child in descendants:
            ancestors.setdefault(child, set()).add(parent)

    # ontology types of every pool entity, per the endpoint
    try:
        sparql("ASK {}")
    except OSError as e:
        raise SystemExit(f"no endpoint at {ENDPOINT} ({e}) -- start Fuseki first "
                         f"(Fuseki cell of dbpedia_endpoint_local.ipynb)")
    classes = set(tbox["classes"])
    types = {e: [] for e in entities}
    for i in range(0, len(entities), 200):
        chunk = entities[i:i + 200]
        res = sparql("SELECT ?e ?t WHERE { VALUES ?e { " + " ".join(chunk) + " } ?e a ?t }")
        for b in res["results"]["bindings"]:
            e, t = f"<{b['e']['value']}>", f"<{b['t']['value']}>"
            if e in types and t in classes and t not in types[e]:
                types[e].append(t)
    print(f"step 2: pool of {len(entities)} entities "
          f"({sum(not t for t in types.values())} untyped)")

    def domain_covered(pred, subject):
        domains = [c for c in DOMAIN[pred] if c != OWL_THING]
        if not domains or not is_entity(subject):
            return True
        return all(any(d == t or d in ancestors.get(t, ()) for t in types[subject])
                   for d in domains)

    def admissible(old, new):
        """T-box test: differs from old subject and object, and falls in an
        explicit domain (a class other than owl:Thing)"""
        s, p, o = new
        return (s != old[0] and s != o
                and any(c != OWL_THING for c in DOMAIN[p])
                and domain_covered(p, s))

    def has_triples(s, p, o):
        """dbo and dbp have a lot of duplication, this is for the mod_twins bit"""
        local = p[1:-1].rsplit("/", 1)[1]
        twins = " ".join(f"<http://dbpedia.org/{ns}/{local}>" for ns in ("ontology", "property"))
        return sparql(f"ASK {{ VALUES ?p {{ {twins} }} {s} ?p {o} }}")["boolean"]

    rng = random.Random(SEED)
    # two samples can share predicate and object; they must not both become
    # the same counterfactual
    counterfactuals, made, failed = [], set(), []
    for sample in one_hop:
        if len(counterfactuals) == COUNTERFACTUALS_WANTED:
            break
        query, question = sample["sparql_query"], sample["intermediary_question"]
        match = ONE_TRIPLE_QUERY.match(query)
        old = match.groups()
        span = locate_subject(question, old[0])
        if span is None:  # subject mention not found -- nothing to rewrite
            failed.append(sample["_id"])
            continue
        found = None
        for term in rng.sample(entities, len(entities)):
            new = (term,) + old[1:]
            if admissible(old, new) and new not in made and not has_triples(*new):
                found = new
                break
        if not found:
            failed.append(sample["_id"])
            continue
        made.add(found)
        a, b = match.span(1)  # group 1 is the subject
        # ---- step 3 ----
        counterfactuals.append({
            "question": rewrite(question, span, found[0]),
            "sparql_query": query[:a] + found[0] + query[b:],
        })
    print(f"        {len(failed)} samples had no admissible empty replacement")
    if len(counterfactuals) < COUNTERFACTUALS_WANTED:
        print(f"        only {len(counterfactuals)} of {COUNTERFACTUALS_WANTED} wanted -- "
              f"the single-triple samples ran out")

    # ---- step 4 ----
    COUNTERFACTUAL_FILE.write_text(json.dumps(counterfactuals, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"step 4: {len(counterfactuals)} counterfactuals -> {COUNTERFACTUAL_FILE.name}")


if __name__ == "__main__":
    main()
