"""Counterfactual 1-hop questions from LC-QuAD (technical_description.md).

Step 1  regex-select every test query whose WHERE clause holds exactly one
        triple, and write them to "lcquad 1 hop.json".
Step 2  replace the subject of each -- predicate and object stay as they are --
        with another entity that occurs in the test data and falls in the
        predicate's domain under the extracted T-box. A replacement is kept
        only when the local DBpedia endpoint holds no triple for it; otherwise
        the next one is tried. Samples are taken in file order until 100
        counterfactuals are made. A sample whose subject is the answer variable
        has nothing to swap.
Step 3  rewrite the subject's mention in the sample's intermediary_question
        with the name of the entity now in its place; the predicate and object
        mentions keep the question's own wording, with the angle brackets
        dropped.
Step 4  write the {question, sparql_query} pairs to "unanswerable 1 hops.json".

T-box semantics are the decoder's, as dbpedia_endpoint_local.ipynb mirrors them:
a subject satisfies a domain when one of its types IS that class or a subclass of
it, and owl:Thing constrains nothing. So every replacement must rest on an
explicit domain -- one naming a class other than owl:Thing. The owl:Thing
fallback (every property/ predicate, and dbpedia.org/ontology ones declaring
nothing) would admit any term at all.

Step 2 needs the local Fuseki endpoint running (the Fuseki cell of
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

# a triple term: a bracketed IRI or a variable
TERM = r"(<[^>]*>|\?\w+)"
# a whole query of one triple, under either of the two SELECT heads:
# SELECT ?uri or SELECT COUNT(?uri). ASK queries are out: SPARKLE's decoder
# only checks that an ASK query's entity and relation exist, so an ASK
# counterfactual is as reachable for it as for us and cannot tell the two apart
ONE_TRIPLE_QUERY = re.compile(
    r"^\s*SELECT\s+DISTINCT\s+(?:COUNT\(\s*\?uri\s*\)|\?uri)\s+WHERE\s*\{\s*"
    + TERM + r"\s+" + TERM + r"\s+" + TERM + r"\s*\.?\s*\}\s*$",
    re.IGNORECASE,
)
# every triple of any query, for collecting the replacement pool
ANY_TRIPLE = re.compile(TERM + r"\s+" + TERM + r"\s+" + TERM)
BRACKETED = re.compile(r"<([^<>]*)>")


def sparql(query):
    url = ENDPOINT + "?query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def is_entity(term):
    return term.startswith("<") and "/resource/" in term


def entity_name(iri):
    """<.../resource/John_Fanning_(businessman)> -> "John Fanning (businessman)"."""
    local = urllib.parse.unquote(iri[1:-1].split("/resource/", 1)[1])
    return " ".join(local.replace("_", " ").split())


def ascii_fold(text):
    """What LC-QuAD's question text keeps of a name: the questions drop every
    non-ASCII character, so Trần_Việt_Hương is asked about as "Trn Vit Hng".
    Case-folded and single-spaced for comparison."""
    return " ".join(text.encode("ascii", "ignore").decode().split()).lower()


def locate_subject(question, subject):
    """(start, end) of the subject's mention in the question, or None.

    Found by name: as a <...> span when bracketed, as bare text otherwise
    (template 2 questions leave the entity unbracketed). A sample whose subject
    mention cannot be found is skipped, since its question could not follow."""
    if not is_entity(subject):
        return None
    name = ascii_fold(entity_name(subject))
    for m in BRACKETED.finditer(question):
        if ascii_fold(m.group(1)) == name:
            return m.span()
    bare = re.search(r"\s+".join(map(re.escape, name.split())), question, re.IGNORECASE)
    return bare.span() if bare else None


def rewrite(question, span, new_subject):
    """Step 3: the subject's mention becomes the new entity's name; whatever
    angle brackets remain (the unchanged predicate and object mentions) go."""
    a, b = span
    return (question[:a] + entity_name(new_subject) + question[b:]).replace("<", "").replace(">", "")


def main():
    # ---- step 1 ----
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))
    one_hop = [s for s in test_data if ONE_TRIPLE_QUERY.match(s["sparql_query"])]
    ONE_HOP_FILE.write_text(json.dumps(one_hop, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"step 1: {len(one_hop)} single-triple samples -> {ONE_HOP_FILE.name}")

    # ---- step 2 ----
    # replacement pool: every entity anywhere in the test data
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

    # ontology types of every pool entity, as the endpoint holds them
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
        """The T-box test for a new subject: it must differ from the old subject
        and from the object, and it must fall in the predicate's explicit domain
        -- one naming a class other than owl:Thing."""
        s, p, o = new
        return (s != old[0] and s != o
                and any(c != OWL_THING for c in DOMAIN[p])
                and domain_covered(p, s))

    def has_triples(s, p, o):
        """Whether the store answers the triple. Asked under both namespace
        twins of the predicate, because the rewritten question cannot tell
        ontology/architect from property/architect."""
        local = p[1:-1].rsplit("/", 1)[1]
        twins = " ".join(f"<http://dbpedia.org/{ns}/{local}>" for ns in ("ontology", "property"))
        return sparql(f"ASK {{ VALUES ?p {{ {twins} }} {s} ?p {o} }}")["boolean"]

    rng = random.Random(SEED)
    # triples already made: two samples can share a predicate and object, and
    # must not then both become the same counterfactual
    counterfactuals, made, failed = [], set(), []
    for sample in one_hop:
        if len(counterfactuals) == COUNTERFACTUALS_WANTED:
            break
        query, question = sample["sparql_query"], sample["intermediary_question"]
        match = ONE_TRIPLE_QUERY.match(query)
        old = match.groups()
        span = locate_subject(question, old[0])
        if span is None:  # the subject's mention was not found -- nothing to rewrite
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
