"""Counterfactual 1-hop questions from LC-QuAD (technical_description.md).

Step 1  regex-select every SELECT test query whose WHERE clause holds exactly
        one triple, and write them to "lcquad 1 hop.json". ASK queries are left
        out: SPARKLE's decoder only checks that each entity and relation of an
        ASK query exists, so an ASK counterfactual is as reachable for SPARKLE
        as for us, and cannot tell the two apart.
Step 2  replace one slot of each -- the subject only, as SWAP_SLOTS is set, so
        predicate and object stay as they are -- with another term that occurs
        in the test data, keeping the predicate's domain and/or range under the
        extracted T-box. A replacement is kept only when the local DBpedia
        endpoint holds no triple for it; otherwise the next one is tried.
        Samples are taken in file order until 100 counterfactuals are made.
        A sample whose subject is the answer variable has nothing to swap.
Step 3  rewrite each <...> span of the sample's intermediary_question with the
        standardised name of the identifier now in that slot of the query, then
        drop the angle brackets.
Step 4  write the {question, sparql_query} pairs to "counterfactual 1 hops.json".

T-box semantics are the decoder's, as dbpedia_endpoint_local.ipynb mirrors them:
a subject satisfies a domain when one of its types IS that class or a subclass of
it, an object satisfies a range likewise, and owl:Thing constrains nothing. So
every replacement must rest on an explicit domain or range -- one naming a class
other than owl:Thing. The owl:Thing fallback (every property/ predicate, and
dbpedia.org/ontology ones declaring nothing) would admit any term at all.

Step 2 needs the local Fuseki endpoint running (the Fuseki cell of
evaluations and results/dbpedia_endpoint_local.ipynb).

    python counterfactuals/build_counterfactuals.py
"""
import json
import random
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parent
TEST_DATA = PROJECT / "lcquad_data" / "test-data.json"
TBOX_RULES = PROJECT / "tbox_reasoner" / "tbox_rules.json"
ONE_HOP_FILE = HERE / "lcquad 1 hop.json"
COUNTERFACTUAL_FILE = HERE / "counterfactual 1 hops.json"
ENDPOINT = "http://localhost:3030/dbpedia/sparql"
COUNTERFACTUALS_WANTED = 100
SEED = 0
# which triple slots may be replaced: 0 subject, 1 predicate, 2 object
SWAP_SLOTS = (0,)

OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
SLOT_NAMES = ("subject", "predicate", "object")

# a triple term: a bracketed IRI or a variable
TERM = r"(<[^>]*>|\?\w+)"
# a whole query of one triple, under either LC-QuAD 1-hop SELECT head:
# SELECT ?uri or SELECT COUNT(?uri)
ONE_TRIPLE_QUERY = re.compile(
    r"^\s*SELECT\s+DISTINCT\s+(?:COUNT\(\s*\?uri\s*\)|\?uri)\s+WHERE\s*\{\s*"
    + TERM + r"\s+" + TERM + r"\s+" + TERM + r"\s*\.?\s*\}\s*$",
    re.IGNORECASE,
)
# every triple of any query, for collecting the replacement pools
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


def predicate_name(iri):
    """<.../ontology/parentOrganisation> -> "parent organisation"."""
    local = iri[1:-1].rsplit("/", 1)[1]
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", local)
    return " ".join(re.sub(r"[\W_]+", " ", words).split()).lower()


def standard_name(term):
    return entity_name(term) if is_entity(term) else predicate_name(term)


def ascii_fold(text):
    """What LC-QuAD's question text keeps of a name: the questions drop every
    non-ASCII character, so Trần_Việt_Hương is asked about as "Trn Vit Hng".
    Case-folded and single-spaced for comparison."""
    return " ".join(text.encode("ascii", "ignore").decode().split()).lower()


def locate(question, terms):
    """slot -> (start, end) of each identifier's mention in the question.

    Entities are found by name: as a <...> span when bracketed, as bare text
    otherwise (template 2 questions leave the entity unbracketed). The predicate
    is the last <...> span left over -- the only other one, in the "What is the
    <class> whose <predicate> is <entity>" forms, is the answer class, which
    comes first and names nothing in the query. A slot that cannot be found is
    left out, and is then never replaced, since its question could not follow."""
    spans, claimed = {}, set()
    brackets = [(m.start(), m.end(), m.group(1)) for m in BRACKETED.finditer(question)]
    for slot in (0, 2):
        if not is_entity(terms[slot]):
            continue
        name = ascii_fold(entity_name(terms[slot]))
        hit = next((b for b in brackets if b not in claimed and ascii_fold(b[2]) == name), None)
        if hit:
            claimed.add(hit)
            spans[slot] = hit[:2]
            continue
        bare = re.search(r"\s+".join(map(re.escape, name.split())), question, re.IGNORECASE)
        if bare:
            spans[slot] = bare.span()
    left = [b for b in brackets if b not in claimed]
    if left:
        spans[1] = left[-1][:2]
    return spans


def rewrite(question, spans, terms):
    """Step 3: every located mention becomes the standardised name of the
    identifier now in its slot; whatever angle brackets remain go."""
    for slot, (a, b) in sorted(spans.items(), key=lambda kv: kv[1][0], reverse=True):
        question = question[:a] + standard_name(terms[slot]) + question[b:]
    return question.replace("<", "").replace(">", "")


def main():
    # ---- step 1 ----
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))
    one_hop = [s for s in test_data if ONE_TRIPLE_QUERY.match(s["sparql_query"])]
    ONE_HOP_FILE.write_text(json.dumps(one_hop, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"step 1: {len(one_hop)} single-triple samples -> {ONE_HOP_FILE.name}")

    # ---- step 2 ----
    # replacement pools: every entity and predicate anywhere in the test data
    entities, predicates = set(), set()
    for sample in test_data:
        for s, p, o in (m.groups() for m in ANY_TRIPLE.finditer(sample["sparql_query"])):
            if p != RDF_TYPE:
                predicates.add(p)
            entities.update(t for t in (s, o) if is_entity(t))
    entities, predicates = sorted(entities), sorted(predicates)

    tbox = json.loads(TBOX_RULES.read_text(encoding="utf-8"))
    DOMAIN = tbox["effective_property_domain_map"]
    RANGE = tbox["effective_property_range_map"]
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
    print(f"step 2: pools of {len(entities)} entities ({sum(not t for t in types.values())} "
          f"untyped) and {len(predicates)} predicates")

    def domain_covered(pred, subject):
        domains = [c for c in DOMAIN[pred] if c != OWL_THING]
        if not domains or not is_entity(subject):
            return True
        return all(any(d == t or d in ancestors.get(t, ()) for t in types[subject])
                   for d in domains)

    def range_admits(pred, obj):
        wanted = [c for c in RANGE[pred] if c != OWL_THING]
        if not wanted or not is_entity(obj):
            return True
        allowed = set(wanted).union(*(SUBCLASSES.get(c, []) for c in wanted))
        return any(t in allowed for t in types[obj])

    def explicit(classes):
        return any(c != OWL_THING for c in classes)

    def admissible(old, new, slot):
        """The T-box test for replacing old[slot]: a new subject must fall in the
        predicate's explicit domain, a new object in its explicit range. A new
        predicate must keep the old one's explicit domain or explicit range (or
        both), must still admit whatever entities the triple holds, and must
        read differently in the question. Where the subject is the answer
        variable it must keep the domain too: those questions name the answer's
        class ("What is the <software> whose <developer> is ..."), and a new
        domain would leave that word describing the wrong thing."""
        s, p, o = new
        if new[slot] == old[slot] or s == o:
            return False
        if slot == 0:
            return explicit(DOMAIN[p]) and domain_covered(p, s)
        if slot == 2:
            return explicit(RANGE[p]) and range_admits(p, o)
        q = old[1]
        if s.startswith("?") and set(DOMAIN[p]) != set(DOMAIN[q]):
            return False
        keeps = (explicit(DOMAIN[p]) and set(DOMAIN[p]) == set(DOMAIN[q])
                 or explicit(RANGE[p]) and set(RANGE[p]) == set(RANGE[q]))
        return (keeps and domain_covered(p, s) and range_admits(p, o)
                and predicate_name(p) != predicate_name(q))

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
    counterfactuals, made, replaced, failed = [], set(), Counter(), []
    for sample in one_hop:
        if len(counterfactuals) == COUNTERFACTUALS_WANTED:
            break
        query, question = sample["sparql_query"], sample["intermediary_question"]
        match = ONE_TRIPLE_QUERY.match(query)
        old = list(match.groups())
        spans = locate(question, old)
        slots = [slot for slot in sorted(spans) if slot in SWAP_SLOTS]
        rng.shuffle(slots)
        found = None
        for slot in slots:
            pool = predicates if slot == 1 else entities
            for term in rng.sample(pool, len(pool)):
                new = old.copy()
                new[slot] = term
                if (admissible(old, new, slot) and tuple(new) not in made
                        and not has_triples(*new)):
                    found = slot, new
                    break
            if found:
                break
        if not found:
            failed.append(sample["_id"])
            continue
        slot, new = found
        made.add(tuple(new))
        a, b = match.span(slot + 1)
        # ---- step 3 ----
        counterfactuals.append({
            "question": rewrite(question, spans, new),
            "sparql_query": query[:a] + new[slot] + query[b:],
        })
        replaced[SLOT_NAMES[slot]] += 1
    print(f"        replaced: {dict(replaced)}; {len(failed)} samples had no admissible "
          f"empty replacement")
    if len(counterfactuals) < COUNTERFACTUALS_WANTED:
        print(f"        only {len(counterfactuals)} of {COUNTERFACTUALS_WANTED} wanted -- "
              f"the single-triple samples ran out")

    # ---- step 4 ----
    COUNTERFACTUAL_FILE.write_text(json.dumps(counterfactuals, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"step 4: {len(counterfactuals)} counterfactuals -> {COUNTERFACTUAL_FILE.name}")


if __name__ == "__main__":
    main()
