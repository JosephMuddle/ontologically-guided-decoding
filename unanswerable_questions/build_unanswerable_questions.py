"""Unanswerable LC-QuAD questions: the test samples whose gold query has an empty
answer set.

An LC-QuAD gold query answers with a set of rows, so it is unanswerable when that
set is empty against DBpedia -- the questions the constrained decoder is meant to
still answer sensibly. The split files hold only questions and queries, no
answers, so emptiness is measured, not read: every test query that begins with
SELECT is run against the local DBpedia 2016-04 store (the same Fuseki endpoint
build_counterfactuals.py and the execution evaluation use) and kept when it
returns no rows.

A gold query carries no LIMIT of its own, so each is capped at LIMIT 1 on the way
out: a single row already proves the answer set non-empty and lets the store stop
early. COUNT queries wrap an aggregate, so their one row can never be empty, and
ASK queries answer with a boolean rather than an answer set -- the two are counted
and reported, not written.

Step 1  run every SELECT query of the test split against the endpoint, several at
        a time, and record the ones that come back with no rows.
Step 2  write those samples whole, in test-file order and with LC-QuAD's
        corrected_question renamed to question, to "unanswerable questions.json".

Step 1 needs the local Fuseki endpoint running (the Fuseki cell of
evaluations and results/dbpedia_endpoint_local.ipynb).

    python unanswerable_questions/build_unanswerable_questions.py
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parent
TEST_DATA = PROJECT / "lcquad_data" / "test-data.json"
UNANSWERABLE_FILE = HERE / "unanswerable questions.json"
ENDPOINT = "http://localhost:3030/dbpedia/sparql"
WORKERS = 8  # queries in flight at once
CHECKPOINT = 100  # queries between progress prints

# a query is run only when SELECT is its first token: COUNT wraps an aggregate and
# is known never to be empty, and ASK answers with a boolean
BEGINS_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
IS_COUNT = re.compile(r"\bCOUNT\s*\(", re.IGNORECASE)
HAS_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)


def sparql(query):
    url = ENDPOINT + "?query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def emptiness_query(query):
    """The query capped at one row, so the store stops as soon as it has proved
    the answer set non-empty. LC-QuAD's gold queries carry no LIMIT, but one is
    left alone if it turns up."""
    q = query.strip().rstrip(";")
    return q if HAS_LIMIT.search(q) else f"{q} LIMIT 1"


def is_empty(sample):
    """(empty, error) for one sample. empty is False on error, which is reported
    separately rather than folded into the unanswerable questions."""
    try:
        result = sparql(emptiness_query(sample["sparql_query"]))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError) as e:
        return False, str(e)
    return len(result["results"]["bindings"]) == 0, None


def main():
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))

    # ---- step 1 ----
    runs, counts = [], []
    for sample in test_data:
        query = sample["sparql_query"]
        if not BEGINS_SELECT.match(query):
            continue
        (counts if IS_COUNT.search(query) else runs).append(sample)
    try:
        sparql("ASK {}")
    except OSError as e:
        raise SystemExit(f"no endpoint at {ENDPOINT} ({e}) -- start Fuseki first "
                         f"(Fuseki cell of dbpedia_endpoint_local.ipynb)")
    print(f"step 1: {len(runs)} SELECT answer sets to measure "
          f"({len(counts)} COUNT queries are never empty, "
          f"{len(test_data) - len(runs) - len(counts)} ASK queries answer with a boolean)")

    unanswerable, errors = [], []
    # map hands the results back in test-file order, so the run is reproducible
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = pool.map(is_empty, runs)
        for k, (sample, (empty, error)) in enumerate(zip(runs, results), 1):
            if error:
                errors.append((sample["_id"], error))
            elif empty:
                unanswerable.append(sample)
            if k % CHECKPOINT == 0:
                print(f"        {k}/{len(runs)} queries run, "
                      f"{len(unanswerable)} empty, {len(errors)} errored", flush=True)
    print(f"        {len(unanswerable)} of {len(runs)} SELECT queries have an empty answer set")
    for _id, error in errors:
        print(f"        error on {_id}: {error}")

    # ---- step 2 ----
    records = [{"_id": s["_id"],
                "question": s["corrected_question"],
                "intermediary_question": s["intermediary_question"],
                "sparql_query": s["sparql_query"],
                "sparql_template_id": s["sparql_template_id"]}
               for s in unanswerable]
    UNANSWERABLE_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"step 2: {len(records)} unanswerable questions -> {UNANSWERABLE_FILE.name}")


if __name__ == "__main__":
    main()
