"""
Type-constrained generation.

Step 1: load the fine-tuned BART weights. A local .env file points at the
weights file via MODEL_WEIGHTS (the file holds only the weights, no config),
so the architecture comes from the base facebook/bart-large config.
strict=False because the checkpoint stores the tied embedding matrix
once (as model.shared.weight) and BART's encoder/decoder/lm_head embeddings
are that same tensor.

Step 2: the xgrammar grammars that define legal SPARQL structure -- the five
beginning templates, and the relation grammar (the predicate whitelist with
the type-triple tail folded in, since a type tail always follows an entity
slot). Each is compiled against the BART tokenizer so it can later produce
next-token masks.

Step 3: masked generation, phase 1 -- the beginning template. generate()
decodes greedily, but before choosing each token it asks the grammar matcher
for the bitmask of legal next tokens and applies it to the logits, so the
produced head is always one of the five templates. The finished head is kept
in query_so_far, which later phases (triples, type tail) keep appending to.

Step 4: the state tracker, state = {"idx", "prev"} in generate()'s scope, as
in parse_query. The end of the beginning decides the first triple slot: a
trailing variable (?uri/?x) means the subject is done, so idx = 1 (relation
next) and prev = that variable; a trailing '<' (the ent templates) means
idx = 0 (entity next), prev = None, and the '<' is stripped from
query_so_far (it is re-added together with the entity itself).
"""
import json
import os
from pathlib import Path

import torch
import xgrammar as xgr
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "facebook/bart-large"


def load_env(path=Path(__file__).parent / ".env"):
    """Read KEY=value lines from a local .env file into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

MODEL_WEIGHTS = Path(os.environ.get("MODEL_WEIGHTS", "model/lcquad_finetuned.safetensors"))
if not MODEL_WEIGHTS.is_absolute():
    MODEL_WEIGHTS = Path(__file__).parent / MODEL_WEIGHTS

config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_config(config)
model.load_state_dict(load_file(MODEL_WEIGHTS), strict=False)
model.eval()

print(f"loaded {MODEL_WEIGHTS.name}")

# compiled against the tokenizer, so each grammar can later produce a
# next-token mask, not just accept/reject a finished string
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
compiler = xgr.GrammarCompiler(tokenizer_info)

# the five legal query openings
BEGINNING_TEMPLATE = r"""
root         ::= select_rel | select_ent | count_rel | count_ent | ask_ent

select_rel   ::= "SELECT DISTINCT ?uri WHERE { " var
select_ent   ::= "SELECT DISTINCT ?uri WHERE { <"
count_rel    ::= "SELECT DISTINCT COUNT(?uri) WHERE { " var
count_ent    ::= "SELECT DISTINCT COUNT(?uri) WHERE { <"
ask_ent      ::= "ASK WHERE { <"

var          ::= "?uri" | "?x"
"""
g = compiler.compile_grammar(BEGINNING_TEMPLATE)


def escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


# relation slot: exactly one whitelisted predicate, or the type tail
# ' <rdf:type> <class> }' -- a type tail always follows an entity slot, so it
# lives in the relation grammar; it is the only place a class may appear, and
# it closes the query early. Literals carry a leading and trailing space: the
# leading space glues onto the ' <' token that separates slots in real
# queries, the trailing space separates from the next slot
TBOX_RULES = json.loads(
    (Path(__file__).parent / "tbox_reasoner" / "tbox_rules.json").read_text(encoding="utf-8")
)
RELATIONS = sorted(TBOX_RULES["effective_property_domain_map"])
CLASSES = TBOX_RULES["classes"]
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RELATION_GRAMMAR = (
    "root ::= relation | type_tail\n"
    + "relation ::= " + " | ".join(f'" {escape(r)} "' for r in RELATIONS) + "\n"
    + f'type_tail ::= " {escape(RDF_TYPE)} " class " }}"\n'
    + "class ::= " + " | ".join(f'"{escape(c)}"' for c in CLASSES)
)
relation_grammar = compiler.compile_grammar(RELATION_GRAMMAR)

print(f"compiled grammars: 5 beginning templates, {len(RELATIONS)} relations + type tail, {len(CLASSES)} classes")


def generate(question):
    """Generate a SPARQL query for a natural-language question, one
    grammar-constrained phase at a time. Phase 1: the beginning template."""
    input_ids = tokenizer(question, return_tensors="pt").input_ids

    ids = [config.decoder_start_token_id]  # decoder input; grows across phases
    query_so_far = ""                      # query text; grows across phases

    # phase 1: the beginning template, e.g. 'SELECT DISTINCT ?uri WHERE { ?x'
    matcher = xgr.GrammarMatcher(g, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, config.vocab_size)
    while not matcher.is_terminated():
        matcher.fill_next_token_bitmask(bitmask)
        logits = model(input_ids=input_ids,
                       decoder_input_ids=torch.tensor([ids])).logits[:, -1, :]
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        next_id = int(logits.argmax())
        matcher.accept_token(next_id)
        ids.append(next_id)
    query_so_far += tokenizer.decode(ids[1:], skip_special_tokens=True)

    # state tracking: the end of the beginning decides what comes next. A
    # trailing variable means the subject is done, so a relation comes next
    # (idx 1) and prev records which variable; a trailing '<' opens an entity
    # slot (idx 0, prev None) -- strip the '<' from the text, it is re-added
    # together with the entity itself
    idx = 1 if query_so_far.endswith(("?uri", "?x")) else 0
    prev = "?uri" if query_so_far.endswith("?uri") else ("?x" if query_so_far.endswith("?x") else None)
    state = {"idx": idx, "prev": prev}
    if idx == 0:
        query_so_far = query_so_far[:-1]

    return query_so_far


if __name__ == "__main__":
    print(generate("How many movies did Stanley Kubrick direct?"))
