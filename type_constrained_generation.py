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

Step 5: phase 2, the triples loop -- so far only the entity slot (idx 0, the
subject right after an ent beginning). Decoding is guided by the merged
all-entities trie: at the start of a triple no relation has been chosen yet,
so any entity in the KB is legal. This is parse_query's trie_match inverted:
instead of checking a given token against the trie, the logits are masked to
the current trie node's children and the model picks. A terminal node also
offers the glued ' <' that opens the coming relation slot -- that is how the
model says "the entity ends here". The spelling is chosen by a slot-local
beam search (beam_spell, width BEAM_WIDTH) scored by mean log-prob per
token, so a two-token variable spelling no longer beats an entity merely by
being shorter.

Step 6: the relation slot (idx 1). The hard mask is always the whole
relation grammar, so every whitelisted relation stays legal and the type
tail can close the query early. When the subject is an entity rather than a
variable (prev an IRI), the matcher is primed with the glued ' <' the
entity slot ended on, and soft ontological guidance is layered on top: the
subject's classes are found by walking every class trie over the entity's
tokens, relations whose effective domains those classes cover form a small
encouragement trie, and tokens continuing one of them get RELATION_BOOST
added to their logits.

Step 7: the object slot (idx 2) and triple chaining. The object is any
entity or variable from the merged trie; entities inside the relation's
effective range are encouraged via OBJECT_BOOST (variables are not: they sit
in every class trie, so unexcluded they would be boosted too). The class
tries are partitioned by most-specific direct type (extract_entities.py
assigns only the direct type to class_entities.json), so a superclass trie
does NOT contain its subclasses: the range boost walks the range class's
trie and every subclass trie in parallel. The slot ends with the model
choosing ' .' (chain another triple, back to idx 0) or ' }' (close the
query).
"""
import json
import os
import pickle
import time
from dataclasses import dataclass
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

# GPU when one is available (e.g. Colab), CPU otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_config(config)
model.load_state_dict(load_file(MODEL_WEIGHTS), strict=False)
model.eval()
model.to(DEVICE)

print(f"loaded {MODEL_WEIGHTS.name} on {DEVICE}")

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

# entity tries, precomputed by preprocessing/build_class_tries.py: one
# dict-of-dicts token trie per class plus a merged trie over every entity in
# the KB (variables ?uri/?x are in every trie). Generation walks these where
# the parser walked them: a ~1.5M-literal entity alternation cannot be
# compiled by xgrammar, but a trie is walked in O(tokens)
TRIE_END = None  # terminal marker key inside a trie node (must match the pkl)
LT_ID = tokenizer("<", add_special_tokens=False).input_ids[0]        # bare '<'
GL_LT_ID = tokenizer(" <", add_special_tokens=False).input_ids[0]    # glued ' <'
QM_ID = tokenizer("?", add_special_tokens=False).input_ids[0]        # bare '?'
GL_QM_ID = tokenizer(" ?", add_special_tokens=False).input_ids[0]    # glued ' ?'
GL_DOT_ID = tokenizer(" .", add_special_tokens=False).input_ids[0]   # glued ' .'
GL_RBRACE_ID = tokenizer(" }", add_special_tokens=False).input_ids[0]  # glued ' }'

_start = time.time()
with open(Path(__file__).parent / "dbpedia" / "class_tries.pkl", "rb") as f:
    CLASS_TRIES = pickle.load(f)
ALL_ENTITIES_TRIE = CLASS_TRIES["__ALL__"]
print(f"loaded {len(CLASS_TRIES)} class tries in {time.time() - _start:.1f}s")

# ---------------------------------------------------------------------------
# soft ontological guidance for the relation slot after an entity subject
# ---------------------------------------------------------------------------

# added to the logits of tokens that continue an ontologically sound relation;
# the experimental knob of the soft-constraint variant (0 == pure hard
# constraint)
RELATION_BOOST = 5.0
OBJECT_BOOST = 5.0  # same idea, for range-compatible objects (idx 2)

# width of the slot-local beam search over entity spellings (beam_spell);
# 1 reproduces greedy picking exactly. SPARKLE used ~7 over the whole query;
# here it is a per-slot knob to tune
BEAM_WIDTH = 7

EFFECTIVE_PROPERTY_DOMAIN_MAP = TBOX_RULES["effective_property_domain_map"]
EFFECTIVE_PROPERTY_RANGE_MAP = TBOX_RULES["effective_property_range_map"]
OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"

# child class -> all its transitive ancestor classes, inverted from the
# parent -> descendants subsumption map: an entity of class C also covers
# every domain that is an ancestor of C
ANCESTORS = {}
for _parent, _descendants in TBOX_RULES["class_subsumptions"].items():
    for _child in _descendants:
        ANCESTORS.setdefault(_child, set()).add(_parent)


def entity_types(entity_text):
    """All classes of a bracketed entity IRI, found by walking every class
    trie over the entity's tokens in parallel; a class matches when its trie
    reaches a terminal node exactly at the end of the entity.

    This deliberately reuses the already-loaded class tries as a membership
    index rather than building an entity -> types map: the walk is
    ~412 classes x ~12 tokens of dict lookups (microseconds per call), while
    an inverted map would duplicate the 283 MB class_entities.json in RAM.
    """
    entity_ids = tokenizer(entity_text, add_special_tokens=False).input_ids
    types = []
    for class_iri, trie in CLASS_TRIES.items():
        if class_iri == "__ALL__":
            continue
        node = trie
        for tok in entity_ids:
            node = node.get(tok)
            if node is None:
                break
        if node is not None and TRIE_END in node:
            types.append(class_iri)
    return types


def encouraged_relations(types):
    """The whitelisted relations whose effective domains are all covered by
    the given types: a type covers a domain class if it is that class or a
    descendant of it, and owl:Thing domains are covered by everything."""
    encouraged = []
    for rel in RELATIONS:
        if all(
            domain == OWL_THING
            or any(domain == t or domain in ANCESTORS.get(t, ()) for t in types)
            for domain in EFFECTIVE_PROPERTY_DOMAIN_MAP[rel]
        ):
            encouraged.append(rel)
    return encouraged


def build_boost_trie(relations):
    """Dict-of-dicts trie over the tokens of each relation literal (' <iri>'),
    returned primed just past the shared glued ' <' token (that token is the
    entity slot's stop signal, already produced when this runs)."""
    if not relations:
        return {}
    root = {}
    for toks in tokenizer([" " + r for r in relations], add_special_tokens=False).input_ids:
        node = root
        for tok in toks:
            node = node.setdefault(tok, {})
    return root[GL_LT_ID]


def range_tries(relation):
    """The class tries an object of this relation may come from: each
    effective range class plus all its subclasses. The class tries are
    partitioned by most-specific direct type, so subclass entities are NOT
    inside the superclass trie and must be unioned explicitly. owl:Thing
    ranges are unconstrained and yield no tries (no boost)."""
    tries = []
    for cls in EFFECTIVE_PROPERTY_RANGE_MAP[relation]:
        if cls == OWL_THING:
            continue
        for sub in [cls] + TBOX_RULES["class_subsumptions"].get(cls, []):
            if sub in CLASS_TRIES:
                tries.append(CLASS_TRIES[sub])
    return tries


# ---------------------------------------------------------------------------
# slot-local beam search over entity spellings (both entity slots, idx 0/2)
# ---------------------------------------------------------------------------

@dataclass
class Beam:
    """One candidate spelling inside an entity slot: the tokens chosen so
    far, their summed log-prob, and the trie cursors they lead to -- node in
    the merged all-entities trie, boost_nodes in the range class tries walked
    in parallel (object slot only). stop_id is set once the beam picks a
    slot-ending token (' <' / ' .' / ' }'), marking the beam finished."""
    tokens: list
    score: float
    node: dict
    boost_nodes: list
    stop_id: int = None

    def mean_logprob(self):
        """Length-normalised beam score. A variable spelling is ~2 tokens, an
        entity ~10, so raw summed log-probs favour variables -- the entity-drop
        failure mode. Mean log-prob per token makes the two compete fairly."""
        return self.score / len(self.tokens)


def beam_spell(input_ids, ids, node, stop_tokens, boost_nodes=()):
    """Beam-search the spelling of one entity or variable. Starting from the
    trie node, every live beam is expanded by its BEAM_WIDTH best next tokens
    (the trie children, plus the stop tokens when the node is terminal), the
    BEAM_WIDTH best beams by mean log-prob are kept, and this repeats until
    every surviving beam is finished. Returns (spelling token ids, stop token
    id); the stop token is excluded from the spelling because it belongs to
    the next slot. Greedy decoding is the BEAM_WIDTH == 1 special case.
    """
    beams = [Beam([], 0.0, node, list(boost_nodes))]
    while any(b.stop_id is None for b in beams):
        candidates = []
        for b in beams:
            if b.stop_id is not None:
                candidates.append(b)  # finished beams carry over unchanged
                continue
            allowed = [t for t in b.node if t is not None]
            if TRIE_END in b.node:
                allowed += stop_tokens  # the entity may end here
            logits = model(input_ids=input_ids,
                           decoder_input_ids=torch.tensor([ids + b.tokens], device=DEVICE)).logits[:, -1, :]
            mask = torch.full_like(logits, float("-inf"))
            mask[0, allowed] = 0.0
            masked = logits + mask
            if b.boost_nodes:
                boosted = {t for bn in b.boost_nodes for t in bn if t is not None}
                boosted.discard(QM_ID)  # variables stay neutral: no range boost
                masked[0, list(boosted)] += OBJECT_BOOST
            logprobs = torch.log_softmax(masked, dim=-1)[0]
            top = logprobs.topk(BEAM_WIDTH)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                # disallowed tokens (log-prob -inf) match neither branch below
                if TRIE_END in b.node and tok in stop_tokens:
                    candidates.append(Beam(b.tokens + [tok], b.score + lp,
                                           b.node, b.boost_nodes, tok))
                elif tok in b.node:
                    # entering the variable branch forfeits the range boost
                    # for the rest of the spelling
                    next_boost = [] if tok == QM_ID else [bn[tok] for bn in b.boost_nodes if tok in bn]
                    candidates.append(Beam(b.tokens + [tok], b.score + lp,
                                           b.node[tok], next_boost))
        candidates.sort(key=lambda b: b.mean_logprob(), reverse=True)
        beams = candidates[:BEAM_WIDTH]
    return beams[0].tokens[:-1], beams[0].stop_id


def generate(question):
    """Generate a SPARQL query for a natural-language question, one
    grammar-constrained phase at a time. Phase 1: the beginning template.
    Phase 2: the triples loop -- subject, relation and object slots, with
    ontological encouragement for relations after an entity subject and for
    objects inside the relation's range -- which runs until the produced
    text closes the query with '}'."""
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(DEVICE)

    ids = [config.decoder_start_token_id]  # decoder input; grows across phases
    query_so_far = ""                      # query text; grows across phases

    # phase 1: the beginning template, e.g. 'SELECT DISTINCT ?uri WHERE { ?x'
    matcher = xgr.GrammarMatcher(g, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, config.vocab_size)
    while not matcher.is_terminated():
        matcher.fill_next_token_bitmask(bitmask)
        logits = model(input_ids=input_ids,
                       decoder_input_ids=torch.tensor([ids], device=DEVICE)).logits[:, -1, :]
        xgr.apply_token_bitmask_inplace(logits, bitmask.to(DEVICE))
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

    # phase 2: the triples. The independent ifs let a whole triple cascade
    # through a single iteration (subject -> relation -> object). Supports
    # any number of triples: the loop ends only when the produced text closes
    # the query with '}'
    while not query_so_far.endswith("}"):
        if state["idx"] == 0:
            # entity slot: the subject of a triple. Guided by the merged
            # all-entities trie -- no relation has been chosen yet, so no
            # domain/range constraint can apply and any entity (or variable)
            # is legal. Two ways in: an ent beginning template already
            # produced the glued ' <' (its '<' was stripped from query_so_far
            # for bookkeeping), forcing an entity; after a ' .' separator
            # nothing is produced yet, so the model picks the subject kind
            # itself with the glued token -- ' <' (entity) or ' ?'
            # (variable). Either way the glued token maps onto the bare trie
            # prime, exactly as trie_match does
            if tokenizer.decode([ids[-1]]) == " <":
                node = ALL_ENTITIES_TRIE[LT_ID]
                prime_text = "<"
            else:
                logits = model(input_ids=input_ids,
                               decoder_input_ids=torch.tensor([ids], device=DEVICE)).logits[:, -1, :]
                mask = torch.full_like(logits, float("-inf"))
                mask[0, [GL_LT_ID, GL_QM_ID]] = 0.0
                next_id = int((logits + mask).argmax())
                ids.append(next_id)
                node = ALL_ENTITIES_TRIE[LT_ID if next_id == GL_LT_ID else QM_ID]
                prime_text = tokenizer.decode([next_id])
            # the spelling is beam-searched; the stop signal is the glued
            # ' <' that opens the coming relation slot (it belongs to the
            # relation slot, so it is excluded from the entity text)
            ent_ids, stop_id = beam_spell(input_ids, ids, node, [GL_LT_ID])
            ids.extend(ent_ids)
            ids.append(stop_id)
            entity_text = prime_text + tokenizer.decode(ent_ids)
            query_so_far += entity_text
            state["prev"] = entity_text.strip()  # clean bracketed IRI or variable
            state["idx"] = 1                     # subject done, a relation comes next
        if state["idx"] == 1:
            # relation slot. The hard mask is always the full relation
            # grammar: every whitelisted relation stays legal, the model
            # chooses among them, and the type tail can close the query with
            # '}', which ends the triples loop
            rm = xgr.GrammarMatcher(relation_grammar, terminate_without_stop_token=True)
            prefix = ""
            if ids[-1] == GL_LT_ID:
                # the entity slot ended on this slot's glued ' <' (it is the
                # last decoder token), so prime the matcher with it; its text
                # goes back in via the prefix
                rm.accept_token(ids[-1])
                prefix = " <"
            boost_node = None
            if state["prev"] not in ("?uri", "?x"):
                # entity subject: layer soft ontological guidance on top of
                # the hard mask -- tokens continuing a relation whose domains
                # the subject's types cover get RELATION_BOOST
                boost_node = build_boost_trie(encouraged_relations(entity_types(state["prev"])))
            rel_ids = []
            while not rm.is_terminated():
                rm.fill_next_token_bitmask(bitmask)
                logits = model(input_ids=input_ids,
                               decoder_input_ids=torch.tensor([ids], device=DEVICE)).logits[:, -1, :]
                xgr.apply_token_bitmask_inplace(logits, bitmask.to(DEVICE))
                if boost_node:
                    logits[0, list(boost_node)] += RELATION_BOOST
                next_id = int(logits.argmax())
                rm.accept_token(next_id)
                ids.append(next_id)
                rel_ids.append(next_id)
                boost_node = boost_node.get(next_id) if boost_node else None
            rel_text = prefix + tokenizer.decode(rel_ids)
            query_so_far += rel_text
            if not query_so_far.endswith("}"):
                state["idx"] = 2
                state["prev"] = rel_text.strip()  # the bracketed relation IRI
        if state["idx"] == 2:
            # object slot: any entity or variable is legal, so the merged
            # all-entities trie is walked from the root (the relation ended
            # on a standalone space token, so the object starts with a bare
            # '<' or '?'). Entities inside the relation's effective range are
            # encouraged: their class tries are walked in parallel and tokens
            # continuing one of them get OBJECT_BOOST
            boost_nodes = range_tries(state["prev"])
            # beam-searched spelling; stop tokens: ' .' chains another
            # triple, ' }' closes the query (either way the token belongs to
            # what follows, so it is excluded from the entity text)
            ent_ids, stop_id = beam_spell(input_ids, ids, ALL_ENTITIES_TRIE,
                                          [GL_DOT_ID, GL_RBRACE_ID], boost_nodes)
            ids.extend(ent_ids)
            ids.append(stop_id)
            query_so_far += tokenizer.decode(ent_ids) + tokenizer.decode([stop_id])
            state["prev"] = tokenizer.decode(ent_ids)  # last entity/variable produced
            if stop_id == GL_DOT_ID:
                state["idx"] = 0  # ' .' -- the next triple's subject comes next

    return query_so_far


if __name__ == "__main__":
    print(generate("What is the region of Tom Perriello ?"))
