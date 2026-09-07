"""
Type-constrained generation.

Step 1: load the merged Qwen2.5-Coder-1.5B checkpoint produced by the
fine-tuning notebook. Qwen is a decoder-only causal language model, so the
question prompt and generated SPARQL share one token sequence.

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
model says "the entity ends here". Spellings are not committed slot by slot:
the whole query is beam-searched (whole_query_beam, width BEAM_WIDTH), so an
entity chosen here can still lose to an alternative once the rest of the
triple has been scored.

Step 6: the relation slot (idx 1). The hard mask is always the whole
relation grammar, so every whitelisted relation stays legal and the type
tail can close the query early. When the subject is an entity rather than a
variable (prev an IRI), the matcher is primed with the glued ' <' the
entity slot ended on, and soft ontological guidance is layered on top: the
subject's classes are found by walking every class trie over the entity's
tokens, relations whose effective domains those classes cover form a small
encouragement trie, and tokens continuing one of them are boosted when the
beam picks which continuations to expand. The score is settled once, at the
end of the slot: a relation that turns out not to be one of them costs
RELATION_BOOST, however many tokens it spanned.

Step 7: the object slot (idx 2) and triple chaining. The object is any
entity or variable from the merged trie; entities inside the relation's
effective range are encouraged via OBJECT_BOOST, again settled once at the
end of the slot. A range says which entity an object may be, not whether it
is an entity at all, so the boost stays out of the slot's first token -- the
one that picks '<' or '?' -- and a variable object is neither encouraged nor
charged. Renormalisation is why that has to be explicit: boosting only the
'<' branch there does not leave '?' neutral, it prices it out of reach by the
whole of OBJECT_BOOST. The class tries are partitioned by most-specific
direct type (extract_entities.py assigns only the direct type to
class_entities.json), so a superclass trie does NOT contain its subclasses:
the range boost walks the range class's trie and every subclass trie in
parallel. The slot ends with the model
choosing ' .' (chain another triple, back to idx 0) or ' }' (close the
query).
"""
import json
import os
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import xgrammar as xgr
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B")


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

MODEL_WEIGHTS = Path(os.environ.get("MODEL_WEIGHTS", "model/qwen_lcquad.safetensors"))
if not MODEL_WEIGHTS.is_absolute():
    MODEL_WEIGHTS = Path(__file__).parent / MODEL_WEIGHTS

# GPU when one is available (e.g. Colab), CPU otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Fine-tuned weights not found at {MODEL_WEIGHTS}. Set MODEL_WEIGHTS to "
        "the .safetensors file saved by fine_tune_qwen.ipynb."
    )

# The notebook fine-tunes every weight of MODEL_ID without touching the
# architecture or the vocabulary, so config and tokenizer still come from the
# base model and only the weights are local. Building from the config means no
# base weights are fetched -- they would all be overwritten anyway.
config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_config(config)

# Qwen2.5 ties lm_head to the input embedding, and save_pretrained drops the
# duplicate, so lm_head.weight is absent from the file: load non-strictly and
# re-tie. Any OTHER missing or unexpected key means these weights do not belong
# to this architecture, which would otherwise leave a silently random model
missing, unexpected = model.load_state_dict(load_file(MODEL_WEIGHTS), strict=False)
missing = [k for k in missing if k != "lm_head.weight"]
if missing or unexpected:
    raise RuntimeError(
        f"{MODEL_WEIGHTS} does not match {MODEL_ID}: "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}"
    )
model.tie_weights()

model.to(dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32)
model.eval()
model.to(DEVICE)

print(f"loaded {MODEL_WEIGHTS.name} into {MODEL_ID} on {DEVICE}")


def decoder_step(input_ids, cache, attention_mask):
    """One decoding step for the whole beam at once: next-token logits for every
    row, and the KV cache grown by the tokens just fed in.

    Row i of every argument and of the result belongs to live hypothesis i. On a
    query's first step `cache` is None and `input_ids` is the whole prompt; from
    then on the cache already holds every token but the newest, so exactly one
    token per row goes through the model. Without this the beam re-read its
    entire prefix at every token of every hypothesis, which is quadratic in the
    query length and linear in the beam width on top."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def reparent_cache(cache, rows):
    """Re-order the cache so that row i holds the keys and values of old row
    rows[i]. Repeats are allowed -- that is how the single prompt row fans out
    into a full beam on the first step -- and omissions are how a hypothesis that
    died or finished releases its row."""
    idx = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(idx)  # transformers Cache objects reorder in place
        return cache
    # legacy format: one (key, value) pair of [batch, heads, seq, dim] per layer
    return tuple(tuple(t.index_select(0, idx) for t in layer) for layer in cache)


# compiled against the tokenizer, so each grammar can later produce a
# next-token mask, not just accept/reject a finished string
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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

# grammar-only rung, used by generate_grammar_only(): ONE grammar for the whole
# query rather than one per slot. Every position accepts any well-formed IRI or
# variable on shape alone -- no trie membership check and no predicate
# whitelist, since three IRIs in a row is perfectly valid SPARQL.
#
# The single grammar is what makes it work. Per-slot grammars ended at a slot
# boundary, which masked out exactly the tokens BPE actually produces there: a
# grammar ending at ">" rejects "> " and "> <" for overrunning it, leaving only
# the bare ">" token, which the model almost never emits in that position. It
# then never closed the IRI and ran away, reaching for oddities like the
# <|fim_suffix|> special token whose spelling happens to end in ">". With one
# grammar spanning the query no boundary is forced anywhere, so the natural
# glued tokens stay legal.
QUERY_TEMPLATE = r"""
root    ::= head triples
head    ::= "SELECT DISTINCT ?uri WHERE { " | "SELECT DISTINCT COUNT(?uri) WHERE { " | "ASK WHERE { "
triples ::= triple (" . " triple)* " }"
triple  ::= term " " term " " term
term    ::= "?uri" | "?x" | "<" body ">"
body    ::= [^<> ]+
"""
query_grammar = compiler.compile_grammar(QUERY_TEMPLATE)

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

# the price of ontological incompatibility, in nats of query score: charged
# once on a relation whose effective domain the subject's types do NOT cover,
# and once on an object outside the relation's effective range. The same number
# also biases which continuations the beam expands, at every token of the slot
# -- but only the one-off reaches the score, so the guidance is worth a fixed
# amount rather than a multiple of it that depends on how BPE happened to split
# the IRI. A penalty and not a credit so that a score can still only fall, which
# is what lets whole_query_beam() retire a hypothesis for good.
# The experimental knob of the soft-constraint variant (0 == pure hard
# constraint)
RELATION_BOOST = 5.0
OBJECT_BOOST = 5.0  # same idea, for range-compatible objects (idx 2)

# width of the whole-query beam search (whole_query_beam); 1 reproduces greedy
# decoding exactly. SPARKLE used ~7 over the whole query, which is now the same
# quantity this sets. Read from the environment so test.py's --beams can set it
# without editing this file
BEAM_WIDTH = int(os.getenv("BEAM_WIDTH", "4"))

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
# whole-query beam search
# ---------------------------------------------------------------------------

@dataclass
class Hyp:
    """One whole-query hypothesis: the tokens chosen so far, their summed
    log-prob, and the constraint state that decides what may legally come next.

    Scoring is the plain sum of log-probs RENORMALISED over the legal tokens,
    less one ontological penalty per incompatible slot, and no length
    normalisation. A token the constraints force therefore costs about nothing.

    Pricing the boosts into the score makes it a guided objective rather than a
    plain log P(query|question): that is deliberate. Left out of the score
    entirely, a boost could only change which candidates got expanded, and since
    a trie node usually has fewer children than BEAM_WIDTH there was nothing to
    expand differently -- it altered 2 outputs in 1000 and flipped no matches.
    Soft guidance has to price the path to do anything at all.

    But it has to price it ONCE. Added to the logits at every token of the slot,
    as it was, a boost survived the renormalisation only at the steps where some
    legal token was left unboosted, so what a compatible path actually collected
    was BOOST times the number of such steps: 5 nats for a relation that parts
    from its rivals in a single token, ten times that for a long object IRI that
    stays inside its range trie all the way down. The strength of the knob was
    therefore set by the tokenizer, and the longer an in-range object the more
    it earned -- the length preference the search is otherwise careful not to
    have. So the boosts now only steer which continuations get expanded
    (legal_logits returns a separate tensor for that) and the score settles up
    once per slot, in advance(): BOOST off a relation whose domain the subject's
    types do not cover, BOOST off an object that ends outside the relation's
    range. Charged that way round -- against the incompatible rather than for
    the compatible -- a score still only falls, which is what lets the search
    retire a hypothesis the moment it drops below a finished one. What is left
    of the length coupling is small and explicit: a query with more triples has
    more slots that can each be wrong once.

    Scoring the unmasked distribution instead looks more principled and is a
    trap: the tighter a rung constrains, the more often the model is pushed onto
    tokens it rates poorly, so every extra token bleeds score and closing the
    query early becomes the cheapest way to stop paying. Measured on a 50-question
    run that produced one-triple queries 30 times against gold's 10, and never
    once produced more triples than gold -- a one-sided error, the signature of a
    search bias rather than a modelling one. These scores are only ever compared
    within a single decode, never across rungs, so renormalising costs nothing.

    Hypotheses are copied on every branch. Trie nodes are read-only dicts and are
    shared; the xgrammar matchers are stateful and so are never stored -- they
    are rebuilt from slot_ids on demand, a handful of accept_token calls against
    the one batched model forward the whole beam shares each step.
    """
    ids: list                 # prompt + generated tokens
    gen: list                 # generated tokens only -- the query
    score: float              # renormalised log-prob - slot penalties
    mode: str                 # 'constrained' (tries + whitelist + boosts),
                              # 'tries' (the same minus the boosts) or 'grammar'
    slot: str                 # query (grammar-only) | begin | open | subject |
                              # relation | object (the trie rungs)
    slot_ids: list            # tokens of the current slot, for replay and prev
    slot_prime: int = None    # already-emitted token that primes this slot's matcher
    slot_prefix: str = ""     # text of this slot already in gen but not in slot_ids
    node: dict = None         # cursor in the merged all-entities trie
    boost_nodes: tuple = ()   # range-class trie cursors (object slot)
    boost_node: dict = None   # encouraged-relation trie cursor (relation slot)
    encouraged: frozenset = frozenset()  # relations the subject's types allow
    ranged: bool = False      # the relation's range is narrower than owl:Thing
    prev: str = None          # previous term, for the ontological boosts
    done: bool = False

    def text(self):
        return tokenizer.decode(self.gen)

    def matcher(self, grammar):
        m = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        if self.slot_prime is not None:
            m.accept_token(self.slot_prime)
        for tok in self.slot_ids:
            m.accept_token(tok)
        return m

    def _grammar(self):
        if self.slot == "query":
            return query_grammar  # grammar-only: one grammar, no slot cycle at all
        if self.slot == "begin":
            return g
        if self.slot == "relation":
            return relation_grammar
        return None  # constrained entity slots are trie-driven, not grammar-driven

    def legal_logits(self, logits, bitmask):
        """(ranking, legal): the next-token logits with everything the
        constraints forbid set to -inf, twice -- once with the ontological
        boosts added on top of what survives, once without.

        The search expands the top of `ranking` and scores what it expanded
        against `legal`, so a boost decides which continuations are tried
        without also inflating the score of every token it happens to touch;
        the flat per-slot price is charged in advance() instead. The two are
        the same tensor wherever no boost applies. The boosts go in in place:
        an illegal token is already -inf and -inf + boost is still -inf, so
        nothing the constraints ruled out can come back through the boost."""
        grammar = self._grammar()
        if grammar is not None:
            legal = logits.clone()
            self.matcher(grammar).fill_next_token_bitmask(bitmask)
            xgr.apply_token_bitmask_inplace(legal, bitmask.to(DEVICE))
            if self.slot == "relation" and self.boost_node:
                ranking = legal.clone()
                ranking[0, list(self.boost_node)] += RELATION_BOOST
                return ranking, legal
            return legal, legal
        if self.slot == "open":
            allowed = [GL_LT_ID, GL_QM_ID]
        else:
            allowed = [t for t in self.node if t is not None]
            if TRIE_END in self.node:
                allowed += [GL_LT_ID] if self.slot == "subject" else [GL_DOT_ID, GL_RBRACE_ID]
        legal = torch.full_like(logits, float("-inf"))
        legal[0, allowed] = logits[0, allowed]
        # from the object's SECOND token on: the first one chooses between an
        # entity and a variable, which the range has no opinion about, and
        # boosting one branch of a two-way choice is not neutrality -- it is the
        # full boost against the other
        if self.slot == "object" and self.slot_ids and self.boost_nodes:
            boosted = {t for bn in self.boost_nodes for t in bn if t is not None}
            if boosted:
                ranking = legal.clone()
                ranking[0, list(boosted)] += OBJECT_BOOST  # illegal ones stay -inf
                return ranking, legal
        return legal, legal

    def advance(self, tok, logprob):
        """The transition: a copy of this hypothesis with tok appended and the
        constraint state moved on. Mirrors the slot cycle of the phase-by-phase
        decoder -- begin, then subject/relation/object until the text closes
        the query with a brace."""
        h = replace(self, ids=self.ids + [tok], gen=self.gen + [tok],
                    score=self.score + logprob, slot_ids=self.slot_ids + [tok])

        if h.slot == "query":
            # grammar-only: one matcher spans the whole query, so there is no
            # slot bookkeeping and nothing to transition between
            h.done = h.matcher(query_grammar).is_terminated()
            return h

        if h.slot == "begin":
            if h.matcher(g).is_terminated():
                text = h.text()
                if text.endswith(("?uri", "?x")):
                    # a trailing variable means the subject is already done
                    h.prev = "?uri" if text.endswith("?uri") else "?x"
                    h.slot, h.slot_ids = "relation", []
                    h.slot_prime, h.slot_prefix = None, ""
                else:
                    # a trailing bracket opens an entity slot: the glued token is
                    # already emitted, so prime the trie past its bare bracket
                    h.slot, h.slot_ids = "subject", []
                    h.node, h.slot_prefix = ALL_ENTITIES_TRIE[LT_ID], "<"
            return h

        if h.slot == "open":  # constrained only: the glued opener was just chosen
            h.node = ALL_ENTITIES_TRIE[LT_ID if tok == GL_LT_ID else QM_ID]
            h.slot = "subject"
            return h

        if h.slot == "subject":
            if tok == GL_LT_ID and TRIE_END in self.node:
                # the entity ends here; the glued token belongs to the relation
                h.prev = (self.slot_prefix + tokenizer.decode(self.slot_ids)).strip()
                h.slot, h.slot_ids, h.node = "relation", [], None
                h.slot_prime, h.slot_prefix = GL_LT_ID, " <"
                encouraged = (
                    encouraged_relations(entity_types(h.prev))
                    if h.mode == "constrained" and h.prev not in ("?uri", "?x")
                    else []
                )
                # the trie steers the beam token by token, the set prices the
                # finished relation once
                h.boost_node = build_boost_trie(encouraged) if encouraged else None
                h.encouraged = frozenset(encouraged)
            else:
                h.node = self.node[tok]
            return h

        if h.slot == "relation":
            if h.matcher(relation_grammar).is_terminated():
                if h.text().endswith("}"):
                    h.done = True  # the type tail closed the query
                else:
                    h.prev = (h.slot_prefix + tokenizer.decode(h.slot_ids)).strip()
                    if self.encouraged and h.prev not in self.encouraged:
                        h.score -= RELATION_BOOST  # the whole price, paid once
                    h.node = ALL_ENTITIES_TRIE
                    if h.mode == "constrained":
                        h.boost_nodes = tuple(range_tries(h.prev))
                        h.ranged = bool(h.boost_nodes)
                    h.slot, h.slot_ids = "object", []
                    h.slot_prime, h.slot_prefix = None, ""
            else:
                h.boost_node = h.boost_node.get(tok) if h.boost_node else None
            return h

        # object slot
        if tok in (GL_DOT_ID, GL_RBRACE_ID) and TRIE_END in self.node:
            if self.ranged and not any(TRIE_END in bn for bn in self.boost_nodes):
                h.score -= OBJECT_BOOST  # the object ended outside the range
            if tok == GL_RBRACE_ID:
                h.done = True
            else:
                h.slot, h.slot_ids = "open", []
                h.node, h.boost_nodes = None, ()
        else:
            h.node = self.node[tok]
            if not self.slot_ids and tok == QM_ID:
                # the object is a variable: the relation's range has no claim on
                # it, so it is neither walked nor charged at the end of the slot
                h.boost_nodes, h.ranged = (), False
            else:
                h.boost_nodes = tuple(bn[tok] for bn in self.boost_nodes if tok in bn)
        return h


def whole_query_beam(start, max_new_tokens=160):
    """Token-synchronous beam search over whole queries.

    Every live hypothesis advances exactly one token per round, so they always
    share a length and the summed log-prob (renormalised over the legal tokens,
    less one flat penalty per ontologically incompatible slot) ranks them fairly
    -- there is nothing left for a length normalisation to correct. Finished
    hypotheses are held in a separate pool and never compete with growing ones
    directly: since log-probs are non-positive and the ontological penalties
    only ever subtract, a live score can only fall, so once no live hypothesis
    can still beat the best completed one the search is provably finished.
    Together that leaves the search itself with no preference for long or short
    queries; only the penalties, one per slot, can express one.

    The beam decodes as ONE batch against ONE KV cache: every live hypothesis is
    a row, the prompt goes through the model once per query, and every later step
    feeds a single token per row. Branching, pruning and finishing are then just a
    reordering of the cache's rows (reparent_cache), which is only sound because
    the search is token-synchronous -- all live rows share a length, so nothing
    has to be padded and one attention mask covers them all.
    """
    bitmask = xgr.allocate_token_bitmask(1, config.vocab_size)
    live, completed, cache = [start], [], None
    for _ in range(max_new_tokens):
        best_done = max((h.score for h in completed), default=float("-inf"))
        keep = [i for i, h in enumerate(live) if h.score > best_done]
        if not keep:
            break
        if len(keep) < len(live):  # a pruned hypothesis takes its cache row with it
            live = [live[i] for i in keep]
            cache = reparent_cache(cache, keep)
        step_ids = torch.tensor(
            [live[0].ids] if cache is None else [[h.ids[-1]] for h in live],
            device=DEVICE)
        # every row is the same length, so the mask is just "attend to all of it"
        logits, cache = decoder_step(step_ids, cache, torch.ones(
            len(live), len(live[0].ids), dtype=torch.long, device=DEVICE))
        candidates = []
        for row, h in enumerate(live):
            ranking, legal = h.legal_logits(logits[row:row + 1], bitmask)
            # renormalised over the legal set, so a forced token costs ~0 and a
            # constrained rung is not pushed into closing early to stop paying.
            # Expanded off `ranking`, scored off `legal`: the boosts pick what
            # gets tried, and advance() charges their flat price once a slot
            # comes out incompatible
            logprobs = torch.log_softmax(legal, dim=-1)[0]
            top = ranking[0].topk(BEAM_WIDTH)
            for val, tok in zip(top.values.tolist(), top.indices.tolist()):
                if val == float("-inf"):
                    continue  # fewer legal tokens than BEAM_WIDTH: skip the pad
                candidates.append((row, h.advance(tok, logprobs[tok].item())))
        if not candidates:
            break
        candidates.sort(key=lambda row_hyp: row_hyp[1].score, reverse=True)
        live, rows = [], []
        for row, h in candidates[:BEAM_WIDTH]:
            if h.done:
                completed.append(h)
            else:
                live.append(h)
                rows.append(row)  # this beam decodes on, so it keeps a cache row
        if not rows:
            break
        cache = reparent_cache(cache, rows)
    return max(completed or live, key=lambda h: h.score)


@torch.no_grad()
def generate_unconstrained(question, max_new_tokens=160):
    """The same prompt decoded with no grammar, no tries and no ontological
    boosts -- whatever the fine-tuned weights produce on their own.

    This is the baseline the constrained decoder is measured against, so it
    shares generate()'s prompt exactly and searches the same number of beams
    (BEAM_WIDTH, here over the whole query rather than per slot); only the
    constraints differ. Generation stops at EOS, which the fine-tune appends
    to every training target."""
    prompt = f"Question: {question}\nSPARQL:\n"
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=BEAM_WIDTH,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_grammar_only(question):
    """Generate a SPARQL query for a natural-language question using the
    grammar alone -- no entity tries and no ontological boosts. The opening is
    one of the five beginning templates; entity slots accept any well-formed
    IRI or variable on shape alone -- the predicate slot included, since three
    IRIs in a row is valid SPARQL -- and the query chains triples on ' .' until
    it closes with ' }'.

    The grammar-only rung of the ablation ladder: same model, same whole-query
    beam search as generate(), only the constraints differ."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="grammar", slot="query", slot_ids=[])
    ).text()


def generate_no_boosts(question):
    """Generate a SPARQL query under the hard constraints only: the beginning
    grammar, the merged entity trie on both entity slots, and the relation
    whitelist -- with none of the ontological encouragement. No domain check on
    the relation that follows an entity subject, no range check on the object.

    The rung between generate_grammar_only() and generate(): everything that
    decides WHICH strings are legal is present, everything that merely nudges
    the model towards ontologically coherent choices is gone, so the gap to
    generate() isolates the boosts."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="tries", slot="begin", slot_ids=[])
    ).text()


def generate(question):
    """Generate a SPARQL query for a natural-language question under the full
    constraint stack: the beginning grammar, the merged entity trie on both
    entity slots, the relation grammar, and the ontological boosts (relations
    whose domains the subject's types cover, objects inside the relation's
    effective range).

    Searched as one whole-query beam of BEAM_WIDTH rather than slot by slot, so
    an opening template or a subject entity can still be revised once the rest
    of the triple turns out implausible. Scoring sums log-probs renormalised
    over the legal tokens and subtracts one flat penalty per incompatible slot;
    see whole_query_beam() and Hyp for why."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="constrained", slot="begin", slot_ids=[])
    ).text()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate one SPARQL query as a smoke test.")
    ap.add_argument("--beams", type=int, default=BEAM_WIDTH,
                    help=f"beam width (default {BEAM_WIDTH}; 1 is greedy)")
    BEAM_WIDTH = ap.parse_args().beams  # module-level rebind, so the search sees it
    print(generate("What is the region of Tom Perriello ?"))
