"""
Type-constrained generation.

Step 1: load the merged Qwen2.5-Coder-1.5B checkpoint

Step 2: the xgrammar grammars that define legal SPARQL structure

Step 3: masked generation, the beginning template.

Step 4: the state tracker, state = {"idx", "prev"} as in parse_query. The end
of the beginning picks the first triple slot: trailing variable (?uri/?x) ->
idx = 1 (relation next), prev = that variable; trailing '<' (ent templates)
-> idx = 0 (entity next), prev = None, and the '<' is stripped from
query_so_far (re-added together with the entity itself).

Step 5: phase 2, the triples loop

Steps 6/7: generate_positive() -- boosts what the T-box licenses, charges what
falls outside. Untyped subject, uncovered domain or owl:Thing declaration
costs nothing. Penalties land on deciding tokens only: a token is penalised
when every completion through it is disjoint.
"""
import functools
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Fine-tuned weights not found at {MODEL_WEIGHTS}. Set MODEL_WEIGHTS to "
        "the .safetensors file saved by fine_tune_qwen.ipynb."
    )


config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_config(config)

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
    """One decoding step for the whole beam: next-token logits per row, cache
    grown by the tokens just fed in.

    Row i of every argument and the result is live hypothesis i. First step:
    cache None, input_ids the whole prompt; after that, one token per row."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def reparent_cache(cache, rows):
    """Re-order the cache so row i holds the keys and values of old row rows[i]."""
    idx = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(idx)  # transformers Cache objects reorder in place
        return cache
    return tuple(tuple(t.index_select(0, idx) for t in layer) for layer in cache)


# compiled against the tokenizer, so grammars can produce next-token masks,
# not just accept/reject a finished string
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


# relation slot: one whitelisted predicate, or the type tail
# ' <rdf:type> <class> }' -- the only place a class may appear, and it closes
# the query early. Literals carry leading/trailing spaces: the leading space
# glues onto the ' <' token between slots, the trailing space separates from
# the next slot
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

# grammar-only rung, used by generate_grammar_only(): ONE grammar for the
# whole query. Every position accepts any well-formed IRI or variable on
# shape alone
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
# dict-of-dicts token trie per class plus a merged all-entities trie
# (variables ?uri/?x in every trie). A ~1.5M-literal alternation cannot be
# compiled by xgrammar; a trie is walked in O(tokens)
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

# the price of ontological incompatibility, in nats: charged once on an
# incompatible relation, once on an incompatible object. The soft-constraint
# knob (0 == pure hard constraint)
RELATION_BOOST = 5.0
OBJECT_BOOST = 5.0  # same idea, for range-compatible objects (idx 2)

# the guidance rungs, as Hyp modes
POSITIVE_MODES = ("positive", "both")
NEGATIVE_MODES = ("negative", "both")
GUIDED_MODES = ("positive", "negative", "both")

# whole-query beam width; 1 reproduces greedy decoding exactly
BEAM_WIDTH = int(os.getenv("BEAM_WIDTH", "4"))

EFFECTIVE_PROPERTY_DOMAIN_MAP = TBOX_RULES["effective_property_domain_map"]
EFFECTIVE_PROPERTY_RANGE_MAP = TBOX_RULES["effective_property_range_map"]
OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"

# child class -> transitive ancestors, inverted from the parent -> descendants
# subsumption map: an entity of class C also covers every domain that is an
# ancestor of C
ANCESTORS = {}
for _parent, _descendants in TBOX_RULES["class_subsumptions"].items():
    for _child in _descendants:
        ANCESTORS.setdefault(_child, set()).add(_parent)


def entity_types(entity_text):
    """All classes of a bracketed entity IRI, by walking every class trie over
    the entity's tokens in parallel; a class matches when its trie reaches a
    terminal node exactly at the entity's end.

    Reuses the class tries as a membership index: ~412 classes x ~12 tokens of
    dict lookups per call, vs duplicating the 283 MB class_entities.json in RAM.
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
    """Whitelisted relations whose effective domains the given types all
    cover: a type covers a domain class if it is that class or a descendant of
    it; owl:Thing domains are covered by everything."""
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
    """Class tries an object of this relation may come from: each effective
    range class plus all its subclasses. The class tries are partitioned by
    most-specific direct type, so subclass entities are NOT inside the
    superclass trie and must be unioned explicitly. owl:Thing ranges are
    unconstrained: no tries, no boost."""
    tries = []
    for cls in EFFECTIVE_PROPERTY_RANGE_MAP[relation]:
        if cls == OWL_THING:
            continue
        for sub in [cls] + TBOX_RULES["class_subsumptions"].get(cls, []):
            if sub in CLASS_TRIES:
                tries.append(CLASS_TRIES[sub])
    return tries


# class -> every class disjoint with it or one of its ancestors. Disjointness
# is inherited downwards on both sides, so a class clashes with anything below
# a class its superclass is disjoint with; clashes() checks the other side's
# ancestors. The T-box ships disjoint_class_map closed and symmetric; closing
# again here keeps clashes() right if that changes
CLASH_UP = {}
for _cls in set(TBOX_RULES["classes"]) | TBOX_RULES["disjoint_class_map"].keys():
    _up = set()
    for _a in {_cls} | ANCESTORS.get(_cls, set()):
        _up.update(TBOX_RULES["disjoint_class_map"].get(_a, ()))
    if _up:
        CLASH_UP[_cls] = frozenset(_up)


def clashes(c1, c2):
    """True when no individual can be an instance of both classes."""
    up = CLASH_UP.get(c1)
    return bool(up) and not up.isdisjoint({c2} | ANCESTORS.get(c2, set()))


def disjoint_relations(types):
    """Whitelisted relations a subject of these types provably cannot head:
    some domain class clashes with one of the types. owl:Thing clashes with
    nothing; an untyped subject makes nothing disjoint -- negative mode acts
    on evidence only."""
    return [rel for rel in RELATIONS
            if any(d != OWL_THING and clashes(t, d)
                   for d in EFFECTIVE_PROPERTY_DOMAIN_MAP[rel] for t in types)]


@functools.lru_cache(maxsize=None)
def disjoint_range_split(relation):
    """(disjoint, rest): the class tries split by clash with the relation's
    effective range. Tries are partitioned by most-specific direct type, so an
    entity from a disjoint trie provably cannot be in range. Both halves come
    back: a token is penalised only when no 'rest' entity runs through it.
    Classes without entities share one variables-only trie object, so dedupe
    by identity. owl:Thing or nothing-disjoint ranges give ((), ())."""
    ranges = [c for c in EFFECTIVE_PROPERTY_RANGE_MAP[relation] if c != OWL_THING]
    disjoint, rest = {}, {}
    for cls, trie in CLASS_TRIES.items():
        if cls == "__ALL__":
            continue
        side = disjoint if any(clashes(cls, r) for r in ranges) else rest
        side[id(trie)] = trie
    if not disjoint:
        return (), ()
    return tuple(disjoint.values()), tuple(rest.values())


def deciding_penalties(disjoint_nodes, rest_nodes):
    """Tokens that commit to a disjoint completion: continued by some disjoint
    cursor and no other. A token still shared with a compatible completion
    decides nothing. Either side exhausted -> no choice left, nothing back
    (penalising all continuations alike would be a no-op anyway)."""
    if not disjoint_nodes or not rest_nodes:
        return set()
    disjoint = {t for n in disjoint_nodes for t in n if t is not None}
    return disjoint - {t for n in rest_nodes for t in n if t is not None}


# ---------------------------------------------------------------------------
# whole-query beam search
# ---------------------------------------------------------------------------

@dataclass
class Hyp:
    """One whole-query hypothesis: tokens so far, summed log-prob, and the
    constraint state that decides what may legally come next.

    Score = log-probs renormalised over the legal tokens.

    Boosts only steer which continuations get expanded (legal_logits returns
    a separate ranking tensor); the score settles once per slot in advance():
    BOOST off a relation whose domain the subject's types don't cover, BOOST
    off an object that ends outside the range. The tighter the rung, the
    more often the model is pushed onto tokens it rates poorly, so closing the
    query early becomes the cheapest exit. Renormalise instead to not have length
    bias

    Hypotheses are copied on every branch. Trie nodes are read-only shared
    dicts; xgrammar matchers are stateful, so never stored.
    """
    ids: list                 # prompt + generated tokens
    gen: list                 # generated tokens only -- the query
    score: float              # renormalised log-prob - slot penalties
    mode: str                 # 'positive' | 'negative' | 'both' (tries + whitelist
                              # + that guidance), 'tries' (no guidance) or 'grammar'
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
    # negative guidance (the negative and both rungs). Cursors come in pairs --
    # disjoint vs everything else -- because a token is penalised only when
    # nothing compatible runs through it
    disjoint_node: dict = None    # disjoint-domain relation trie cursor (relation slot)
    rest_node: dict = None        # cursor over every other relation, and rdf:type
    disjoint_rels: frozenset = frozenset()  # relations the subject's types clash with
    disjoint_nodes: tuple = ()    # class tries that clash with the range (object slot)
    rest_nodes: tuple = ()        # every other class trie
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
        """(ranking, legal): next-token logits with forbidden tokens at -inf,
        twice -- once with the ontological boosts on top, once without.

        The search expands the top of `ranking` and scores against `legal`, so
        boosts decide what gets tried without inflating the score of every
        token they touch; the flat per-slot price lands in advance(). Same
        tensor where no boost applies. Boosts go in in place: an illegal token
        is already -inf and -inf + boost stays -inf."""
        grammar = self._grammar()
        if grammar is not None:
            legal = logits.clone()
            self.matcher(grammar).fill_next_token_bitmask(bitmask)
            xgr.apply_token_bitmask_inplace(legal, bitmask.to(DEVICE))
            if self.slot == "relation":
                boosted = list(self.boost_node) if self.boost_node else []
                penalised = (deciding_penalties((self.disjoint_node,),
                                                (self.rest_node,) if self.rest_node else ())
                             if self.disjoint_node else set())
                if boosted or penalised:
                    ranking = legal.clone()
                    if boosted:
                        ranking[0, boosted] += RELATION_BOOST
                    if penalised:
                        ranking[0, list(penalised)] -= RELATION_BOOST  # -inf stays -inf
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
        # from the object's SECOND token on: the first chooses entity vs
        # variable, which the range has no opinion about; boosting one branch
        # of a two-way choice would be the full boost against the other
        if self.slot == "object" and self.slot_ids:
            boosted = {t for bn in self.boost_nodes for t in bn if t is not None}
            penalised = deciding_penalties(self.disjoint_nodes, self.rest_nodes)
            if boosted or penalised:
                ranking = legal.clone()
                if boosted:
                    ranking[0, list(boosted)] += OBJECT_BOOST  # illegal ones stay -inf
                if penalised:
                    ranking[0, list(penalised)] -= OBJECT_BOOST
                return ranking, legal
        return legal, legal

    def advance(self, tok, logprob):
        """The transition: copy of this hypothesis with tok appended, constraint
        state moved on. Slot cycle: begin, then subject/relation/object until a
        brace closes the query."""
        h = replace(self, ids=self.ids + [tok], gen=self.gen + [tok],
                    score=self.score + logprob, slot_ids=self.slot_ids + [tok])

        if h.slot == "query":
            # grammar-only: one matcher spans the whole query -- no slot
            # bookkeeping, nothing to transition between
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
                guided = h.mode in GUIDED_MODES and h.prev not in ("?uri", "?x")
                types = entity_types(h.prev) if guided else []
                encouraged = (encouraged_relations(types)
                              if guided and h.mode in POSITIVE_MODES else [])
                disjoint = (disjoint_relations(types)
                            if guided and h.mode in NEGATIVE_MODES else [])
                # the tries steer the beam token by token, the sets price the
                # finished relation once
                h.boost_node = build_boost_trie(encouraged) if encouraged else None
                h.encouraged = frozenset(encouraged)
                h.disjoint_rels = frozenset(disjoint)
                h.disjoint_node = h.rest_node = None
                if disjoint:
                    h.disjoint_node = build_boost_trie(disjoint)
                    h.rest_node = build_boost_trie(
                        [r for r in RELATIONS if r not in h.disjoint_rels] + [RDF_TYPE])
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
                    if h.prev in self.disjoint_rels:
                        h.score -= RELATION_BOOST  # negative guidance: provably disjoint
                    h.node = ALL_ENTITIES_TRIE
                    if h.mode in POSITIVE_MODES:
                        h.boost_nodes = tuple(range_tries(h.prev))
                        h.ranged = bool(h.boost_nodes)
                    if h.mode in NEGATIVE_MODES:
                        h.disjoint_nodes, h.rest_nodes = disjoint_range_split(h.prev)
                    h.slot, h.slot_ids = "object", []
                    h.slot_prime, h.slot_prefix = None, ""
            else:
                h.boost_node = h.boost_node.get(tok) if h.boost_node else None
                h.disjoint_node = h.disjoint_node.get(tok) if h.disjoint_node else None
                # the rest only matters while there is a disjoint side to contrast
                h.rest_node = (h.rest_node.get(tok)
                               if h.disjoint_node and h.rest_node else None)
            return h

        # object slot
        if tok in (GL_DOT_ID, GL_RBRACE_ID) and TRIE_END in self.node:
            if self.ranged and not any(TRIE_END in bn for bn in self.boost_nodes):
                h.score -= OBJECT_BOOST  # the object ended outside the range
            if any(TRIE_END in n for n in self.disjoint_nodes):
                h.score -= OBJECT_BOOST  # negative: its class clashes with the range
            if tok == GL_RBRACE_ID:
                h.done = True
            else:
                h.slot, h.slot_ids = "open", []
                h.node, h.boost_nodes = None, ()
                h.disjoint_nodes, h.rest_nodes = (), ()
        else:
            h.node = self.node[tok]
            if not self.slot_ids and tok == QM_ID:
                # the object is a variable: the relation's range has no claim on
                # it, so it is neither walked nor charged at the end of the slot
                h.boost_nodes, h.ranged = (), False
                h.disjoint_nodes, h.rest_nodes = (), ()
            else:
                h.boost_nodes = tuple(bn[tok] for bn in self.boost_nodes if tok in bn)
                h.disjoint_nodes = tuple(n[tok] for n in self.disjoint_nodes if tok in n)
                # the rest only matters while there is a disjoint side to contrast
                h.rest_nodes = (tuple(n[tok] for n in self.rest_nodes if tok in n)
                                if h.disjoint_nodes else ())
        return h


def whole_query_beam(start, max_new_tokens=160):
    """Token-synchronous beam search over whole queries.

    Every live hypothesis advances one token per round, so they always share a
    length and the summed log-prob (renormalised, less per-slot penalties)
    ranks them fairly -- nothing left for length normalisation to correct.
    Finished hypotheses sit in a separate pool: scores only fall, so once no
    live hypothesis can still beat the best completed one the search is
    provably done. The search itself has no preference for long or short
    queries; only the penalties can express one.

    One batch, one KV cache: each live hypothesis is a row, the prompt goes
    through the model once, later steps feed one token per row. Branching,
    pruning and finishing are just a reordering of cache rows
    (reparent_cache) -- sound only because the search is token-synchronous:
    all rows share a length, no padding, one attention mask covers all.
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
            # Expanded off `ranking`, scored off `legal`; advance() charges the
            # flat price once a slot comes out incompatible
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
    """No grammar, no tries, no boosts -- whatever the fine-tuned weights
    produce on their own.

    Baseline for the constrained decoder: same prompt, same beam count
    (BEAM_WIDTH, over the whole query); only the constraints differ. Stops at
    EOS, which the fine-tune appends to every training target."""
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
    """Grammar alone -- no entity tries, no boosts. The opening is one of the
    five beginning templates; entity slots accept any well-formed IRI or
    variable on shape alone (predicate included -- three IRIs in a row is
    valid SPARQL); triples chain on ' .' until ' }'."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="grammar", slot="query", slot_ids=[])
    ).text()


def generate_no_boosts(question):
    """Hard constraints only: beginning grammar, merged entity trie on both
    entity slots, relation whitelist -- no ontological encouragement, no
    domain check on the relation, no range check on the object. The rung
    between generate_grammar_only() and the guided rungs."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="tries", slot="begin", slot_ids=[])
    ).text()


def generate_positive(question):
    """Hard constraints plus positive ontological guidance: relations whose
    domains the subject's types cover, and objects inside the relation's
    effective range, are boosted; what falls outside is charged.

    One whole-query beam of BEAM_WIDTH rather than slot by slot, so an opening
    template or subject entity can still be revised once the rest of the
    triple turns out implausible. See whole_query_beam() and Hyp for scoring.
    """
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="positive", slot="begin", slot_ids=[])
    ).text()


def generate_negative(question):
    """Hard constraints plus negative guidance: relations whose effective
    domain is disjoint with the subject's types, and objects whose class
    clashes with the relation's effective range, are suppressed on the
    deciding tokens and charged once. Nothing is held against a choice for
    want of type evidence."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="negative", slot="begin", slot_ids=[])
    ).text()


def generate_both(question):
    """Both kinds of guidance at once: compatible continuations boosted,
    clashing ones suppressed, charges added -- an uncovered choice pays once,
    a provably disjoint one twice."""
    prompt = f"Question: {question}\nSPARQL:\n"
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return whole_query_beam(
        Hyp(ids=ids, gen=[], score=0.0, mode="both", slot="begin", slot_ids=[])
    ).text()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate one SPARQL query as a smoke test.")
    ap.add_argument("--beams", type=int, default=BEAM_WIDTH,
                    help=f"beam width (default {BEAM_WIDTH}; 1 is greedy)")
    BEAM_WIDTH = ap.parse_args().beams  # module-level rebind, so the search sees it
    print(generate_positive("What is the region of Tom Perriello ?"))
