# T-box rule generation
This constrains an LLM's output to be type-valid SPARQL according to the OWL ontology defined
This requires creating a system that can suggest next token based on types and subtypes
One problem is subsumption, subclassof, subpropertyof, but also disjoint class and same as are important as well. Symmetry/inverseOf need to be checked. domain and range too.

So far, class subsumption has been built where the output is a dictionary with keys being every parent value and values being every descendant (either child or grandchild etc etc recursively)

Now, equivalent must be done for equivalent class, but probably a different data structure. I chose to use a list of lists where each sublist is a group of classes which are equivalent to one another using the equivalentOf

Now property subsumption is done using subpropertyof

disjointclass is done also, with a big dictionary that contains a lot of duplication and includes all subclass expansion

# DBPEDIA preprocessing
All we need is the entities and their types, the information about domain and range is in the owl file. Need a dictionary with the keys being a list of all types and their values being all instances within that type

# Restriction of output
Using outlines "choice" function or guidance "select" function to make the model select from a list. However, some calculations need to be done before then involving class subsumption.
Sparkle does this using a python control flow, rather than a grammar. We could use XGrammar or Outlines.


# Transitive types and transitive properties
Transitive types are all the types which are ancestors of a given type. Any property which has a domain of type t also has a domain of all its ancestors, as type t is necessarily also all of those types. There is an equivalent "subpropertyof" for properties. 

for transitive domains, all the domains must be in the direct (or transitive) types of object t. This is implied in the case of direct domains, but when we have to infer domains (as in some cases) it's better to be more specific.


if you're choosing a relation P to connect a subject entity of type t to an object entity of type u:
P's effective domain must be a non-strict superclass of t (or of one of t's types) — the domain test you already have.
P's effective range must be a non-strict superclass of u (or of one of u's types) — the same test, run against the object's transitive-types set.

# Per query grammars
This makes the grammar context dependent kind of

``python
root          ::= select_uri | count_uri | ask

select_uri    ::= "SELECT DISTINCT ?uri WHERE { " body " }"
count_uri     ::= "SELECT DISTINCT COUNT(?uri) WHERE { " body " }"
ask           ::= "ASK WHERE { " entity " " relation " " entity " }"

body          ::= triple | triple " . " triple | triple " . " triple " . " triple
triple        ::= rel_triple | type_triple
rel_triple    ::= term " " relation " " term
type_triple   ::= bridge " rdf:type " class
term          ::= var | entity
var           ::= "?uri" | "?x"
bridge        ::= "?uri" | "?x"

{RELATIONS}   # relation ::= "<...>" | "<...>" | ...
{ENTITIES}    # entity   ::= "<...>" | "<...>" | ...
{CLASSES}     # class    ::= "<...>" | "<...>" | ...
``

we cannot just have a grammar which ends where the user chooses an entity. They must have
a further series of restrictions based on type to further restrict.

# Next Steps
We need to make two functions for relations and entities to determine the legal next tokens, based on "t box rules json"

1. Make a function called "legal next relation" which takes an entity as input, and then takes that entity's type from the entities.pkl file (in this pkl file the entity names are keys and the types are values). Then, using tbox_rules.json, find all relations in the "effective property domain map" dictionary which have the domain of the type of the entity. Then find all transitive types of the entity and find all relations for which they are the domain. Then return this list of legal relations. Variables (i.e ?x and ?uri) have the owl:thing type (use the proper DBPedia notation) and no transitive types.

1a. Make a function for "legal next entity" based on the logic that when given a relation, the 

2. There is an edge case: if ?uri or ?x begins a triple, there are two ways this triple can finish.
One of them is <relation><entity> and the other is <type><class>. The problem comes from the fact that if a triple is started with a ?uri and ?x, then <relation> could be any relation, and therefore <entity> could be any entity. This does not fit with the recursive grammar, so we need to do two things within the control flow:

2a. compile a grammar of all relations using the same alternation function as we do with entities
2b. make a separate branch such that if the last token was ?uri or ?x and the automaton is in "rel" mode, draw a relation from the relation alternated grammar


The flow becomes: extract_entities.py → class_entities.json → build_class_tries.py → class_tries.pkl → loaded at runtime. Let me write the builder:

 we test this fully by replacing the query with every query in the lcquad train data? make testing it on the training data a separate function that runs from main and prints information such as how much time it took to parse every query, average time to parse per query, percentage of queries passed, percentage of queries not passed, and then percentage brakdowns of entity rejection/rel rejection and a sample of 5 of the queries not passed (assuming that any don't pass)

 62% of queries failed because of relation rejection - they all have "property" in them which is not in the owl file and has no domain or range, so can't be type constrained. We can make a trie which is run in parallel to the type constraints for relations.


IMPORTANT: We had to add 3 predicates to the whitelist, as they appear as the wrong namespace in the lcquad dataset (ontology when it should be resource and vice versa)


Maybe we can make the Thing trie as a superset of each class trie, such that we strictly only allow tokens within the thing trie, but encourage tokens within the relevant class trie

# Notes on grammar and entity missingness
354 queries in the 4000 train data queries were rejected by our grammar - they reference entities which do not exist in the entities pkl. These are entities which exist in dbpedia but carry no rdf:type assertion in the type dumps.

# Clean repo run order

Prerequisites:
- `tbox_reasoner/dbpedia_2016-04.owl`
- `predicates.txt`
- `dbpedia/instance_types_en.ttl.bz2`
- `dbpedia/instance_types_transitive_en.ttl.bz2`
- `lcquad_data/train-data.json`
- `.env` containing `DATA_PATH=dbpedia`

Run the pipeline in this order:

1. `python tbox_reasoner/surface_reasoning.py`
   - Reads the OWL T-box and `predicates.txt`.
   - Writes `tbox_reasoner/tbox_rules.json`.

2. `python preprocessing/extract_entities.py`
   - Reads the two `instance_types*.ttl.bz2` dumps.
   - Writes `dbpedia/entities.pkl` and `dbpedia/class_entities.json`.

3. `python preprocessing/build_class_tries.py`
   - Reads `dbpedia/class_entities.json` and `tbox_reasoner/tbox_rules.json`.
   - Writes `dbpedia/class_tries.pkl`.
   - Rebuild this whenever the tokenizer/model changes; it currently uses `facebook/bart-large`.

4. `python _scratch_trie.py`
   - Loads `tbox_rules.json`, `entities.pkl`, and `class_tries.pkl`.
   - Runs the strict acceptor over every LC-QuAD train query and prints the full failure report.

# Hybrid hard/soft entity guidance

Strategy: keep entity existence as a hard constraint, but make T-box range compatibility a soft preference.

- Use `CLASS_TRIES["__ALL__"]` to create a dense vocab-sized hard mask at each entity-generation step.
  - Tokens continuing the all-entities trie receive mask value `0`.
  - Every other token receives `-inf`.
  - This prevents hallucinated entities but keeps every known entity reachable.

- For an object slot, derive the legal range classes from `tbox_rules.json` as usual.

- For each legal class trie, create a dense vocab-sized bonus mask.
  - Tokens continuing that class trie receive `+RANGE_BONUS`.
  - Every other token receives `0`.
  - Do not use `-inf` in these class masks, and do not literally sum several hard class masks: the entity only needs to belong to one legal class. Combine the class bonus masks by elementwise maximum, which represents the union of the legal class continuations.

- Add the combined masks to the model logits:
  - outside `__ALL__`: `-inf`;
  - inside `__ALL__` but not range-compatible: unchanged model logit;
  - inside `__ALL__` and range-compatible: model logit plus `RANGE_BONUS`.

- Apply the same idea at every token inside the entity IRI. The model still chooses each token autoregressively; the masks only restrict and reweight the available choices.

- Keep variables (`?uri`, `?x`) specially handled. They can remain allowed, but decide later whether they should receive the range bonus or stay neutral.

- A relation-level analogue can be added later:
  - hard mask over the predicate whitelist;
  - dense bonus mask over domain-compatible relations.

Implementation outline (dense version first, no optimisation yet):

1. At the start of an entity slot, record the current trie state for `__ALL__`.
2. At each generation step, materialise the current `__ALL__` node's outgoing tokens as a vocab-sized hard mask.
3. For each legal range class, materialise the corresponding class-trie node's outgoing tokens as a vocab-sized bonus mask.
4. Merge the class bonus masks with elementwise maximum.
5. Add the hard mask and merged bonus mask to the model logits.
6. Let the model choose the next token, then advance the trie state according to that chosen token.
7. Tune `RANGE_BONUS` empirically.

This does not solve missing entities: an entity absent from `entities.pkl` is still outside `__ALL__` and therefore impossible. It only converts domain/range type conflicts from hard rejections into score disadvantages.