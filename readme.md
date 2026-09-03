# Summary
The idea of this is to use an ontology as a guide for type-correct decoding of semantic parsing of SPARQL queries. We use the DBPedia knowledge graph, as it has the best t-box a-box separation of knowledge graphs. We test on the LCQuad 1.0 dataset, as its questions are entirely DBPedia questions. Our approach allows users to ask questions which do not have answers in the knowledge graph, while using the ontology of the knowledge graph as a way to help understand the user's questions. This is less computationally intensive than many of the entity-linking based approaches in KGQA, as we do not need to embed or search the knowledge graph for relevant entities, we only need to know how to construct a query.

# Approach
We fine-tune a BART-Large model on the LCQuad 1.0 dataset. We then add a constrained decoding approach, which uses a combination of grammars and state tracking. We begin by constraining the model to only produce one of the five beginning templates present in the LCQuad dataset. After this, we enter a loop for triple construction. If the beginning template ended with a variable (i.e ?uri or ?x) then we begin the triple construction at the relation stage, where the model is able to choose any relation from a grammar compiled of all relations in the ontology. 
One special case is queries asking whether one entity is an instantiation of a particular class (a "type query"). To deal with this case, we enumerate all classes in the ontology and treat type queries as special cases of relations, which immediately terminate generation (this is due to the fact that in the LCQuad dataset, a type query is only used at the end of the query).
If the beginning of the query does not end with a variable, however, the model is constrained to a trie of all entities in the knowledge graph, meaning it can produce only a valid entity.

After the production of either an entity or a relation, the next relation/entity to be produced is encouraged to be ontologically valid. This is "encouragement" and not "constraint" due to the sparsity of data in the ontology and KG. Many entities in the DBPedia dataset do not have types, and many of the relations do not have domains or ranges. Therefore, strict ontological constraints would mean that the majority of gold queries within the LCQuad dataset could not be produced. Instead, the ontology is consulted to find relevant relations or entities, and the logits masks of these relations or entities are boosted within the model's logits. Relations with no domain or range are defaulted to have "thing" as their assigned type, meaning they are always boosted.

The approach is best summarised as follows: creating syntactically invalid sparql is forbidden, and creating ontologically valid queries is encouraged.

After successful construction of one triple, the model decides whether to terminate the query, or to add another triple. This allows for multi-hop queries.

# Notes on entities and complexity

Classes and relations are stored using an XGrammar object. This is due to simplicity, and XGrammar simply compiles these into tries behind the scenes. There are, however, too many entities to compile into an XGrammar effectively, particularly given that producing ontologically restricted bitmasks means recompiling an entity grammar based on the previous relation. Therefore, the entities are compiled directly into tries and stored in a .pkl file. Each trie represents a class, so for example we have a trie representing all entities of type "person", a trie representing all entities of type "vehicle", etc. We only use the direct types for the trie, not the transitive types, in order to save memory. When we want to find the encouraged entities for the next entity to be output, we therefore find the union of all relevant class and subclass tries and overlay them on the overall entity trie. Again, this is because the ontology has large gaps in the data.

Trying pure greedy decoding led to some pretty bad results. So, we introduce a per-slot beam search. The actual structure of the query does not need beam search - we can do that greedily. Empirically, it was found that even with greedy beam search, 92% of the time the correct template was chosen.

# Clean repo run order

Everything needed for the constrained-decoding pipeline. (The SPARQL-endpoint
execution evaluation is a separate, optional side project and is not covered
here.)

Prerequisites already in the repo:
- `tbox_reasoner/dbpedia_2016-04.owl` -- DBpedia 2016-04 ontology
- `lcquad_data/predicates.txt` -- LC-QuAD predicate whitelist
- `lcquad_data/train-data.json` / `lcquad_data/test-data.json` -- LC-QuAD v1 splits

Prerequisites to fetch:
- The two instance-type dumps, placed in `dbpedia/`:
  [instance_types_en.ttl.bz2](https://downloads.dbpedia.org/2016-04/core-i18n/en/instance_types_en.ttl.bz2)
  and [instance_types_transitive_en.ttl.bz2](https://downloads.dbpedia.org/2016-04/core-i18n/en/instance_types_transitive_en.ttl.bz2)
- The merged Qwen checkpoint directory, placed at
   `model/qwen25-coder-1.5b-lcquad` (or configure `MODEL_PATH` in `.env`).
   To train your own, run `fine_tuning/fine_tune_qwen.ipynb` on Colab.
- A `.env` file in the project root:
  ```
  DATA_PATH=dbpedia
   MODEL_PATH=model/qwen25-coder-1.5b-lcquad
  ```
   (`MODEL_PATH` is optional; the path above is the default.)

Python deps: `pip install torch transformers xgrammar safetensors rdflib python-dotenv`

Run the pipeline in this order:

1. `python tbox_reasoner/surface_reasoning.py`
   - Reads the OWL T-box and `lcquad_data/predicates.txt`.
   - Writes `tbox_reasoner/tbox_rules.json` (already committed; rerun only if the ontology or the whitelist changes).

2. `python preprocessing/extract_entities.py`
   - Reads the two `instance_types*.ttl.bz2` dumps from `DATA_PATH`.
   - Writes `dbpedia/entities.pkl` and `dbpedia/class_entities.json`.

3. `python preprocessing/build_class_tries.py`
   - Reads `dbpedia/class_entities.json` and `tbox_reasoner/tbox_rules.json`.
   - Writes `dbpedia/class_tries.pkl`.
   - Rebuild this whenever the tokenizer/model changes; it defaults to
     `Qwen/Qwen2.5-Coder-1.5B` and can be overridden with `TRIE_MODEL_ID`.

4. `python type_constrained_generation.py`
   - Loads the merged Qwen model, t-box rules, and class tries, then generates a query for one built-in question as a smoke test.

5. `python test.py`
   - Generates queries for all 1000 LC-QuAD test questions.
   - Writes `output.json` (rewritten every 10 questions) and prints exact-match and modulo-namespace-twins accuracy.