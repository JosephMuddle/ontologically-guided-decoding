# Summary
The idea of this is to use an ontology as a guide for type-correct decoding of semantic parsing of SPARQL queries. We use the DBPedia knowledge graph, as it has the best t-box a-box separation of knowledge graphs. We test on the LCQuad 1.0 dataset, as its questions are entirely DBPedia questions. Our approach allows users to ask questions which do not have answers in the knowledge graph, while using the ontology of the knowledge graph as a way to help understand the user's questions. This is less computationally intensive than many of the entity-linking based approaches in KGQA, as we do not need to embed or search the knowledge graph for relevant entities, we only need to know how to construct a query.

# Approach
We fine-tune a BART-Large model on the LCQuad 1.0 dataset. We then add a constrained decoding approach, which uses a combination of grammars and state tracking. We begin by constraining the model to only produce one of the five beginning templates present in the LCQuad dataset. After this, we enter a loop for triple construction. If the beginning template ended with a variable (i.e ?uri or ?x) then we begin the triple construction at the relation stage, where the model is able to choose any relation from a grammar compiled of all relations in the ontology. 
One special case is queries asking whether one entity is an instantiation of a particular class (a "type query"). To deal with this case, we enumerate all classes in the ontology and treat type queries as special cases of relations, which immediately terminate generation (this is due to the fact that in the LCQuad dataset, a type query is only used at the end of the query).
If the beginning of the query does not end with a variable, however, the model is constrained to a trie of all entities in the knowledge graph, meaning it can produce only a valid entity.

After the production of either an entity or a relation, the next relation/entity to be produced is encouraged to be ontologically valid. This is "encouragement" and not "constraint" due to the sparsity of data in the ontology and KG. Many entities in the DBPedia dataset do not have types, and many of the relations do not have domains or ranges. Therefore, strict ontological constraints would mean that the majority of gold queries within the LCQuad dataset could not be produced. Instead, the ontology is consulted to find relevant relations or entities, and the logit masks of these relations or entities are boosted within the model's logit. The approach is best summarised as follows: creating syntactically invalid sparql is forbidden, and creating ontologically valid queries is encouraged.

After successful construction of one triple, the model decides whether to terminate the query, or to add another triple. This allows for multi-hop queries.

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
