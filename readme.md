# Summary
The idea of this is to use an ontology as a guide for type-correct decoding of semantic parsing of SPARQL queries. We use the DBPedia knowledge graph, as it has the best t-box a-box separation of knowledge graphs. We test on the LCQuad 1.0 dataset, as its questions are entirely DBPedia questions. Our approach allows users to ask questions which do not have answers in the knowledge graph, while using the ontology of the knowledge graph as a way to help understand the user's questions. This is potentially less computationally intensive than many of the entity-linking based approaches in KGQA, as we do not need to embed or search the knowledge graph for relevant entities, we only need to know how to construct a query. 

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