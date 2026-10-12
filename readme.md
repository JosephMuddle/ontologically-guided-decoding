# Summary
We use a TBox and type declarations to encourage type-correct SPARQL query generation from large language models. At decoding time, we up-weight or down-weight token probabilities according to how likely each token is to continue a TBox-compatible query. We evaluate on DBpedia and the LC-QuAD 1.0 dataset. Our approach does not require subgraph entity retrieval to construct the SPARQL query. As a result, users can ask questions whose answer sets are empty in the graph without changing the semantics of the generated SPARQL query.

# Repo structure
- `type_constrained_generation.py` -- the decoder: grammar constraints plus ontology-guided beam search
- `test.py` -- generates queries for the LC-QuAD test split at each constraint level
- `tbox_reasoner/` -- extracts domains, ranges, subclass and disjointness rules from the DBpedia ontology
- `lcquad_data/` -- LC-QuAD v1 splits and the predicate whitelist
- `preprocessing/` -- builds the entity tries from the DBpedia instance-type dumps
- `evaluations and results/` -- beam-width sweep results; notebooks for the constraint ablation and for executing generated queries against a local Apache Jena Fuseki endpoint
- `unanswerable_questions/` -- builds empty-answer-set questions; notebook comparing our decoder with SPARKLE on them
- `fine_tuning/` -- Colab notebook that produces the fine-tuned weights
- `dbpedia/`, `model/` -- generated artifacts and weights, not included in repo

# Clean repo run order

Everything needed for the constrained-decoding pipeline. (The SPARQL-endpoint
execution evaluation is a separate, optional side project and is not covered
here.) NOTE: in order to build unanswerable questions, the fuseki DBPedia server
needs to be active. This is in the evaluations and results/dbpedia_endpoint_local.ipynb notebook

Prerequisites already in the repo:
- `tbox_reasoner/dbpedia_2016-04.owl` -- DBpedia 2016-04 ontology
- `lcquad_data/predicates.txt` -- LC-QuAD predicate whitelist
- `lcquad_data/train-data.json` / `lcquad_data/test-data.json` -- LC-QuAD v1 splits

Prerequisites to fetch:
- The two instance-type dumps, placed in `dbpedia/`:
  [instance_types_en.ttl.bz2](https://downloads.dbpedia.org/2016-04/core-i18n/en/instance_types_en.ttl.bz2)
  and [instance_types_transitive_en.ttl.bz2](https://downloads.dbpedia.org/2016-04/core-i18n/en/instance_types_transitive_en.ttl.bz2)
- The fine-tuned weights file, placed at `model/qwen_lcquad.safetensors`
  (or configure `MODEL_WEIGHTS` in `.env`). It holds only the weights; the
  config and tokenizer come from the public base model `Qwen/Qwen2.5-Coder-1.5B`.
  To train your own, run `fine_tuning/fine_tune_qwen.ipynb` on Colab.
- A `.env` file in the project root:
  ```
  DATA_PATH=dbpedia
  MODEL_WEIGHTS=model/qwen_lcquad.safetensors
  ```
   (Both are optional; the paths above are the defaults. `MODEL_ID` overrides
   the base model the weights are loaded into.)

Python deps: `pip install torch transformers xgrammar safetensors rdflib python-dotenv`

Run the pipeline in this order:

1. Fine-tune the model -- `fine_tuning/fine_tune_qwen.ipynb` on Colab.
   - Trains the base model on the LC-QuAD train split and writes
     `model/qwen_lcquad.safetensors`.
   - Skip if you already have the weights file.

2. `python tbox_reasoner/surface_reasoning.py`
   - Reads the OWL T-box and `lcquad_data/predicates.txt`.
   - Writes `tbox_reasoner/tbox_rules.json` (already committed; rerun only if the ontology or the whitelist changes).

3. `python preprocessing/extract_entities.py`
   - Reads the two `instance_types*.ttl.bz2` dumps from `DATA_PATH`.
   - Writes `dbpedia/entities.pkl` and `dbpedia/class_entities.json`.

4. `python preprocessing/build_class_tries.py`
   - Reads `dbpedia/class_entities.json` and `tbox_reasoner/tbox_rules.json`.
   - Writes `dbpedia/class_tries.pkl`.
   - Rebuild this whenever the tokenizer/model changes; it defaults to
     `Qwen/Qwen2.5-Coder-1.5B` and can be overridden with `TRIE_MODEL_ID`.

5. `python type_constrained_generation.py`
   - Loads the fine-tuned weights onto the base model, with the t-box rules and class tries, then generates a query for one built-in question as a smoke test.

6. `python test.py`
   - Generates queries for all 1000 LC-QuAD test questions.
   - Writes `output.json` (rewritten every 10 questions) and prints exact-match and modulo-namespace-twins accuracy.