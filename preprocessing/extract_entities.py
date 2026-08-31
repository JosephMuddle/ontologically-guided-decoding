import time
import pickle
import bz2
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
# assigned to entities with no direct type record, so untyped entities are
# bucketed (and trie'd) under owl:Thing instead of being dropped
OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"


def keep_type(iri):
    # Keep all DBpedia type info (including owl:Thing); only strip out
    # schema.org and wikidata. The dbo "Wikidata:Qxxxx" classes are wikidata
    # concepts mirrored into the dbo namespace, so they're excluded too.
    if "schema.org" in iri:
        return False
    if "wikidata.org" in iri:
        return False
    if "Wikidata:" in iri:
        return False
    return True


def parse_triple(line):
    # line format: <subject> <predicate> <object> .
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    if line.endswith(" ."):
        line = line[:-2]
    # object IRIs have no internal spaces, so a 2-split is safe for type triples
    parts = line.split(" ", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2].strip()


def main():
    direct_file = os.path.join(DATA_PATH, "instance_types_en.ttl.bz2")
    transitive_file = os.path.join(DATA_PATH, "instance_types_transitive_en.ttl.bz2")

    # {entity_iri: {"type": <direct type>, "transitive_types": [<dbpedia types>]}}
    entities = dict()

    # 1) direct (most-specific) type from instance_types
    with bz2.open(direct_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="instance_types"):
            parsed = parse_triple(line)
            if parsed is None:
                continue
            head_iri, relation_iri, type_iri = parsed
            if relation_iri != RDF_TYPE:
                continue
            info = entities.setdefault(
                head_iri, {"type": None, "transitive_types": []}
            )
            info["type"] = type_iri

    # 2) transitive (ancestor) types from instance_types_transitive, dbpedia only
    with bz2.open(transitive_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="instance_types_transitive"):
            parsed = parse_triple(line)
            if parsed is None:
                continue
            head_iri, relation_iri, type_iri = parsed
            if relation_iri != RDF_TYPE:
                continue
            if not keep_type(type_iri):
                continue
            info = entities.setdefault(
                head_iri, {"type": None, "transitive_types": []}
            )
            info["transitive_types"].append(type_iri)

    # entities with no direct type record (they appear only in the transitive
    # file) get owl:Thing assigned before export: this keeps them in
    # class_entities below -- and hence in the owl:Thing class trie and the
    # merged all-entities trie -- instead of dropping them entirely
    for info in entities.values():
        if info["type"] is None:
            info["type"] = OWL_THING

    out_path = os.path.join(DATA_PATH, "entities.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(entities, f)

    # inverted index: each class -> all entities whose (final) direct type is
    # that class; consumed to build per-class token tries without a full scan
    class_entities = dict()
    for head_iri, info in entities.items():
        if info["type"] is not None:
            class_entities.setdefault(info["type"], []).append(head_iri)

    index_path = os.path.join(DATA_PATH, "class_entities.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(class_entities, f)

    print(f"Wrote {len(entities):,} entities -> {out_path}")
    print(f"Wrote {len(class_entities):,} classes -> {index_path}")


if __name__ == "__main__":
    main()
