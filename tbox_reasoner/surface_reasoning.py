"""Extract surface-level class relations from the DBpedia ontology OWL file.

All URIs are written as bracketed IRIs (``<http://...>``), matching the SPARQL
surface form used downstream.

Every property-related calculation is restricted to the LC-QuAD predicate
whitelist (``predicates.txt``) -- the only predicates that can appear in
LC-QuAD queries. Class calculations (subsumption, disjointness, equivalence)
are unrestricted.

The class subsumption dictionary maps each parent class URI to all of its
descendant child class URIs via ``rdfs:subClassOf``. Descendants include direct
children, grandchildren, and deeper subclasses.

The equivalent class groups list contains URI groups connected by
``owl:equivalentClass``.

The classes list contains every class URI in the ontology: classes declared
with ``rdf:type owl:Class``, every class participating in
``rdfs:subClassOf`` links, and same-namespace ``owl:equivalentClass`` aliases
(e.g. Location, declared solely as the equivalent of Place). Unlike the
subsumption dictionary (whose keys are parents only), leaf, root, and alias
classes are included too.

The property subsumption dictionary maps each whitelisted parent property URI
to all of its whitelisted descendant child property URIs via
``rdfs:subPropertyOf``. Expansion runs over the full graph, so paths may pass
through non-whitelisted intermediates, but only whitelisted properties appear
as keys and descendants.

The disjoint class dictionary maps each class URI to class URIs it is disjoint
with via ``owl:disjointWith``. It is expanded symmetrically and through subclass
descendants for faster lookup.

The effective property domain dictionary maps each whitelisted property URI to
every subject class URI declared for it via ``rdfs:domain``, accumulated
conjunctively across its whole ``rdfs:subPropertyOf`` ancestry (ancestors are
walked whether or not they are whitelisted -- their domains still apply): an
entity is legal for the property only if every listed domain is covered by its
direct type or transitive types. Properties with no domain anywhere up the
ancestor chain fall back to ``owl:Thing``. This covers every raw infobox
(``dbpedia.org/property/``) predicate, which the OWL T-box does not define at
all, so infobox predicates are unconstrained in both slots.

The effective property range dictionary maps each whitelisted property URI to
every object class or datatype URI declared for it via ``rdfs:range``,
accumulated conjunctively across its whole ``rdfs:subPropertyOf`` ancestry
(ancestors are walked whether or not they are whitelisted): an entity is legal
for the property's object slot only if every listed range is covered by its
direct type or transitive types. Properties with no range anywhere up the
ancestor chain fall back to ``owl:Thing``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS


# Path to the DBpedia ontology OWL file to read.
DBPEDIA_OWL_FILE = Path(__file__).with_name("dbpedia_2016-04.owl")
# Path to the LC-QuAD predicate whitelist: one bare IRI per line. Only these
# predicates can appear in LC-QuAD queries, so all property calculations are
# restricted to them.
PREDICATE_WHITELIST_FILE = Path(__file__).parent.parent / "predicates.txt"
# Where to write the extracted relations as JSON. Set to None to print to stdout.
OUTPUT_FILE = Path(__file__).with_name("tbox_rules.json")
# Fallback domain/range for properties with no declared class.
OWL_THING = f"<{OWL.Thing}>"
# Namespace of the DBpedia ontology; used to keep external equivalents
# (schema.org, wikidata, ...) out of the classes list.
DBPEDIA_ONTOLOGY_NS = "http://dbpedia.org/ontology/"


def bracketed_uri(uri: URIRef) -> str:
    """Return the URI as a bracketed IRI, e.g. ``<http://...>``."""
    return f"<{uri}>"


def load_predicate_whitelist(
    whitelist_file: str | Path = PREDICATE_WHITELIST_FILE,
) -> list[str]:
    """Load the LC-QuAD predicate whitelist as sorted bracketed IRIs.

    The file holds one bare IRI per line with trailing commas; blank lines
    and junk lines (anything not starting with ``http``) are skipped.
    """
    predicates: set[str] = set()
    for line in Path(whitelist_file).read_text(encoding="utf-8").splitlines():
        iri = line.strip().rstrip(",").strip()
        if iri.startswith("http"):
            predicates.add(f"<{iri}>")
    return sorted(predicates)


def load_ontology_graph(owl_file: str | Path = DBPEDIA_OWL_FILE) -> Graph:
    """Parse the ontology RDF/XML file into an rdflib graph."""
    graph = Graph()
    graph.parse(Path(owl_file), format="xml")
    return graph


def extract_direct_subclass_map_from_graph(
    graph: Graph,
) -> dict[str, set[str]]:
    """Return direct ``parent -> children`` class links from an rdflib graph."""
    direct_subclasses: dict[str, set[str]] = defaultdict(set)

    for child_class, parent_class in graph.subject_objects(RDFS.subClassOf):
        if not isinstance(child_class, URIRef):
            continue
        if not isinstance(parent_class, URIRef) or parent_class == child_class:
            continue
        direct_subclasses[bracketed_uri(parent_class)].add(bracketed_uri(child_class))

    return dict(direct_subclasses)


def extract_direct_subproperty_map_from_graph(
    graph: Graph,
) -> dict[str, set[str]]:
    """Return direct ``parent -> children`` property links from an rdflib graph."""
    direct_subproperties: dict[str, set[str]] = defaultdict(set)

    for child_property, parent_property in graph.subject_objects(RDFS.subPropertyOf):
        if not isinstance(child_property, URIRef):
            continue
        if (
            not isinstance(parent_property, URIRef)
            or parent_property == child_property
        ):
            continue
        direct_subproperties[bracketed_uri(parent_property)].add(bracketed_uri(child_property))

    return dict(direct_subproperties)


def expand_to_recursive_subsumptions(
    direct_descendants: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Expand direct parent-child links into ``parent -> all descendants``."""
    direct_sets = {
        parent: set(children) for parent, children in direct_descendants.items()
    }
    descendants_by_parent: dict[str, set[str]] = {}

    def descendants_for(parent_uri: str, active_path: set[str]) -> set[str]:
        if parent_uri in descendants_by_parent:
            return descendants_by_parent[parent_uri]
        if parent_uri in active_path:
            return set()

        active_path.add(parent_uri)
        descendants: set[str] = set()

        for child_uri in direct_sets.get(parent_uri, set()):
            descendants.add(child_uri)
            descendants.update(descendants_for(child_uri, active_path))

        active_path.remove(parent_uri)
        descendants.discard(parent_uri)
        descendants_by_parent[parent_uri] = descendants
        return descendants

    return {
        parent: sorted(descendants_for(parent, set()))
        for parent in sorted(direct_sets)
    }


def extract_class_subsumptions_from_graph(graph: Graph) -> dict[str, list[str]]:
    """Extract recursive class subsumptions from an rdflib graph."""
    direct_subclasses = extract_direct_subclass_map_from_graph(graph)
    return expand_to_recursive_subsumptions(direct_subclasses)


def extract_property_subsumptions_from_graph(
    graph: Graph,
    whitelist: Iterable[str],
) -> dict[str, list[str]]:
    """Extract recursive property subsumptions, restricted to the whitelist.

    Expansion runs over the full graph (paths may pass through
    non-whitelisted intermediates); only whitelisted properties appear as
    keys and descendants.
    """
    keep = set(whitelist)
    direct_subproperties = extract_direct_subproperty_map_from_graph(graph)
    subsumptions = expand_to_recursive_subsumptions(direct_subproperties)
    return {
        parent: [descendant for descendant in descendants if descendant in keep]
        for parent, descendants in subsumptions.items()
        if parent in keep
    }


def extract_direct_property_domain_map_from_graph(
    graph: Graph,
) -> dict[str, set[str]]:
    """Return direct ``property -> domain classes`` links from an rdflib graph."""
    property_domains: dict[str, set[str]] = defaultdict(set)

    for property_uri, domain_class in graph.subject_objects(RDFS.domain):
        if not isinstance(property_uri, URIRef):
            continue
        if not isinstance(domain_class, URIRef):
            continue
        property_domains[bracketed_uri(property_uri)].add(bracketed_uri(domain_class))

    return dict(property_domains)


def extract_property_domain_map_from_graph(
    graph: Graph,
    property_uris: Iterable[str],
) -> dict[str, list[str]]:
    """Extract the effective ``rdfs:domain`` classes per whitelisted property.

    Every domain declared on the property or on any of its
    ``rdfs:subPropertyOf`` ancestors applies conjunctively, so all of them are
    accumulated: an entity is legal for the property only if every listed
    domain is covered by its direct type or transitive types. Ancestors are
    walked whether or not they are whitelisted -- their domains still apply.
    Properties with no domain anywhere up the ancestor chain fall back to
    ``owl:Thing``.
    """
    direct_property_domains = extract_direct_property_domain_map_from_graph(graph)
    property_uris = set(property_uris)

    # child -> direct parent properties, for walking the subPropertyOf chain upward
    parent_properties: dict[str, set[str]] = defaultdict(set)
    for parent_uri, child_uris in extract_direct_subproperty_map_from_graph(graph).items():
        for child_uri in child_uris:
            parent_properties[child_uri].add(parent_uri)

    def effective_domain(property_uri: str) -> set[str]:
        # breadth-first walk accumulating every domain declared on the property
        # or any transitive ancestor; all apply as conjunctive constraints
        visited = {property_uri}
        queue = [property_uri]
        domains: set[str] = set()
        while queue:
            next_queue = []
            for uri in queue:
                domains.update(direct_property_domains.get(uri, ()))
                for parent_uri in parent_properties.get(uri, ()):
                    if parent_uri not in visited:
                        visited.add(parent_uri)
                        next_queue.append(parent_uri)
            queue = next_queue
        return domains

    return {
        property_uri: sorted(effective_domain(property_uri) or {OWL_THING})
        for property_uri in sorted(property_uris)
    }


def extract_direct_property_range_map_from_graph(
    graph: Graph,
) -> dict[str, set[str]]:
    """Return direct ``property -> range classes`` links from an rdflib graph."""
    property_ranges: dict[str, set[str]] = defaultdict(set)

    for property_uri, range_class in graph.subject_objects(RDFS.range):
        if not isinstance(property_uri, URIRef):
            continue
        if not isinstance(range_class, URIRef):
            continue
        property_ranges[bracketed_uri(property_uri)].add(bracketed_uri(range_class))

    return dict(property_ranges)


def extract_property_range_map_from_graph(
    graph: Graph,
    property_uris: Iterable[str],
) -> dict[str, list[str]]:
    """Extract the effective ``rdfs:range`` classes per whitelisted property.

    Every range declared on the property or on any of its
    ``rdfs:subPropertyOf`` ancestors applies conjunctively, so all of them are
    accumulated: an entity is legal for the property's object slot only if
    every listed range is covered by its direct type or transitive types.
    Ancestors are walked whether or not they are whitelisted -- their ranges
    still apply. Properties with no range anywhere up the ancestor chain fall
    back to ``owl:Thing``.
    """
    direct_property_ranges = extract_direct_property_range_map_from_graph(graph)
    property_uris = set(property_uris)

    # child -> direct parent properties, for walking the subPropertyOf chain upward
    parent_properties: dict[str, set[str]] = defaultdict(set)
    for parent_uri, child_uris in extract_direct_subproperty_map_from_graph(graph).items():
        for child_uri in child_uris:
            parent_properties[child_uri].add(parent_uri)

    def effective_range(property_uri: str) -> set[str]:
        # breadth-first walk accumulating every range declared on the property
        # or any transitive ancestor; all apply as conjunctive constraints
        visited = {property_uri}
        queue = [property_uri]
        ranges: set[str] = set()
        while queue:
            next_queue = []
            for uri in queue:
                ranges.update(direct_property_ranges.get(uri, ()))
                for parent_uri in parent_properties.get(uri, ()):
                    if parent_uri not in visited:
                        visited.add(parent_uri)
                        next_queue.append(parent_uri)
            queue = next_queue
        return ranges

    return {
        property_uri: sorted(effective_range(property_uri) or {OWL_THING})
        for property_uri in sorted(property_uris)
    }


def extract_direct_disjoint_class_map_from_graph(
    graph: Graph,
) -> dict[str, set[str]]:
    """Return direct symmetric class disjointness from an rdflib graph."""
    disjoint_classes: dict[str, set[str]] = defaultdict(set)

    for left_class, right_class in graph.subject_objects(OWL.disjointWith):
        if not isinstance(left_class, URIRef):
            continue
        if not isinstance(right_class, URIRef) or left_class == right_class:
            continue

        left_uri = bracketed_uri(left_class)
        right_uri = bracketed_uri(right_class)
        disjoint_classes[left_uri].add(right_uri)
        disjoint_classes[right_uri].add(left_uri)

    return dict(disjoint_classes)


def expand_disjoint_classes(
    direct_disjoint_classes: Mapping[str, Iterable[str]],
    class_subsumptions: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Expand class disjointness through subclass descendants."""
    disjoint_classes: dict[str, set[str]] = defaultdict(set)

    for left_class, right_classes in direct_disjoint_classes.items():
        left_descendants = set(class_subsumptions.get(left_class, []))
        left_family = {left_class, *left_descendants}

        for right_class in right_classes:
            right_descendants = set(class_subsumptions.get(right_class, []))
            right_family = {right_class, *right_descendants}

            for left_uri in left_family:
                for right_uri in right_family:
                    if left_uri == right_uri:
                        continue
                    disjoint_classes[left_uri].add(right_uri)
                    disjoint_classes[right_uri].add(left_uri)

    return {
        class_uri: sorted(disjoint_uris)
        for class_uri, disjoint_uris in sorted(disjoint_classes.items())
    }


def extract_disjoint_class_map_from_graph(
    graph: Graph,
    class_subsumptions: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, list[str]]:
    """Extract symmetric recursive class disjointness from an rdflib graph."""
    if class_subsumptions is None:
        class_subsumptions = extract_class_subsumptions_from_graph(graph)

    direct_disjoint_classes = extract_direct_disjoint_class_map_from_graph(graph)
    return expand_disjoint_classes(direct_disjoint_classes, class_subsumptions)


def extract_equivalent_class_groups_from_graph(graph: Graph) -> list[list[str]]:
    """Return URI groups connected by ``owl:equivalentClass`` triples."""
    parent_by_class: dict[str, str] = {}

    def find(class_uri: str) -> str:
        parent_by_class.setdefault(class_uri, class_uri)
        if parent_by_class[class_uri] != class_uri:
            parent_by_class[class_uri] = find(parent_by_class[class_uri])
        return parent_by_class[class_uri]

    def union(left_class: str, right_class: str) -> None:
        left_root = find(left_class)
        right_root = find(right_class)
        if left_root != right_root:
            parent_by_class[right_root] = left_root

    for left_class, right_class in graph.subject_objects(OWL.equivalentClass):
        if not isinstance(left_class, URIRef):
            continue
        if not isinstance(right_class, URIRef) or left_class == right_class:
            continue
        union(bracketed_uri(left_class), bracketed_uri(right_class))

    groups_by_root: dict[str, set[str]] = defaultdict(set)
    for class_uri in parent_by_class:
        groups_by_root[find(class_uri)].add(class_uri)

    return sorted(
        [sorted(group) for group in groups_by_root.values() if len(group) > 1],
        key=lambda group: group[0],
    )


def extract_classes_from_graph(graph: Graph) -> list[str]:
    """Return every class URI in the ontology as sorted bracketed IRIs.

    Covers classes declared with ``rdf:type owl:Class``, every class
    participating in ``rdfs:subClassOf`` links, and same-namespace
    ``owl:equivalentClass`` aliases (e.g. Location, which is declared solely
    as the equivalent of Place) -- so leaf, root, and alias classes that
    never appear in the subsumption dictionary are included too. External
    equivalents (schema.org, wikidata, ...) are not classes of this ontology
    and are excluded.
    """
    classes: set[str] = set()
    for class_uri in graph.subjects(RDF.type, OWL.Class):
        if isinstance(class_uri, URIRef):
            classes.add(bracketed_uri(class_uri))
    for child_class, parent_class in graph.subject_objects(RDFS.subClassOf):
        if isinstance(child_class, URIRef):
            classes.add(bracketed_uri(child_class))
        if isinstance(parent_class, URIRef):
            classes.add(bracketed_uri(parent_class))
    for left_class, right_class in graph.subject_objects(OWL.equivalentClass):
        for class_uri in (left_class, right_class):
            if isinstance(class_uri, URIRef) and str(class_uri).startswith(
                DBPEDIA_ONTOLOGY_NS
            ):
                classes.add(bracketed_uri(class_uri))
    return sorted(classes)


def main() -> None:
    graph = load_ontology_graph(DBPEDIA_OWL_FILE)
    whitelist = load_predicate_whitelist()
    class_subsumptions = extract_class_subsumptions_from_graph(graph)
    property_subsumptions = extract_property_subsumptions_from_graph(graph, whitelist)
    effective_property_domain_map = extract_property_domain_map_from_graph(
        graph, whitelist
    )
    effective_property_range_map = extract_property_range_map_from_graph(
        graph, whitelist
    )
    disjoint_class_map = extract_disjoint_class_map_from_graph(
        graph,
        class_subsumptions,
    )
    equivalent_class_groups = extract_equivalent_class_groups_from_graph(graph)
    classes = extract_classes_from_graph(graph)
    surface_relations = {
        "class_subsumptions": class_subsumptions,
        "classes": classes,
        "disjoint_class_map": disjoint_class_map,
        "equivalent_class_groups": equivalent_class_groups,
        "effective_property_domain_map": effective_property_domain_map,
        "effective_property_range_map": effective_property_range_map,
        "property_subsumptions": property_subsumptions,
    }

    serialized = json.dumps(surface_relations, indent=2, sort_keys=True)
    if OUTPUT_FILE is not None:
        OUTPUT_FILE.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
