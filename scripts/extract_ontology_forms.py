#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import rdflib
from rdflib.namespace import RDFS, SKOS, OWL         
from rdflib import RDF                            

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_SPLITTER_RE = re.compile(r"[_\-]+")

_BAD_RE = re.compile(
    r"""
    ^n?[a-f0-9]{8,}$    
    |^https?://
    """,
    re.X | re.I,
)


def _yield_forms(label: str) -> Iterable[str]:
    txt = label.strip()
    if _BAD_RE.match(txt):
        return
    yield txt
    yield _CAMEL_RE.sub(" ", txt)
    yield _SPLITTER_RE.sub(" ", txt)


# ──────────────────────────────────────────────────────────────────────────
def extract(graph_path: Path) -> Iterable[tuple[str, str]]:
    g = rdflib.Graph()
    g.parse(graph_path)

    allowed_subjects: set[rdflib.term.Node] = set(g.subjects(RDF.type, OWL.Class))

    properties_to_check = (RDFS.label, SKOS.prefLabel, SKOS.altLabel)

    for prop in properties_to_check:
        for s, _, l_obj in g.triples((None, prop, None)):
            if s not in allowed_subjects:                 
                continue
            for f in _yield_forms(str(l_obj)):
                yield f, str(s)

    for s in g.subjects():
        if s not in allowed_subjects:                    
            continue
        local = str(s).split("#")[-1].split("/")[-1]
        for f in _yield_forms(local):
            yield f, str(s)


def run(rdf_path: str | Path, out_csv: str | Path | None = None) -> Path:
    rdf_path = Path(rdf_path)
    if out_csv is None:
        out_csv = rdf_path.with_suffix(".surface_forms.csv")
    out_csv = Path(out_csv)

    with out_csv.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["form", "uri"])
        for sf, uri in sorted(extract(rdf_path)):
            w.writerow([sf, uri])

    return out_csv

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("rdf", type=Path, help="Път до онтологичния файл (RDF/OWL/TTL).")
    ap.add_argument("out_csv", type=Path, help="Къде да се запише surface_forms.csv")
    args = ap.parse_args()

    with args.out_csv.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["form", "uri"])
        for sf, uri in sorted(extract(args.rdf)):
            w.writerow([sf, uri])

    print(f"Surface-forms записани в {args.out_csv}")
