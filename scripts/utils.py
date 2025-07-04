#!/usr/bin/env python3
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import rdflib


@lru_cache(maxsize=32)
def get_graph(path: Path) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(path)
    return g


@lru_cache(maxsize=2048)
def get_comment(onto_path: Path, uri: str) -> str | None:
    g = get_graph(onto_path)
    q = f"""
    SELECT ?txt WHERE {{
      <{uri}> (rdfs:comment|skos:definition|dc:description) ?txt
    }} LIMIT 1
    """
    for (txt,) in g.query(q):
        return str(txt)
    return None
