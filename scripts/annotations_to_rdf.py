#!/usr/bin/env python3
"""
annotations_to_rdf.py  predictions.json  http://localhost:7200/repositories/WineRepo

Конвертира flat JSON анотации ➜ Turtle (Open Annotation Core)
и ги POST-ва към GraphDB (или извежда на stdout, ако URL не е даден).

Формат Turtle, един примерен запис:
_:b0  a oa:Annotation ;
      oa:hasBody  <http://…#Wine> ;
      oa:hasTarget [
          a oa:SpecificResource ;
          oa:hasSource "Barolo.txt" ;
          oa:hasSelector [
               a oa:TextPositionSelector ;
               oa:start 81 ;
               oa:end   85
          ]
      ] .
"""

from __future__ import annotations

import json
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Iterable

import requests


def load_flat_json(fp: Path) -> Iterable[dict]:
    """Генератор на анотации от flat JSON файл."""
    data = json.loads(fp.read_text(encoding="utf-8"))
    for ann in data:
        yield ann


def ann_to_turtle(ann: dict) -> str:
    """Една анотация ➜ Turtle блок."""
    bnode = f"_:b{uuid.uuid4().hex[:8]}"
    return textwrap.dedent(
        f"""
        {bnode}  a <http://www.w3.org/ns/oa#Annotation> ;
              <http://www.w3.org/ns/oa#hasBody>   <{ann['uri']}> ;
              <http://www.w3.org/ns/oa#hasTarget> [
                    a <http://www.w3.org/ns/oa#SpecificResource> ;
                    <http://www.w3.org/ns/oa#hasSource> "{ann['doc']}" ;
                    <http://www.w3.org/ns/oa#hasSelector> [
                         a <http://www.w3.org/ns/oa#TextPositionSelector> ;
                         <http://www.w3.org/ns/oa#start> {ann['start']} ;
                         <http://www.w3.org/ns/oa#end>   {ann['end']}
                    ]
              ] .
        """
    ).strip()


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: annotations_to_rdf.py predictions.json [GRAPHDB_REPO_URL]")

    pred_fp = Path(sys.argv[1])
    gdb_url = sys.argv[2] if len(sys.argv) == 3 else None

    turtle_chunks = [ann_to_turtle(a) for a in load_flat_json(pred_fp)]
    ttl_doc = "@prefix oa: <http://www.w3.org/ns/oa#> .\n\n" + "\n\n".join(turtle_chunks)

    if gdb_url:
        print(f"POST-ing {len(turtle_chunks)} annotations to {gdb_url} …")
        r = requests.post(f"{gdb_url}/statements",
                          data=ttl_doc.encode("utf-8"),
                          headers={"Content-Type": "text/turtle"})
        print(f"GraphDB response: {r.status_code}  {r.text[:200]}")
    else:
        print(ttl_doc)


if __name__ == "__main__":
    main()
