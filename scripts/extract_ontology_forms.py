#!/usr/bin/env python3
"""
Извлича surface-форми от RDF/OWL онтология → CSV: form,uri

Включва разширено генериране на форми:
- Различни регистри (lowercase, Titlecase)
- Разцепване на camelCase и snake_case
- Форми за множествено число
- Лематизация (основна форма на думата)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import inflect
import rdflib
import spacy
from rdflib.namespace import RDFS, SKOS

# --- Инициализация на необходимите библиотеки ---
p = inflect.engine()
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Изтегляне на spaCy модела 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# --- Регулярни изрази за нормализация и филтриране ---
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_SNAKE_RE = re.compile(r"[_-]+")
_BAD_RE = re.compile(
    r"""
    ^n?[a-f0-9]{8,}$      |  # hex / uuid
    ^[a-z0-9]{10,}s?$     |  # hex + 's'
    ^[a-z]+[0-9]{4,}$     |  # дума + много цифри
    ^[bcdfghjklmnpqrstvwxz]{5,}$  # без гласни
    """,
    re.I | re.X,
)

def _keep(form: str) -> bool:
    """Филтрира „боклучави“ или неинформативни форми."""
    return not _BAD_RE.match(form) and len(form) >= 3


def _yield_forms(label: str) -> set[str]:
    """
    Генерира множество от потенциални повърхностни форми от един етикет.
    """
    forms: set[str] = set()
    base = label.strip()
    if not base:
        return forms

    # 1. Основни вариации на регистъра
    forms.update({base, base.lower(), base.title()})

    # 2. Разцепване на camelCase и snake_case
    camel = _CAMEL_RE.sub(" ", base).strip()
    snake = _SNAKE_RE.sub(" ", base).strip()
    forms.update({camel, camel.lower(), snake, snake.lower()})

    # 3. Добавяне на форми за множествено число
    plural = p.plural(base)
    if plural and plural != base:
        forms.update({plural, plural.lower()})
        
    # 4. (НОВО) Добавяне на лема (основна форма на думата)
    # Помага за съвпадение на "wines" в текста с концепт "wine".
    doc = nlp(base)
    lemma = " ".join([token.lemma_ for token in doc])
    if lemma and lemma != base.lower():
        forms.add(lemma)

    return {f for f in forms if len(f) > 1}


def extract(graph_path: Path):
    """Генерира всички потенциални двойки (форма, URI) от онтологията."""
    g = rdflib.Graph()
    g.parse(graph_path)

    # (НОВО) Включваме skos:prefLabel, rdfs:label и skos:altLabel
    properties_to_check = [RDFS.label, SKOS.altLabel, SKOS.prefLabel]

    for prop in properties_to_check:
        for s, _, l_obj in g.triples((None, prop, None)):
            for f in _yield_forms(str(l_obj)):
                yield f, str(s)
            
    # Извличане на форми от самия URI
    for s in g.subjects():
        if isinstance(s, rdflib.URIRef):
            local_name = s.split("#")[-1].split("/")[-1]
            for f in _yield_forms(local_name):
                yield f, str(s)


def run(ontology: str | Path, outfile: str | Path) -> Path:
    """Основна функция за извличане на форми и запис в CSV файл."""
    ontology = Path(ontology)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    dedup = {}
    for form, uri in extract(ontology):
        key = form.lower()
        # Запазваме първия срещнат регистър за дадена форма
        if key not in dedup and _keep(form):
            dedup[key] = (form, uri)

    with outfile.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["form", "uri"])
        # Записваме сортирани за консистентност
        wr.writerows(sorted(list(dedup.values())))

    return outfile


# ――――― CLI Interface ―――――
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Извлича повърхностни форми от онтология.")
    ap.add_argument("ontology", type=Path, help="Път до онтологичния файл (RDF/OWL/TTL).")
    ap.add_argument("outfile", type=Path, help="Път до изходния CSV файл.")
    args = ap.parse_args()
    
    run(args.ontology, args.outfile)
    print(f"[✔] Повърхностните форми са извлечени в → {args.outfile}")