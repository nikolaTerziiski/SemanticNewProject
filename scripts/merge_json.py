#!/usr/bin/env python3
"""
Exact-matcher с лемматизация.
  python scripts/matcher_exact.py corpus/Barolo.txt \
         --forms resources/surface_forms.csv
"""

import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

############################
# 1. Surface-forms helpers #
############################
def load_surface_forms(csv_path: Path) -> Dict[str, str]:
    """CSV -> {form: uri}"""
    with open(csv_path, encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        return {row["form"]: row["uri"] for row in rdr}

############################
# 2.  Matcher              #
############################
def match(text: str,
          surface_forms: Dict[str, str],
          lemmas_map: Dict[str, str]) -> List[dict]:
    """Връща [{span,start,end,uri}, …]"""
    matches = []

    # (а) оригинален текст
    for form, uri in surface_forms.items():
        for m in re.finditer(rf"\b{re.escape(form)}\b", text, flags=re.I):
            matches.append({"span": m.group(0),
                            "start": m.start(),
                            "end": m.end(),
                            "uri": uri})

    # (б) лемматизиран текст
    doc = nlp(text)
    lemmas_text = " ".join(tok.lemma_ for tok in doc)
    for lemma, uri in lemmas_map.items():
        for m in re.finditer(rf"\b{re.escape(lemma)}\b", lemmas_text, flags=re.I):
            # намираме реалните индекси в оригиналния текст
            token = next(t for t in doc if t.lemma_ == lemma)
            matches.append({"span": token.text,
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "uri": uri})

    # remove duplicates
    uniq = {(m["start"], m["end"], m["uri"]): m for m in matches}
    return sorted(uniq.values(), key=lambda x: x["start"])

############################
# 3.  CLI                  #
############################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("textfile", type=Path)
    ap.add_argument("--forms", type=Path, required=True,
                    help="surface_forms.csv, генериран от extract_ontology_forms_full.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    surface_forms = load_surface_forms(args.forms)
    lemmas_map     = {nlp(k)[0].lemma_: v for k, v in surface_forms.items()}

    text = args.textfile.read_text(encoding="utf-8")
    result = match(text, surface_forms, lemmas_map)

    out_path = args.out or args.textfile.with_suffix("_exact.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[✔] {len(result)} matches → {out_path}")

if __name__ == "__main__":
    main()
