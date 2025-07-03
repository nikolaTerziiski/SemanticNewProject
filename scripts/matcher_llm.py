#!/usr/bin/env python3
"""
LLM-based disambiguation matcher with caching to optimize performance.

Key optimizations include:
1.  **LLM Caching**: Avoids re-asking the LLM the same question.
2.  **Word Boundary Matching**: Prevents partial word matches.
3.  **Sentence-Aware Context**: Provides full sentences to the LLM.
4.  **Dynamic Similarity Threshold**: Uses a stricter cutoff for single-word terms.
5.  **Conservative "Yes/No" Prompt**: Instructs the LLM to be cautious.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse, urlunparse

import faiss
import numpy as np
import requests
import spacy
from scripts.utils import get_comment

# --- Initialize spaCy for sentence detection ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe('sentencizer')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe('sentencizer')
    

# ────────────────── Helpers & Configuration ──────────────────
_STOP_RE = re.compile(r"^(full|red|white|light|medium|dry|sweet|bold|rich|acidic)$", re.I)

def _canon(u: str) -> str:
    p = urlparse(u)
    return urlunparse((p.scheme.lower(), p.netloc.lower(),
                       p.path.rstrip("/"), "", "", p.fragment.lower()))


def _embed(texts: List[str], ep: str, model: str) -> np.ndarray:
    r = requests.post(ep, json={"input": texts, "model": model}, timeout=60)
    r.raise_for_status()
    return np.asarray([d["embedding"] for d in r.json()["data"]],
                      dtype="float32")


def _informative(sf: str) -> bool:
    return (" " in sf) or (len(sf) > 4 and not _STOP_RE.match(sf))


def _yes_no(
    snippet: str,
    cand: Tuple[str, str, str, str],
    *, 
    chat_ep: str, 
    chat_model: str,
    llm_cache: Dict  # (OPTIMIZATION) Pass the cache dictionary
) -> bool:
    """
    Asks the LLM a Yes/No question, using a cache to avoid redundant calls.
    """
    # (OPTIMIZATION) Use the snippet and candidate URI as a unique key.
    cache_key = (snippet, cand[1]) 
    if cache_key in llm_cache:
        return llm_cache[cache_key]

    sf, _, lbl, comm = cand
    prompt = textwrap.dedent(f"""
        Answer only with "Yes" or "No".
        Text snippet: "{snippet}"
        Does the term "{sf}" in this context refer to the concept: {lbl} ({comm})?
    """).strip()

    payload = {
        "model": chat_model, "temperature": 0.0, "max_tokens": 8, "stream": False,
        "messages": [
            {"role": "system", "content": "You are a precise ontology expert. If you are unsure, you must answer 'No'."},
            {"role": "user", "content": prompt},
        ],
    }
    
    try:
        r = requests.post(chat_ep, json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip().lower()
        result = (content == "yes")
    except (requests.RequestException, KeyError, IndexError):
        result = False

    # (OPTIMIZATION) Store the new result in the cache before returning.
    # The cache is a standard dict, which is thread-safe enough for this use case.
    llm_cache[cache_key] = result
    return result

# ────────────────── Main Annotation Logic ──────────────────

def annotate_document(
    txt_path: str | Path,
    idx_path: str | Path,
    lbl_path: str | Path,
    *,
    ontology_path: str | Path,
    embed_endpoint: str = "http://localhost:1234/v1/embeddings",
    embed_model: str = "mistral-embed",
    chat_endpoint: str = "http://localhost:1234/v1/chat/completions",
    chat_model: str = "mistral",
    top_k: int = 10,
    threshold: float = 0.55,
    single_word_threshold: float = 0.75,
    max_workers: int = 8,
) -> List[dict]:
    
    # (OPTIMIZATION) Initialize the cache for this annotation run.
    llm_cache = {}

    index = faiss.read_index(str(idx_path))
    with Path(lbl_path).open(encoding="utf-8") as fh:
        lab = json.load(fh)
    forms, uris = lab["forms"], lab["uris"]
    labels = [re.sub(r"[_#]", " ", u.split("#")[-1].split("/")[-1]) or u for u in uris]

    text = Path(txt_path).read_text(encoding="utf-8")
    mentions = _find_mentions(text, forms)
    text_doc = nlp(text)

    out, seen = [], set()
    for sf, s, e in mentions:
        if not _informative(sf):
            continue
            
        snippet = _snippet(text_doc, s, e)
        q = _embed([snippet], embed_endpoint, embed_model)
        faiss.normalize_L2(q)
        D, I = index.search(q, top_k)

        is_single_word = " " not in sf
        current_threshold = single_word_threshold if is_single_word else threshold
        
        smap = {i: float(sc) for i, sc in zip(I[0], D[0]) if sc >= current_threshold}

        best = {}
        for idx, sc in smap.items():
            uri = uris[idx]
            if uri not in best or sc > best[uri][1]:
                best[uri] = (idx, sc)
        if not best:
            continue

        def _mk(idx):
            return (sf, uris[idx], labels[idx],
                    get_comment(ontology_path, uris[idx]) or "")

        cand_idx = [v[0] for v in best.values()]
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            # (OPTIMIZATION) Pass the cache to each worker thread.
            ok_map = ex.map(
                lambda p: _yes_no(snippet, p, chat_ep=chat_endpoint, chat_model=chat_model, llm_cache=llm_cache),
                [_mk(i) for i in cand_idx]
            )

        for (uri, (idx, sc)), ok in zip(best.items(), ok_map):
            if not ok:
                continue
            
            key = (sf.lower(), s, e, _canon(uri))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "surface_form": sf, "start": s, "end": e,
                "uri": uri, "label": labels[idx], "score": float(sc), "doc": Path(txt_path).name,
            })
            
    return sorted(out, key=lambda x: x['start'])

# ───────── Updated Utils: Mentions / Snippets ─────────
def _find_mentions(txt: str, forms: List[str]) -> list[tuple[str, int, int]]:
    mentions = []
    unique_lower_forms = set(f.lower() for f in forms if f)

    for form in unique_lower_forms:
        try:
            for match in re.finditer(rf"\b{re.escape(form)}\b", txt, re.IGNORECASE):
                mentions.append((match.group(0), match.start(), match.end()))
        except re.error:
            continue
    return sorted(list(set(mentions)), key=lambda x: x[1])

def _snippet(doc: spacy.tokens.Doc, s: int, e: int) -> str:
    span = doc.char_span(s, e, alignment_mode="expand")
    if span and span.sent:
        return span.sent.text.replace("\n", " ").strip()
    return doc.text[max(0, s - 70):min(len(doc.text), e + 70)].replace("\n", " ")

# ───────── CLI Quick Test ─────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        sys.exit("Usage: python scripts/matcher_llm.py <doc.txt> <index.faiss> <labels.json> <ontology.ttl>")
    doc, idx, lbl, onto = map(Path, sys.argv[1:5])
    res = annotate_document(doc, idx, lbl, ontology_path=onto)
    print(json.dumps(res, ensure_ascii=False, indent=2))