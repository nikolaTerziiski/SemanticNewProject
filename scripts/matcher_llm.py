from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
import concurrent.futures as cf
import json
import re
from pathlib import Path
from typing import List, Tuple

# I *might* need itertools later on; leaving it for future experiments.
import itertools  # noqa: F401

# ── third‑party deps ──────────────────────────────────────────────────────
import faiss  # type: ignore
import numpy as np
import requests
import spacy

# ── spaCy bootstrap (lazy download on first run) ─────────────────────────
try:
    _nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "tagger"])
except OSError:  # Model not found – grab it the slow way
    spacy.cli.download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "tagger"])

# Sentencizer is lightweight and good enough for sentence boundaries
if "sentencizer" not in _nlp.pipe_names:
    _nlp.add_pipe("sentencizer")

# ─────────────────────────────────────────────────────────────────────────


def _embed(texts: List[str], endpoint: str, model: str) -> np.ndarray:
    """Hit LM‑Studio’s /embeddings endpoint and return **L2‑normalised** vectors."""

    payload = {"input": texts, "model": model}
    resp = requests.post(endpoint, json=payload, timeout=60)
    resp.raise_for_status()

    vecs = np.asarray(resp.json()["data"][0]["embedding"], dtype="float32")

    # Shape protection – always (n_rows, n_dim)
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]

    # The FAISS index was trained on unit vectors
    faiss.normalize_L2(vecs)
    return vecs


def _yes_no(
    snippet: str,
    cand: Tuple[str, str, str, str],
    *,
    chat_ep: str,
    chat_model: str,
    llm_cache: dict,
) -> bool:
    """Fire a quick *Yes/No* Q to the chat model and return the interpretation.

    The LLM sometimes gets cheeky with longer answers, so we only look at the
    first char and consider any 'y'/'Y' a positive.
    """

    sf, uri, label, _ = cand
    cache_key = (snippet, uri)

    # Cheap memoisation – the same (sentence, candidate) combo repeats a lot.
    if cache_key in llm_cache:
        return llm_cache[cache_key]

    prompt = (
        "Answer with 'Yes' or 'No' only.\n\n"
        f"Does the phrase «{sf}» in the sentence «{snippet}» "
        f"refer to the ontology concept «{label}»?"
    )

    resp = requests.post(
        chat_ep,
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()

    ans = resp.json()["choices"][0]["message"]["content"].strip().lower()
    ok = ans.startswith("y")
    llm_cache[cache_key] = ok
    return ok


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
    single_word_threshold: float = 0.85,  # raised from 0.75 during tinkering
    max_workers: int = 8,
) -> List[dict]:
    """Return a list of ontology annotations for *txt_path*.

    Parameters largely mirror the original script; they’ve been kept to avoid
    accidental behaviour drift.
    """  # noqa: D401 – preferable as one-liner for now

    # ---- Load index & label metadata ------------------------------------
    index = faiss.read_index(str(idx_path))

    with open(lbl_path, encoding="utf8") as fh:
        meta = json.load(fh)

    labels: List[str] = [
        re.sub(r"[_#]", " ", uri.split("#")[-1].split("/")[-1])
        for uri in meta["uris"]
    ]

    # TODO: use *ontology_path* to pull in alt labels & definitions.
    _onto_path = Path(ontology_path)  # kept for future enrichment – unused for now

    # ---- Read document ---------------------------------------------------
    doc_text = Path(txt_path).read_text(encoding="utf8")
    doc = _nlp(doc_text)

    annotations: List[dict] = []
    seen: set[tuple[str, int, int, str]] = set()
    llm_cache: dict = {}

    # ---- Loop over sentences --------------------------------------------
    for sent in doc.sents:
        snippet = sent.text

        # Greedy surface‑form regex (copied from the OG script)
        for match in re.finditer(r"\b\w(?:[\w\- ]*\w)?\b", snippet):
            sf = match.group(0)
            start = sent.start_char + match.start()
            end = sent.start_char + match.end()

            vec = _embed([sf], embed_endpoint, embed_model)
            distances, indices = index.search(vec, top_k)

            # Index returns 2‑D arrays; we only have a single query row
            candidates = {
                meta["uris"][idx]: (idx, dist)
                for idx, dist in zip(indices[0], distances[0])
                if dist >= (
                    single_word_threshold if " " not in sf else threshold
                )
            }
            if not candidates:
                continue 

            # Ask the LLM to disambiguate the *snippet* vs each candidate
            with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
                ok_map = pool.map(
                    lambda p: _yes_no(
                        snippet,
                        p,
                        chat_ep=chat_endpoint,
                        chat_model=chat_model,
                        llm_cache=llm_cache,
                    ),
                    [
                        (sf, uri, labels[idx], snippet)
                        for uri, (idx, _) in candidates.items()
                    ],
                )

            # Събираме тези, които са окей
            for (uri, (idx, score)), ok in zip(candidates.items(), ok_map):
                if not ok:
                    continue

                dup_key = (sf.lower(), start, end, labels[idx].lower())
                if dup_key in seen:
                    continue  # Тоест, ако имаме вече по-добър, просто скипваме

                seen.add(dup_key)

                annotations.append(
                    {
                        "surface_form": sf,
                        "start": start,
                        "end": end,
                        "uri": uri,
                        "label": labels[idx],
                        "score": float(score),
                        "doc": Path(txt_path).name,
                    }
                )

    # Най-добрия annotation
    span_best: dict[tuple[str, int, int], dict] = {}
    for ann in annotations:
        k = (ann["doc"], ann["start"], ann["end"])
        if k not in span_best or ann["score"] > span_best[k]["score"]:
            span_best[k] = ann

    # Final sort: left‑to‑right, break ties by descending score
    return sorted(span_best.values(), key=lambda x: (x["start"], -x["score"]))

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        sys.exit(
            "Usage: python matcher_llm.py <doc.txt> <index.faiss> <labels.json> <ontology.rdf>"
        )

    doc_path, idx_path, lbl_path, onto_path = map(Path, sys.argv[1:5])

    result = annotate_document(
        doc_path,
        idx_path,
        lbl_path,
        ontology_path=onto_path,
    )

    # Print with Unicode intact – much nicer on a Bulgarian terminal :)
    print(json.dumps(result, ensure_ascii=False, indent=2))
