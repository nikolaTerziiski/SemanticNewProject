#!/usr/bin/env python3
"""
Строи FAISS индекс + labels.json от surface_forms.csv

Бързо embed-ване:
• реже „боклуци“ (същия регекс)
• праща чанкове по 256 низа
• 4 паралелни HTTP заявки към LM Studio

CLI пример:
    python scripts/build_index.py surface_forms.csv \
        --endpoint http://localhost:1234/v1/embeddings \
        --model mistral-embed \
        -o resources/
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
import requests

_BAD_RE = re.compile(
    r"""
    ^n?[a-f0-9]{8,}$      |
    ^[a-z0-9]{10,}s?$     |
    ^[a-z]+[0-9]{4,}$     |
    ^[bcdfghjklmnpqrstvwxz]{5,}$
    """,
    re.I | re.X,
)


def _keep(form: str) -> bool:
    return not _BAD_RE.match(form) and len(form) >= 3


def _embed_batch(texts: List[str], endpoint: str, model: str) -> np.ndarray:
    payload = {"input": texts, "model": model}
    r = requests.post(endpoint, json=payload, timeout=120)
    r.raise_for_status()
    vecs = [d["embedding"] for d in r.json()["data"]]
    return np.asarray(vecs, dtype="float32")


def _embed(
    texts: Iterable[str],
    *,
    endpoint: str,
    model: str,
    batch: int = 256,
    workers: int = 4,
) -> np.ndarray:
    texts = list(texts)
    chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        arrays = list(
            ex.map(
                _embed_batch,
                chunks,
                itertools.repeat(endpoint),
                itertools.repeat(model),
            )
        )
    return np.vstack(arrays)


def run(
    csv_path: str | Path,
    out_dir: str | Path = Path("resources"),
    *,
    mode: str = "endpoint",
    embed_endpoint: str = "http://localhost:1234/v1/embeddings",
    embed_model: str = "mistral-embed",
    clean: bool = True,
    batch: int = 256,
    workers: int = 4,
) -> tuple[Path, Path]:
    if mode != "endpoint":
        raise ValueError("Само mode='endpoint' е поддържан.")
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forms, uris = [], []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = row["form"]
            if clean and not _keep(f):
                continue
            forms.append(f)
            uris.append(row["uri"])

    vecs = _embed(
        forms,
        endpoint=embed_endpoint,
        model=embed_model,
        batch=batch,
        workers=workers,
    )
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    idx_p = out_dir / "ont_index.faiss"
    lbl_p = out_dir / "labels.json"
    faiss.write_index(index, str(idx_p))
    with lbl_p.open("w", encoding="utf-8") as fh:
        json.dump({"forms": forms, "uris": uris}, fh, ensure_ascii=False, indent=2)

    return idx_p, lbl_p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--endpoint", default="http://localhost:1234/v1/embeddings")
    ap.add_argument("--model", default="mistral-embed")
    ap.add_argument("-o", "--out_dir", type=Path, default=Path("resources"))
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()

    run(
        args.csv,
        out_dir=args.out_dir,
        embed_endpoint=args.endpoint,
        embed_model=args.model,
        clean=not args.no_clean,
    )
    print("[✔] Index & labels записани")
