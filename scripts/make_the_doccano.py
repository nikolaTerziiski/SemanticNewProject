# scripts/make_tasks_doccano.py
import json, pathlib

CORPUS  = pathlib.Path("corpus")
OUTPUT  = pathlib.Path("output")
OUTFILE = pathlib.Path("resources/tasks_doccano.jsonl")

with OUTFILE.open("w", encoding="utf-8") as fh:
    for pred_file in OUTPUT.glob("*.json"):
        data = json.load(pred_file.open(encoding="utf-8"))
        if not data:                 # празен файл
            continue
        doc_name = data[0]["doc"]    # напр. "Barolo.txt"
        text = (CORPUS / doc_name).read_text(encoding="utf-8", errors="ignore")

        # Doccano иска [start, end, label]
        labels = [[m["start"], m["end"], m["surface"]] for m in data]

        record = {
            "text":   text,
            "meta":   {"filename": doc_name},
            "label": labels
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅  wrote {OUTFILE}  ({sum(1 for _ in open(OUTFILE, encoding='utf-8'))} docs)")
