# app.py
import json
import pathlib
import tempfile

import streamlit as st
from scripts.build_index import run as build_index
from scripts.extract_ontology_forms import run as extract_forms
# This now expects annotate_document to accept 'single_word_threshold'
from scripts.matcher_llm import annotate_document


# ───────── Streamlit Page Configuration ─────────
st.set_page_config(page_title="LLM Ontology Annotator", layout="wide")
st.title("🕸️ Онтологичен анотатор с LLM")
st.info("Качете онтология и текстов файл, за да започнете. Системата използва локален LLM чрез LM Studio за обработка.")

# ───────── 1. File Uploads ─────────
col1, col2 = st.columns(2)
with col1:
    onto_file = st.file_uploader("📄 Качи онтология (.rdf / .owl / .ttl)",
                                 type=["rdf", "owl", "ttl"])
with col2:
    text_file = st.file_uploader("📝 Качи текст (.txt / .md)",
                                 type=["txt", "md"])

# ───────── 2. Parameters UI ─────────
with st.expander("⚙️ Разширени параметри", expanded=False):
    top_k = st.slider(
        "Top-k кандидати (FAISS)", 1, 30, 10,
        help="Броят на най-близките кандидати от онтологията, които да се подадат на LLM за дисамбигуация."
    )
    # Expose both thresholds for fine-tuning
    default_thresh = st.slider(
        "Праг на сходство (за фрази)", 0.0, 1.0, 0.55, 0.05,
        help="Минималното косинусово сходство за термини с повече от една дума."
    )
    single_word_thresh = st.slider(
        "Праг на сходство (за единични думи)", 0.0, 1.0, 0.75, 0.05,
        help="По-строг праг за единични думи, за да се филтрират общи термини като 'вино' или 'регион'."
    )

run_btn = st.button("🚀 Анотирай", disabled=not (onto_file and text_file), type="primary")

# ───────── Helper function for session state ─────────
def ss_get(key, default):
    """Safely get a value from the session state."""
    return st.session_state.setdefault(key, default)

# ───────── 3. Main Pipeline Execution ─────────
if run_btn:
    # Clear previous results
    if "matches" in st.session_state:
        del st.session_state["matches"]

    with st.spinner("Работи… може да отнеме до 1-2 минути."):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmp = pathlib.Path(tmpdir_str)
            ont_path = tmp / onto_file.name
            txt_path = tmp / text_file.name
            ont_path.write_bytes(onto_file.read())
            txt_path.write_bytes(text_file.read())

            # FIX: Read the text content here, while the file is guaranteed to exist.
            text_content = txt_path.read_text(encoding="utf-8")

            status_area = st.empty()

            status_area.write("Стъпка 1/3: Извличане на повърхностни форми от онтологията...")
            csv_path = tmp / "surface_forms.csv"
            extract_forms(ont_path, csv_path)

            status_area.write("Стъпка 2/3: Създаване на FAISS индекс от формите...")
            idx_path, lbl_path = build_index(
                csv_path, out_dir=tmp,
                embed_endpoint="http://localhost:1234/v1/embeddings",
                embed_model="mistral-embed", # NOTE: Change if your model name is different
            )

            status_area.write("Стъпка 3/3: Анотиране на текста с LLM дисамбигуация...")
            matches = annotate_document(
                txt_path, idx_path, lbl_path,
                ontology_path=ont_path,
                embed_endpoint="http://localhost:1234/v1/embeddings",
                chat_endpoint="http://localhost:1234/v1/chat/completions",
                chat_model="mistral", # NOTE: Change if your model name is different
                top_k=top_k,
                threshold=default_thresh,
                single_word_threshold=single_word_thresh
            )
            status_area.empty()

    # Store results in session state for interactive use
    st.session_state["text_content"] = text_content
    st.session_state["matches"] = matches
    # Create a dictionary to hold the state of checkboxes for the GOLD standard
    st.session_state["gold"] = {
        (m["surface_form"], m["start"], m["end"], m["uri"]): True
        for m in matches
    }
    st.rerun()
    
# ───────── 4. Display Results and Interaction ─────────
if "matches" in st.session_state:
    matches = st.session_state["matches"]
    st.success(f"✅ Намерени {len(matches)} съвпадения")
    
    # Display results in a dataframe
    st.dataframe(matches, use_container_width=True)

    st.subheader("✏️ Ръчна проверка (✔ приеми / ❌ отхвърли)")
    # Create checkboxes for user validation
    for i, m in enumerate(matches):
        key = (m["surface_form"], m["start"], m["end"], m["uri"])
        st.session_state["gold"][key] = st.checkbox(
            f'**{m["surface_form"]}** @ ({m["start"]}:{m["end"]}) → *{m["label"]}* (Score: {m["score"]:.2f})',
            value=ss_get("gold", {}).get(key, True),
            key=f'gold_{i}'
        )

    # --- Download Buttons ---
    st.subheader("⬇️ Изтегляне на резултати")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # AUTO.json (all matches found by the system)
    json_auto = json.dumps(matches, ensure_ascii=False, indent=2)
    dl_col1.download_button("AUTO.json", json_auto,
                            f"matches_auto.json", "application/json")

    # CSV
    if matches:
        csv_header = ",".join(matches[0].keys())
        csv_rows = [",".join(map(str, m.values())) for m in matches]
        csv_lines = "\n".join([csv_header] + csv_rows)
        dl_col2.download_button("matches.csv", csv_lines,
                                f"matches.csv", "text/csv")

    # GOLD.json (only user-accepted matches)
    # Create a mapping from key to full match object for robust lookup
    match_map = { (m["surface_form"], m["start"], m["end"], m["uri"]): m for m in matches }
    gold_matches = [
        match_map[m_key] for m_key, ok in st.session_state.get("gold", {}).items() if ok
    ]
    json_gold = json.dumps(gold_matches, ensure_ascii=False, indent=2)
    dl_col3.download_button("GOLD.json", json_gold,
                            f"matches_gold.json", "application/json")