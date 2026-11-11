from __future__ import annotations

import io
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("DEEPTRASH_MODEL_DIR", str(BASE_DIR / "models")))
DATA_DIR = Path(os.getenv("DEEPTRASH_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = DATA_DIR / "deeptrash_history.sqlite3"
METRICS_PATH = BASE_DIR / "evaluation" / "test_model_comparison.csv"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MODEL_FILES = {
    "DenseNet121": ["DenseNet121.keras", "TrashNet_densenet121_final.h5"],
    "EfficientNetV2S": ["EfficientNetV2S.keras", "TrashNet_EfficientNetV2S.keras"],
    "MobileNetV3Large": ["MobileNetV3Large.keras", "TrashNet_MobileNetV3Large.keras"],
    "ConvNeXtTiny": ["ConvNeXtTiny.keras", "TrashNet_ConvNeXtTiny.keras"],
}
MODEL_INFO = {
    "DenseNet121": "DenseNet121 · transfer learning",
    "EfficientNetV2S": "EfficientNetV2S · transfer learning",
    "MobileNetV3Large": "MobileNetV3-Large · lightweight experiment",
    "ConvNeXtTiny": "ConvNeXt-Tiny · lightweight experiment",
}
RECOMMENDATIONS = {
    "cardboard": ("Flatten clean cardboard and keep it dry. Remove food-soiled sections, tape and plastic where local collection rules require it.", "Paper/cardboard recycling stream, if accepted locally."),
    "glass": ("Empty and rinse the container. Handle broken glass carefully; do not place loose shards in a bag unless local guidance says how.", "Glass collection point or bottle bank, if available."),
    "metal": ("Empty and rinse cans or containers. Separate non-packaging metal items because collection rules may differ.", "Metal or mixed-container recycling stream, if accepted locally."),
    "paper": ("Keep paper clean and dry. Greasy, wax-coated or heavily contaminated paper may not be recyclable in your area.", "Paper recycling stream, if accepted locally."),
    "plastic": ("Empty and rinse the item. Check the resin symbol and your local collector's accepted-plastics list; a recycling symbol alone does not guarantee acceptance.", "Plastic/container collection stream, if accepted locally."),
    "trash": ("This model category represents the TrashNet 'trash' class, not a verified hazardous-waste assessment. Check the item and local guidance before disposal.", "Residual waste only when it is not accepted for reuse, recycling or a special collection stream."),
}
st.set_page_config(page_title="DeepTrash | Smart Waste AI", page_icon="♻️", layout="wide")

st.markdown("""
<style>
:root { --deep:#164b3c; --mint:#e9f5ee; }
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {background:linear-gradient(120deg,#123d32,#26775c);padding:1.45rem 1.7rem;border-radius:20px;color:white;margin-bottom:1rem;}
.hero h1 {color:white;font-size:2.15rem;margin:0;}
.hero p {color:#e1f2e8;margin:.35rem 0 0 0;}
.kicker {color:#287458;text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;font-weight:700;}
.result-card {background:#f0f8f3;border:1px solid #d6e9dd;border-radius:15px;padding:1rem 1.2rem;}
.small-note {color:#64756c;font-size:.86rem;}
div[data-testid="stMetric"] {background:#f7faf8;border:1px solid #e5eee8;padding:12px;border-radius:12px;}
</style>
<div class="hero">
  <div style="font-size:.8rem;letter-spacing:.16em;font-weight:700;color:#bfe7d0">MASTER'S PROJECT · DEEP LEARNING</div>
  <h1>♻️ DeepTrash</h1>
  <p>AI-powered waste image classification · six TrashNet categories · four model architectures</p>
</div>
""", unsafe_allow_html=True)

# ---------- Persistence ----------
def connect_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT,
        model_name TEXT NOT NULL, predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL, probabilities_json TEXT NOT NULL,
        inference_ms REAL NOT NULL
    )""")
    conn.commit()
    return conn

def save_prediction(filename, model_name, label, confidence, probabilities, inference_ms):
    conn = connect_db()
    conn.execute(
        "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), filename,
         model_name, label, float(confidence), json.dumps(probabilities),
         float(inference_ms))
    )
    conn.commit()
    conn.close()

def read_history(limit=None):
    conn = connect_db()
    sql = "SELECT created_at, filename, model_name, predicted_class, confidence, inference_ms FROM predictions ORDER BY created_at DESC"
    if limit:
        df = pd.read_sql_query(sql + " LIMIT ?", conn, params=(int(limit),))
    else:
        df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def discover_models():
    found = {}
    for name, candidates in MODEL_FILES.items():
        for candidate in candidates:
            path = MODEL_DIR / candidate
            if path.is_file():
                found[name] = path
                break
    return found

@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(model_path: str, modified_time: float):
    # Cache one network at a time to reduce memory use on free hosting.
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess(image: Image.Image, model_name: str):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    if model_name == "DenseNet121":
        arr = tf.keras.applications.densenet.preprocess_input(arr)
    # EfficientNetV2S, MobileNetV3Large and ConvNeXtTiny in the notebooks
    # were instantiated with include_preprocessing=True; pass RGB 0..255.
    return np.expand_dims(arr, axis=0)

def run_prediction(image, model_name, model_path):
    model = load_model(str(model_path), model_path.stat().st_mtime)
    batch = preprocess(image, model_name)
    started = time.perf_counter()
    output = model.predict(batch, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    output = np.asarray(output, dtype=np.float64)
    if output.shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"{model_name} returned {output.shape[-1]} outputs; expected {len(CLASS_NAMES)}.")
    # Softmax heads should already sum to one. Normalize defensively.
    output = np.clip(output, 0, None)
    total = float(output.sum())
    if total <= 0:
        raise ValueError("The model returned invalid probability scores.")
    output /= total
    idx = int(np.argmax(output))
    return CLASS_NAMES[idx], float(output[idx]), dict(zip(CLASS_NAMES, output.tolist())), elapsed_ms

def chart_probabilities(probabilities):
    chart_df = pd.DataFrame({"Waste category": list(probabilities.keys()),
                             "Model score": [v * 100 for v in probabilities.values()]})
    st.bar_chart(chart_df.set_index("Waste category"), horizontal=True, height=250)

def model_asset_help():
    st.warning("No trained model files were detected.")
    st.markdown(
        "Run the merged Kaggle notebook, download **DeepTrash_All_Models.zip**, "
        "and extract the four `.keras` files into this project's `models/` folder. "
        "The model files are not bundled here because they must come from your training run."
    )
    st.code("models/\\n  DenseNet121.keras\\n  EfficientNetV2S.keras\\n  MobileNetV3Large.keras\\n  ConvNeXtTiny.keras")

available_models = discover_models()
if not available_models:
    model_asset_help()
    st.stop()

# Class label integrity: compare bundled labels if supplied.
labels_file = MODEL_DIR / "class_names.json"
if labels_file.exists():
    try:
        saved_labels = json.loads(labels_file.read_text(encoding="utf-8"))
        if saved_labels != CLASS_NAMES:
            st.error(f"Class label order mismatch. Model labels file contains {saved_labels}, while app expects {CLASS_NAMES}.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read class_names.json: {exc}")
        st.stop()

# ---------- Navigation ----------
pages = ["Classify waste", "Compare models", "History", "Recycling guide", "Admin dashboard", "Project & methodology"]
selected_page = st.sidebar.radio("Navigation", pages)
st.sidebar.markdown("---")
st.sidebar.markdown("**Available models**")
for model_name in MODEL_FILES:
    st.sidebar.write(("🟢 " if model_name in available_models else "⚪ ") + model_name)
st.sidebar.caption(f"{len(available_models)} of 4 model files found")
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · TrashNet six-class classification")

# ---------- Classifier ----------
if selected_page == "Classify waste":
    st.markdown('<div class="kicker">Image inference</div>', unsafe_allow_html=True)
    st.title("Classify a waste item")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        chosen_model = st.selectbox("Prediction model", list(available_models.keys()))
        source = st.radio("Choose image source", ["Upload image", "Take a photo"], horizontal=True)
        if source == "Upload image":
            uploaded = st.file_uploader("Upload a waste photo", type=["jpg","jpeg","png","webp"])
        else:
            uploaded = st.camera_input("Take a photo of one waste item")
    if uploaded:
        try:
            image = Image.open(uploaded)
            with left:
                st.image(image, caption="Image submitted for classification", use_container_width=True)
                classify = st.button("Identify waste", type="primary", use_container_width=True)
            if classify:
                with st.spinner(f"Running {chosen_model}..."):
                    label, confidence, probs, elapsed = run_prediction(image, chosen_model, available_models[chosen_model])
                save_prediction(getattr(uploaded, "name", "camera_capture"), chosen_model, label, confidence, probs, elapsed)
                st.session_state["latest_prediction"] = {
                    "label":label,"confidence":confidence,"probs":probs,
                    "model":chosen_model,"elapsed":elapsed
                }
        except Exception as exc:
            st.error(f"Image could not be processed: {exc}")
    with right:
        result = st.session_state.get("latest_prediction")
        if result:
            st.markdown('<div class="kicker">Prediction result</div>', unsafe_allow_html=True)
            st.markdown(f"## {result['label'].capitalize()}")
            a,b = st.columns(2)
            a.metric("Confidence score", f"{result['confidence']*100:.1f}%")
            b.metric("Inference time", f"{result['elapsed']:.0f} ms")
            st.caption(f"Model: {result['model']}")
            chart_probabilities(result["probs"])
            if result["confidence"] < .55:
                st.warning("Low confidence: try a brighter, closer photo with one item and a simple background.")
            rec, disposal = RECOMMENDATIONS[result["label"]]
            st.subheader("Suggested handling")
            st.write(rec)
            st.info(disposal)
            st.caption("Recommendations are general guidance. Always check local waste-collection rules.")
        else:
            st.info("Your prediction will appear here after you upload or capture an image and select **Identify waste**.")

# ---------- Compare models ----------
elif selected_page == "Compare models":
    st.markdown('<div class="kicker">Research evaluation</div>', unsafe_allow_html=True)
    st.title("Model comparison dashboard")
    st.write("Run the same image through each available architecture and compare predicted classes, confidence scores and inference time.")
    source = st.radio("Image source", ["Upload image", "Take a photo"], horizontal=True, key="compare_source")
    if source == "Upload image":
        compare_file = st.file_uploader("Choose one image for all models", type=["jpg","jpeg","png","webp"], key="compare_upload")
    else:
        compare_file = st.camera_input("Capture one image for comparison", key="compare_camera")
    if compare_file:
        compare_image = Image.open(compare_file)
        st.image(compare_image, caption="Shared test image", width=320)
        if st.button("Compare available models", type="primary"):
            rows, errors = [], []
            progress = st.progress(0)
            for i, (name, path) in enumerate(available_models.items()):
                try:
                    with st.spinner(f"Evaluating {name}..."):
                        label, conf, probs, elapsed = run_prediction(compare_image, name, path)
                    rows.append({"Model":name,"Predicted class":label,"Confidence (%)":round(conf*100,2),
                                 "Inference (ms)":round(elapsed,2),
                                 **{f"{c} (%)":round(probs[c]*100,2) for c in CLASS_NAMES}})
                    # Release current model before the next architecture to limit RAM.
                    st.cache_resource.clear()
                    tf.keras.backend.clear_session()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                progress.progress((i+1)/len(available_models))
            if rows:
                st.session_state["model_comparison"] = pd.DataFrame(rows)
            if errors:
                for error in errors:
                    st.error(error)
    comp = st.session_state.get("model_comparison")
    if comp is not None:
        st.subheader("Prediction comparison")
        st.dataframe(comp[["Model","Predicted class","Confidence (%)","Inference (ms)"]], use_container_width=True, hide_index=True)
        st.subheader("Confidence by model")
        st.bar_chart(comp.set_index("Model")[["Confidence (%)"]])
        st.subheader("Class probability comparison")
        prob_cols = [f"{c} (%)" for c in CLASS_NAMES]
        st.dataframe(comp[["Model"]+prob_cols], use_container_width=True, hide_index=True)
        st.download_button("Download comparison CSV", comp.to_csv(index=False).encode("utf-8"),
                           "deeptrash_model_comparison.csv","text/csv")
        st.caption("This is a per-image comparison, not a substitute for a held-out test-set evaluation.")

# ---------- History ----------
elif selected_page == "History":
    st.markdown('<div class="kicker">Local application records</div>', unsafe_allow_html=True)
    st.title("Classification history")
    history = read_history()
    if history.empty:
        st.info("No predictions have been saved on this application yet.")
    else:
        m1,m2,m3 = st.columns(3)
        m1.metric("Total predictions", len(history))
        m2.metric("Most common class", history["predicted_class"].mode().iat[0].capitalize())
        m3.metric("Average confidence", f"{history['confidence'].mean()*100:.1f}%")
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Export history as CSV", history.to_csv(index=False).encode("utf-8"),
                           "deeptrash_classification_history.csv","text/csv")
        st.subheader("Predictions by category")
        st.bar_chart(history["predicted_class"].value_counts())
        st.caption("History is stored in the app's SQLite database. Free cloud hosts may reset local files when the app restarts or redeploys.")

# ---------- Recycling guide ----------
elif selected_page == "Recycling guide":
    st.markdown('<div class="kicker">After classification</div>', unsafe_allow_html=True)
    st.title("Recycling recommendations")
    st.write("Choose a predicted category to view general handling suggestions.")
    category = st.selectbox("Waste category", CLASS_NAMES)
    advice, stream = RECOMMENDATIONS[category]
    st.subheader(category.capitalize())
    st.write(advice)
    st.success(stream)
    st.warning("The classifier predicts a visual dataset category. It does not detect chemical, biological, battery, sharp-object or other hazardous properties. Follow your local authority's disposal guidance.")

# ---------- Admin ----------
elif selected_page == "Admin dashboard":
    st.markdown('<div class="kicker">Project monitoring</div>', unsafe_allow_html=True)
    st.title("Admin dashboard")
    st.write("Basic usage analytics and model asset status for the research prototype.")
    password = os.getenv("DEEPTRASH_ADMIN_PASSWORD", "")
    if not password:
        st.warning("Admin dashboard is locked. Set DEEPTRASH_ADMIN_PASSWORD in Streamlit Cloud → App settings → Secrets (or as a local environment variable) to enable it.")
        authorized = False
    else:
        entered = st.text_input("Admin password", type="password")
        authorized = entered == password
        if entered and not authorized:
            st.error("Incorrect password.")
    if authorized:
        history = read_history()
        total = len(history)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total classifications", total)
        c2.metric("Models available", f"{len(available_models)}/4")
        c3.metric("Categories observed", history["predicted_class"].nunique() if total else 0)
        c4.metric("Mean confidence", f"{history['confidence'].mean()*100:.1f}%" if total else "—")
        st.subheader("Model files")
        model_status = []
        for name in MODEL_FILES:
            path = available_models.get(name)
            model_status.append({"Model":name,"Status":"Available" if path else "Missing",
                                 "File size (MB)":round(path.stat().st_size/1024**2,1) if path else None})
        st.dataframe(pd.DataFrame(model_status), use_container_width=True, hide_index=True)
        if total:
            c_left,c_right = st.columns(2)
            with c_left:
                st.subheader("Category distribution")
                st.bar_chart(history["predicted_class"].value_counts())
            with c_right:
                st.subheader("Usage by model")
                st.bar_chart(history["model_name"].value_counts())
            st.subheader("Recent activity")
            st.dataframe(history.head(20), use_container_width=True, hide_index=True)
            st.download_button("Download all application records", history.to_csv(index=False).encode("utf-8"),
                               "deeptrash_admin_export.csv","text/csv")
        if METRICS_PATH.exists():
            st.subheader("Held-out test-set results from notebook")
            try:
                metrics = pd.read_csv(METRICS_PATH)
                st.dataframe(metrics, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Could not load evaluation summary: {exc}")
        else:
            st.caption("Notebook test metrics will appear here when evaluation/test_model_comparison.csv is added.")

# ---------- Methodology ----------
elif selected_page == "Project & methodology":
    st.markdown('<div class="kicker">Academic documentation</div>', unsafe_allow_html=True)
    st.title("Project overview and methodology")
    st.markdown("""
**Project objective.** Build an image-based waste classification prototype that predicts one of six TrashNet classes: cardboard, glass, metal, paper, plastic or trash.

**Model architectures.** The merged training notebook contains DenseNet121, EfficientNetV2S, MobileNetV3-Large and ConvNeXt-Tiny. This website loads the trained `.keras` files; it does not train models in the browser.

**Image pipeline.** Input images are EXIF-corrected, converted to RGB, resized to 224 × 224 pixels and passed through architecture-appropriate preprocessing. The class-index order follows the notebook's sorted class-folder names.

**Evaluation.** The model comparison page compares predictions on one user-provided image. For academic performance claims, use the notebook's held-out test set and report accuracy, macro precision, macro recall, macro-F1, confusion matrices, model size and latency.

**Limitations.** TrashNet is a constrained image dataset. Real-world performance may differ under new lighting, backgrounds, camera devices, mixed waste, occlusion and local packaging variations. Confidence is a model score, not a calibrated guarantee. The app's disposal advice is generic and must be checked against local rules.
""")
    st.subheader("Model files detected")
    st.write({name: str(path.name) for name,path in available_models.items()})
    st.subheader("Suggested thesis evaluation")
    st.markdown("""
1. Keep a held-out test set separate from training and validation.
2. Report per-class precision, recall and F1-score as well as macro-F1.
3. Compare all four architectures using the same test images and class order.
4. Measure inference latency on the intended deployment hardware.
5. Conduct a separate real-world photo test and document failure cases.
6. Explain privacy, data retention and the limitations of disposal recommendations.
""")

st.markdown("---")
st.caption("DeepTrash · Master's course research prototype · Predictions are decision support, not authoritative waste-management instructions.")
