import json
from pathlib import Path
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import tensorflow as tf

from recommendations import get_advice

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "agrovision_mobilenetv2.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
DB_PATH = BASE_DIR / "agrovision.db"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #f4f8f2; }
    [data-testid="stSidebar"] { background: #123d2b; }
    [data-testid="stSidebar"] * { color: #f4fff6 !important; }
    .hero {
        background: linear-gradient(110deg,#123d2b,#26734d);
        padding: 26px 30px; border-radius: 18px; color: white;
        margin-bottom: 20px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
    .hero p { color: #e1f2e5; margin: 8px 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #e1ebdf;
        border-radius: 14px; padding: 16px 18px;
    }
    .small-muted { color: #66776b; font-size: .9rem; }
    div.stButton > button {
        background: #26734d; color: white; border: 0; border-radius: 9px;
    }
    div.stButton > button:hover { background: #185638; color: white; }
</style>
""", unsafe_allow_html=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)
        conn.commit()

def save_prediction(filename, label, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions(created_at, filename, predicted_class, confidence) VALUES(?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), filename, label, float(confidence))
        )
        conn.commit()

def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at, filename, predicted_class, confidence FROM predictions ORDER BY id DESC",
            conn
        )

@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def predict(image, model, class_names):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    probabilities = model.predict(arr, verbose=0)[0]
    order = np.argsort(probabilities)[::-1][:3]
    return [(class_names[int(i)], float(probabilities[int(i)])) for i in order]

def display_label(label):
    return label.replace("___", " — ").replace("_", " ")

init_db()
model, class_names = load_model_and_classes()

with st.sidebar:
    st.markdown("## 🌱 AgroVision AI")
    st.caption("Smart farming assistant")
    page = st.radio("Navigation", ["Overview", "Disease Scanner", "Prediction History", "About"])
    st.divider()
    st.caption("AI model status")
    if model is None:
        st.warning("Model not trained yet")
        st.caption("Run: python train_model.py")
    else:
        st.success("Model ready")

st.markdown("""
<div class="hero">
  <h1>🌿 AgroVision AI</h1>
  <p>Intelligent crop health monitoring and plant disease detection</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    history = load_history()
    total = len(history)
    avg_conf = history["confidence"].mean() * 100 if total else 0
    unique = history["predicted_class"].nunique() if total else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scans", total)
    c2.metric("Average confidence", f"{avg_conf:.1f}%")
    c3.metric("Classes detected", unique)

    st.subheader("Welcome to your smart farming dashboard")
    st.write("Upload a clear image of a crop leaf to screen for diseases supported by your trained model.")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### What you can do")
        st.markdown(
            "- **Scan a leaf:** upload JPG, JPEG or PNG images.\n"
            "- **Review results:** view the predicted class and model confidence.\n"
            "- **Get guidance:** read general care and prevention suggestions.\n"
            "- **Track activity:** review previous scans and confidence trends."
        )
    with right:
        st.markdown("### Getting started")
        if model is None:
            st.info("1. Download and extract the PlantVillage dataset.\n\n2. Run `python train_model.py`.\n\n3. Refresh this page.")
        else:
            st.success("Your disease detection model is ready. Open Disease Scanner to begin.")
    if total:
        st.subheader("Recent scans")
        st.dataframe(history.head(8), use_container_width=True, hide_index=True)

elif page == "Disease Scanner":
    st.subheader("🔬 Plant disease scanner")
    if model is None:
        st.error("No trained model found. Follow the training instructions in README.md, then refresh the app.")
        st.stop()

    uploaded = st.file_uploader("Upload a crop leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_result = st.columns([1, 1.2])
        with col_img:
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
        with col_result:
            if st.button("Analyze leaf image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    results = predict(image, model, class_names)
                    label, confidence = results[0]
                    save_prediction(uploaded.name, label, confidence)
                    st.session_state["last_result"] = (label, confidence, results)
            if "last_result" in st.session_state:
                label, confidence, results = st.session_state["last_result"]
                st.markdown("### Prediction")
                st.success(display_label(label))
                st.metric("Model confidence", f"{confidence * 100:.1f}%")
                st.progress(min(max(confidence, 0.0), 1.0))
                if confidence < 0.60:
                    st.warning("Low-confidence result. Capture another clear image and seek expert confirmation.")
                st.markdown("### Other possible classes")
                for other_label, other_conf in results[1:]:
                    st.write(f"{display_label(other_label)} — {other_conf * 100:.1f}%")
                st.markdown("### Crop-care guidance")
                advice = get_advice(label)
                st.write(advice["summary"])
                for item in advice["steps"]:
                    st.markdown(f"- {item}")
                st.caption(advice["disclaimer"])
    else:
        st.info("Tip: photograph one leaf in good natural light. Keep the leaf in focus and avoid heavy shadows.")

elif page == "Prediction History":
    st.subheader("📊 Prediction history")
    history = load_history()
    if history.empty:
        st.info("No predictions yet. Scan a leaf to build your history.")
    else:
        history["confidence_percent"] = (history["confidence"] * 100).round(1)
        st.dataframe(history, use_container_width=True, hide_index=True)
        counts = history["predicted_class"].value_counts().reset_index()
        counts.columns = ["Disease class", "Scans"]
        fig = px.bar(counts, x="Disease class", y="Scans", title="Scans by predicted class",
                     color_discrete_sequence=["#26734d"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            history.to_csv(index=False).encode("utf-8"),
            file_name="agrovision_prediction_history.csv",
            mime="text/csv"
        )

elif page == "About":
    st.subheader("About AgroVision AI")
    st.write(
        "AgroVision AI is an educational decision-support prototype. It uses a trained "
        "image-classification model to estimate which known leaf category is most similar "
        "to an uploaded image."
    )
    st.markdown("**Current model:** MobileNetV2 transfer learning")
    st.markdown("**Supported categories:** the classes present in the dataset used for training.")
    st.warning(
        "This tool is not a laboratory diagnosis or a substitute for an agricultural "
        "extension officer. Model confidence is not the probability that a diagnosis is correct. "
        "Do not apply pesticides or other treatments solely from this prediction."
    )
