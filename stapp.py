
import re
import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from transformers import pipeline
from ultralytics import YOLO
from deepface import DeepFace
import pytesseract

# ---- Page config must be BEFORE any other st.* call ----
st.set_page_config(page_title="CV Final Project", layout="wide")

# ---- Custom blue style ----
def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1d4ed8 0, #020617 45%, #020617 100%);
            color: #e5e7eb;
        }
        .stTabs [role="tablist"] button {
            background-color: rgba(15, 23, 42, 0.5);
            color: #e5e7eb;
            border-radius: 999px;
            padding: 0.35rem 1.1rem;
            margin-right: 0.35rem;
            border: 1px solid rgba(59, 130, 246, 0.6);
        }
        .stTabs [role="tablist"] button[data-selected="true"] {
            background-color: #2563eb;
            color: #f9fafb;
            border-color: #60a5fa;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid #60a5fa;
            background: linear-gradient(90deg, #2563eb, #1d4ed8);
            color: #f9fafb;
            padding: 0.35rem 1.2rem;
        }
        .stButton > button:hover {
            border-color: #93c5fd;
            filter: brightness(1.05);
        }
        .stFileUploader label {
            color: #e5e7eb !important;
        }
        .blue-card {
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            border-radius: 1rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(59, 130, 246, 0.6);
        }
        .blue-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #e5e7eb;
            margin-bottom: 0.25rem;
        }
        .blue-subtitle {
            font-size: 0.9rem;
            color: #9ca3af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def zeroshot(classifier, classes, img):
    """Zero-shot image classification."""
    scores = classifier(
        img,
        candidate_labels=classes,
        hypothesis_template="This is {}.",
    )
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores

def run_segmentation(model, img):
    """YOLOv11 segmentation on a PIL image. Returns segmented PIL image."""
    img_np = np.array(img)
    results = model(img_np)[0]
    seg_bgr = results.plot()  # BGR array with masks
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(seg_rgb)

def extract_ocr_text(img, lang="eng"):
    """Run Tesseract OCR on a PIL image (English only)."""
    
    # Convert to grayscale
    gray = img.convert("L")

    # Upscale if image is small (improves OCR)
    w, h = gray.size
    if max(w, h) < 900:
        gray = gray.resize((w * 2, h * 2))


    text = pytesseract.image_to_string(gray, lang=lang)
    return text

def summarize_text(text, max_sentences=2):
    text = text.strip()
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if s]

    return " ".join(sentences[:max_sentences])

def detect_emotions(img):
    """
    Safe DeepFace emotion detection wrapper.
    Returns None if model cannot be loaded (weights missing / no internet).
    """
    try:
        result = DeepFace.analyze(
            img_path=np.array(img),
            actions=["emotion"],
            enforce_detection=False,
        )
    except Exception as e:
        print("Emotion detection error:", e)
        return None

    emotions = []
    if isinstance(result, list):
        for r in result:
            emotions.append(r.get("dominant_emotion", "unknown"))
    else:
        emotions.append(result.get("dominant_emotion", "unknown"))

    return emotions

def recognize_faces(img, db_path, model_name):
    """
    DeepFace face recognition against DB_PATH.
    Returns list of recognized names, only if distance is small enough.
    """
    try:
        results = DeepFace.find(
            img_path=np.array(img),
            db_path=db_path,
            model_name=model_name,
            distance_metric="cosine",
            threshold=1.0,
            enforce_detection=False,
        )
    except ValueError:
        return []

    DISTANCE_THRESHOLD = 0.80  

    found = set()
    dfs = results if isinstance(results, list) else [results]

    for df in dfs:
        if df is None or df.empty:
            continue

        df_sorted = df.sort_values("distance", ascending=True)
        best = df_sorted.iloc[0]
        best_identity = best["identity"]
        best_distance = float(best["distance"])

        # Debug print into terminal
        print("Best match:", best_identity, "distance:", best_distance)

        if best_distance > DISTANCE_THRESHOLD:
            # not confident enough ⇒ treat as unknown
            return []

        name = os.path.basename(best_identity).rsplit(".", 1)[0]
        found.add(name)

    return list(found)

# ---- Add CSS before drawing UI ----
add_custom_css()

# ---- Models & config (spinner) ----
with st.spinner("Initializing models, this may take a bit..."):
    # Zero-shot classifier (Task 1)
    MODEL_ZERO_NAME = "openai/clip-vit-base-patch16"
    CLASSIFIER_ZERO = pipeline(
        "zero-shot-image-classification",
        model=MODEL_ZERO_NAME,
    )

    CLASSES = [
        "a photo of nature",
        "a photo of a cat",
        "a photo of a party",
        "a photo of food",
        "a city street",
        "a building interior",
        "a document with text",
        "a person's photo",
        "a landscape with water",
    ]

    # DeepFace models & DB (Task 3)
    DEEPFACE_MODEL_NAME = "VGG-Face"
    BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
    DB_PATH = os.path.join(BASE_DIR, "app", "data", "db")
    os.makedirs(DB_PATH, exist_ok=True)

    # YOLOv11 segmentation model (Task 2)
    SEG_MODEL_NAME = "yolo11n-seg.pt"
    SEG_MODEL = YOLO(SEG_MODEL_NAME)

# ---- Top section ----
st.title("💙 Computer Vision Final Project")
st.caption(
    "Zero-shot classification • Face recognition with emotions • YOLOv11 segmentation • OCR + summarization"
)

tab1, tab2, tab3 = st.tabs(
    [
        "1. Classification & Faces",
        "2. YOLOv11 Segmentation",
        "3. OCR + Text Summarization",
    ]
)

# ---------- TAB 1 ----------
with tab1:
    st.markdown(
        """
        <div class="blue-card">
            <div class="blue-title">Image classification + friends recognition + emotions</div>
            <div class="blue-subtitle">
                Upload a photo and the app will classify it, try to find your friends in the database,
                and estimate the dominant emotions for detected faces.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="tab1_uploader",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        if any(file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            with st.spinner("Processing image..."):
                bytes_data = uploaded_file.read()
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")

                scores = zeroshot(
                    classifier=CLASSIFIER_ZERO,
                    classes=CLASSES,
                    img=img,
                )

            st.subheader("🎯 Zero-shot classification")
            st.image(img, caption="Uploaded image", width="stretch")

            df_scores = pd.DataFrame(scores).set_index("label")
            st.bar_chart(df_scores["score"])
            st.caption("Raw scores:")
            st.write(scores)

            st.caption(f"DB_PATH used by app: {DB_PATH}")
            st.caption(f"Files in DB: {os.listdir(DB_PATH)}")

            st.subheader("👥 Face recognition (DeepFace)")
            recognized = recognize_faces(
                img=img,
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
            )
            if recognized:
                st.success("Recognized: " + ", ".join(recognized))
            else:
                st.info("No known faces found.")

            st.subheader("😊 Emotion detection")
            emotions = detect_emotions(img)

            if emotions is None:
                st.warning(
                    "Emotion detection is currently unavailable because the DeepFace "
                    "emotion model could not be downloaded in this environment."
                )
            elif emotions:
                st.write("Detected emotions:", ", ".join(emotions))
            else:
                st.write("No emotions detected.")

# --- Here starts the second tab ;)
with tab2:
    st.markdown(
        """
        <div class="blue-card">
            <div class="blue-title">YOLOv11 segmentation</div>
            <div class="blue-subtitle">
                The model highlights objects with masks and bounding boxes using YOLOv11 segmentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_seg = st.file_uploader(
        "Upload an image (JPEG/PNG) for segmentation",
        type=["jpg", "jpeg", "png"],
        key="tab2_uploader",
    )

    if uploaded_seg is not None:
        file_name = uploaded_seg.name
        if any(file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            with st.spinner("Running YOLOv11 segmentation..."):
                bytes_data = uploaded_seg.read()
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                seg_img = run_segmentation(SEG_MODEL, img)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original image")
                st.image(img, width="stretch")
            with col2:
                st.subheader("Segmented image")
                st.image(seg_img, caption="YOLOv11 segmentation result", width="stretch")
        else:
            st.error(
                "File read error: please upload .jpg, .jpeg or .png",
                icon="⚠️",
            )

#  --- Here starts the third tab
with tab3:
    st.markdown(
        """
        <div class="blue-card">
            <div class="blue-title">OCR with Tesseract + Text summarization</div>
            <div class="blue-subtitle">
                Extract text from an image using Tesseract, then summarize it with a transformer model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_ocr = st.file_uploader(
        "Upload an image (JPEG/PNG) with text",
        type=["jpg", "jpeg", "png"],
        key="tab3_uploader",
    )

    if uploaded_ocr is not None:
        file_name = uploaded_ocr.name
        if any(file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            with st.spinner("Running OCR..."):
                bytes_data = uploaded_ocr.read()
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                text = extract_ocr_text(img)

            st.subheader("🖼️ Image")
            st.image(img, use_container_width=True)

            st.subheader("📝 Extracted text")
            text = st.text_area(
                "OCR result (you can edit before summarizing):",
                value=text,
                height=200,
            )

            if st.button("Summarize text"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(text)
                st.subheader("📌 Summary")
                st.write(summary)
        else:
            st.error(
                "File read error: please upload .jpg, .jpeg or .png",
                icon="⚠️",
            )
