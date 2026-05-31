import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import io
import time
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionText OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00ff88 0%, #00c9ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #888;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

.stat-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.stat-number {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}

.stat-label {
    color: #666;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.result-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #00ff88;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #e0e0e0;
    line-height: 1.8;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-green { background: #00ff8822; color: #00ff88; border: 1px solid #00ff8844; }
.badge-yellow { background: #ffcc0022; color: #ffcc00; border: 1px solid #ffcc0044; }
.badge-red { background: #ff444422; color: #ff4444; border: 1px solid #ff444444; }

.step-header {
    font-family: 'Space Mono', monospace;
    color: #00ff88;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
}

div[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #1e1e1e;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00c9ff);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.stSlider > div { color: #aaa; }
.stCheckbox > label { color: #aaa; font-family: 'DM Sans', sans-serif; }
.stSelectbox label { color: #aaa !important; }
</style>
""", unsafe_allow_html=True)


# ─── Image Preprocessing ────────────────────────────────────────────────────
def preprocess_image(img_array, method="adaptive", denoise=True, deskew=False):
    """Apply preprocessing pipeline to improve OCR accuracy."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if method == "adaptive":
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        _, processed = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "simple":
        _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        processed = gray

    if deskew:
        coords = np.column_stack(np.where(processed > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = processed.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed = cv2.warpAffine(
                processed, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    return processed


# ─── OCR Extraction ─────────────────────────────────────────────────────────
def extract_text(processed_img, lang="eng", psm=6):
    """Run Tesseract OCR and return text + word-level confidence data."""
    config = f"--oem 3 --psm {psm}"

    # Full text
    text = pytesseract.image_to_string(processed_img, lang=lang, config=config)

    # Word-level data with confidence
    data = pytesseract.image_to_data(
        processed_img, lang=lang, config=config,
        output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = int(data["conf"][i])
        if word and conf > 0:
            words.append(word)
            confidences.append(conf)

    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    df = pd.DataFrame({
        "Word": words,
        "Confidence (%)": confidences
    })

    return text.strip(), avg_conf, df


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="step-header">⚙ Configuration</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Preprocessing Method**")
    thresh_method = st.selectbox(
        "", ["adaptive", "otsu", "simple", "none"],
        label_visibility="collapsed"
    )

    st.markdown("**Tesseract PSM Mode**")
    psm_options = {
        "Auto (PSM 6)": 6,
        "Single line (PSM 7)": 7,
        "Single word (PSM 8)": 8,
        "Sparse text (PSM 11)": 11,
        "Full page (PSM 3)": 3,
    }
    psm_label = st.selectbox("", list(psm_options.keys()), label_visibility="collapsed")
    psm_val = psm_options[psm_label]

    st.markdown("**Language**")
    lang_options = {"English": "eng", "Hindi (if installed)": "hin"}
    lang_label = st.selectbox("", list(lang_options.keys()), label_visibility="collapsed")
    lang_val = lang_options[lang_label]

    st.markdown("---")
    st.markdown("**Options**")
    denoise = st.checkbox("Denoise image", value=True)
    deskew = st.checkbox("Auto-deskew (fix tilt)", value=False)
    show_preprocessed = st.checkbox("Show preprocessed image", value=True)

    st.markdown("---")
    st.markdown(
        '<p style="color:#444; font-size:0.75rem; font-family:Space Mono,monospace;">'
        'VisionText OCR v1.0<br>Built with Tesseract + OpenCV</p>',
        unsafe_allow_html=True
    )


# ─── Main UI ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">VisionText OCR</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Optical Character Recognition · Powered by Tesseract + OpenCV</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your image here",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    help="Supports PNG, JPG, BMP, TIFF, WebP"
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="step-header">01 · Original Image</p>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

    # ── Preprocess
    with st.spinner("Preprocessing..."):
        processed = preprocess_image(img_array, method=thresh_method, denoise=denoise, deskew=deskew)

    with col2:
        if show_preprocessed:
            st.markdown('<p class="step-header">02 · Preprocessed</p>', unsafe_allow_html=True)
            st.image(processed, use_container_width=True, clamp=True)
        else:
            st.markdown('<p class="step-header">02 · Ready to Extract</p>', unsafe_allow_html=True)
            st.info("Preprocessing applied. Toggle 'Show preprocessed image' in sidebar to preview.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run OCR
    if st.button("🔍 Extract Text"):
        with st.spinner("Running OCR engine..."):
            t0 = time.time()
            extracted_text, avg_conf, word_df = extract_text(processed, lang=lang_val, psm=psm_val)
            elapsed = round(time.time() - t0, 2)

        # ── Stats Row
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{avg_conf}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(word_df)}</div>
                <div class="stat-label">Words Found</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(extracted_text)}</div>
                <div class="stat-label">Characters</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{elapsed}s</div>
                <div class="stat-label">Process Time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence Badge
        if avg_conf >= 80:
            badge = '<span class="badge badge-green">High Accuracy</span>'
        elif avg_conf >= 50:
            badge = '<span class="badge badge-yellow">Medium Accuracy</span>'
        else:
            badge = '<span class="badge badge-red">Low Accuracy — Try different settings</span>'

        st.markdown(f'<p class="step-header">03 · Extracted Text &nbsp; {badge}</p>', unsafe_allow_html=True)

        if extracted_text:
            st.markdown(f'<div class="result-box">{extracted_text}</div>', unsafe_allow_html=True)
        else:
            st.warning("No text detected. Try adjusting the preprocessing method or PSM mode in the sidebar.")

        # ── Exports
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="step-header">04 · Export Results</p>', unsafe_allow_html=True)

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                "⬇ Download TXT",
                data=extracted_text,
                file_name="ocr_result.txt",
                mime="text/plain",
                use_container_width=True
            )

        with dl2:
            csv_data = word_df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                data=csv_data,
                file_name="ocr_words.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl3:
            json_data = word_df.to_json(orient="records", indent=2)
            st.download_button(
                "⬇ Download JSON",
                data=json_data,
                file_name="ocr_words.json",
                mime="application/json",
                use_container_width=True
            )

        # ── Word Confidence Table
        if not word_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="step-header">05 · Word Confidence Breakdown</p>', unsafe_allow_html=True)

            def color_conf(val):
                if val >= 80:
                    return "color: #00ff88"
                elif val >= 50:
                    return "color: #ffcc00"
                return "color: #ff4444"

            styled_df = word_df.style.map(color_conf, subset=["Confidence (%)"])
            st.dataframe(styled_df, use_container_width=True, height=300)

else:
    # ── Empty state
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; border: 1px dashed #2a2a2a; border-radius: 16px;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🖼</div>
        <p style="color: #555; font-family: Space Mono, monospace; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em;">
            Upload an image to begin extraction
        </p>
        <p style="color: #333; font-size: 0.8rem; margin-top: 0.5rem;">
            Supports PNG · JPG · BMP · TIFF · WebP
        </p>
    </div>
    """, unsafe_allow_html=True)
