import streamlit as st
import numpy as np
import pickle
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import load_model
from keras.utils import pad_sequences

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense · Emotion AI",
    page_icon="🧠",
    layout="centered"
)

# ─── Constants ────────────────────────────────────────────────────
MAX_LENGTH = 50
PADDING    = 'post'

EMOTION_EMOJIS = {
    'joy':      '☀️',
    'sadness':  '🌧️',
    'anger':    '🔥',
    'fear':     '🌑',
    'love':     '✦',
    'surprise': '⚡'
}

EMOTION_COLORS = {
    'joy':      '#FFB700',
    'sadness':  '#4A9EFF',
    'anger':    '#FF3D3D',
    'fear':     '#A855F7',
    'love':     '#FF6B9D',
    'surprise': '#00E5FF'
}

EMOTION_DESC = {
    'joy':      'Pure happiness radiating through your words',
    'sadness':  'A weight of sorrow lingers in the text',
    'anger':    'Intense frustration detected in the tone',
    'fear':     'Unease and dread echo through the sentence',
    'love':     'Warmth and deep affection flow here',
    'surprise': 'Shock and astonishment in every word'
}

# ─── Inject Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080B14 !important;
    color: #E8EAF0 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0D1F3C 0%, #080B14 60%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Main container ── */
.block-container {
    max-width: 760px !important;
    padding: 3rem 2rem 4rem !important;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: #4A9EFF;
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: 0.8;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 0%, #A0B4D6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #566480;
    letter-spacing: 0.02em;
    line-height: 1.7;
}

.hero-line {
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #4A9EFF, transparent);
    margin: 1.5rem auto 0;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #566480;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 2rem;
}

/* ── Model Selector ── */
.model-selector {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}

/* Override Streamlit radio */
[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0.75rem !important;
    flex-direction: row !important;
}

[data-testid="stRadio"] label {
    background: #0D1525 !important;
    border: 1px solid #1E2D47 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #8899BB !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stRadio"] label:hover {
    border-color: #4A9EFF !important;
    color: #4A9EFF !important;
}

/* ── Text Area ── */
[data-testid="stTextArea"] textarea {
    background: #0A1020 !important;
    border: 1px solid #1E2D47 !important;
    border-radius: 12px !important;
    color: #C8D4E8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    resize: none !important;
    transition: border-color 0.25s !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #4A9EFF !important;
    box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.1) !important;
    outline: none !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #2A3A56 !important;
}

[data-testid="stTextArea"] label {
    display: none !important;
}

/* ── Predict Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1A3A6E 0%, #0D2050 100%) !important;
    border: 1px solid #2A5299 !important;
    border-radius: 10px !important;
    color: #7EB3FF !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1E4A8A 0%, #0F2A6A 100%) !important;
    border-color: #4A9EFF !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(74, 158, 255, 0.2) !important;
}

/* ── Result Card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 2rem 1.6rem;
    margin: 1.5rem 0 1rem;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--emotion-color);
    opacity: 0.9;
}

.result-emotion-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
}

.result-emoji {
    font-size: 2.4rem;
    line-height: 1;
}

.result-emotion-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}

.result-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #566480;
    letter-spacing: 0.02em;
    margin-top: 0.5rem;
    line-height: 1.6;
}

.confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.2rem;
}

.confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #566480;
    text-transform: uppercase;
    white-space: nowrap;
}

.confidence-bar-bg {
    flex: 1;
    height: 4px;
    background: #1A2540;
    border-radius: 4px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

.confidence-value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    white-space: nowrap;
}

/* ── Warning ── */
[data-testid="stAlert"] {
    background: #120A00 !important;
    border: 1px solid #3D2200 !important;
    border-radius: 10px !important;
    color: #FF9F43 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #4A9EFF !important;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid #141E30;
    margin: 1.5rem 0;
}

/* ── Chart section label ── */
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: #3A4E6E;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #1E2D47;
    text-transform: uppercase;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080B14; }
::-webkit-scrollbar-thumb { background: #1E2D47; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(os.path.abspath(__file__))
    scratch_model    = load_model(os.path.join(base_path, 'lstm_scratch.h5'))
    pretrained_model = load_model(os.path.join(base_path, 'lstm_glove.h5'))
    with open(os.path.join(base_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base_path, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    return scratch_model, pretrained_model, tokenizer, le

scratch_model, pretrained_model, tokenizer, le = load_artifacts()


# ─── Text Cleaning ────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Prediction ───────────────────────────────────────────────────
def predict_emotion(text, model):
    cleaned  = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING)
    probs    = model.predict(padded, verbose=0)[0]
    pred_idx = np.argmax(probs)
    emotion  = le.inverse_transform([pred_idx])[0]
    return emotion, probs


# ─── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Deep Learning · NLP · Sequence Models</div>
    <div class="hero-title">EmoSense</div>
    <div class="hero-sub">
        An LSTM-powered engine that reads the emotional<br>
        fingerprint hidden inside your words.
    </div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ─── Model Selector ───────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Select Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "model",
    ["LSTM From Scratch", "GloVe Pretrained LSTM"],
    horizontal=True,
    label_visibility="collapsed"
)
selected_model = scratch_model if model_choice == "LSTM From Scratch" else pretrained_model


# ─── Text Input ───────────────────────────────────────────────────
st.markdown('<div class="section-label">02 · Enter Text</div>', unsafe_allow_html=True)
user_input = st.text_area(
    "text",
    placeholder="Type or paste any sentence here…   e.g. I can't believe how incredible today was",
    height=130,
    label_visibility="collapsed"
)

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

predict_clicked = st.button("⬡  Analyze Emotion", use_container_width=True)


# ─── Prediction Output ────────────────────────────────────────────
if predict_clicked:
    if user_input.strip() == "":
        st.warning("⚠  Please enter some text before analyzing.")
    else:
        with st.spinner("Reading emotional signature…"):
            emotion, probs = predict_emotion(user_input, selected_model)

        color      = EMOTION_COLORS[emotion]
        emoji      = EMOTION_EMOJIS[emotion]
        desc       = EMOTION_DESC[emotion]
        confidence = probs.max() * 100

        # ── Result Card ───────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card" style="
            background: linear-gradient(135deg, {color}10 0%, #0A1020 100%);
            border: 1px solid {color}30;
            --emotion-color: {color};
        ">
            <div class="result-emotion-row">
                <span class="result-emoji">{emoji}</span>
                <span class="result-emotion-name" style="color:{color}">{emotion.upper()}</span>
            </div>
            <div class="result-desc">{desc}</div>
            <div class="confidence-row">
                <span class="confidence-label">Confidence</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence:.1f}%; background:{color}"></div>
                </div>
                <span class="confidence-value" style="color:{color}">{confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Chart ─────────────────────────────────────
        st.markdown('<div class="chart-title">Emotion Probability Distribution</div>',
                    unsafe_allow_html=True)

        emotions     = le.classes_
        bar_colors   = [EMOTION_COLORS[e] for e in emotions]
        alphas       = [1.0 if e == emotion else 0.3 for e in emotions]

        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor('#080B14')
        ax.set_facecolor('#080B14')

        y_pos = range(len(emotions))
        bars  = ax.barh(list(y_pos), probs * 100,
                        color=bar_colors,
                        alpha=1.0,
                        height=0.55,
                        edgecolor='none')

        # Dim non-predicted bars
        for bar, e in zip(bars, emotions):
            if e != emotion:
                bar.set_alpha(0.2)

        # Value labels
        for bar, val, e in zip(bars, probs * 100, emotions):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%',
                    va='center', ha='left',
                    color=EMOTION_COLORS[e] if e == emotion else '#2A3A56',
                    fontsize=9,
                    fontfamily='monospace',
                    fontweight='bold' if e == emotion else 'normal')

        # Emoji + label on y axis
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [f"{EMOTION_EMOJIS[e]}  {e}" for e in emotions],
            fontsize=9.5,
            fontfamily='monospace',
            color='#8899BB'
        )

        ax.set_xlim(0, 115)
        ax.set_xlabel('')
        ax.tick_params(axis='x', colors='#1E2D47', labelsize=7)
        ax.tick_params(axis='y', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#141E30')
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(labelcolor='#2A3A56')

        # Grid lines
        ax.xaxis.grid(True, color='#141E30', linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close()

        # ── Scores Table ──────────────────────────────────────────
        st.markdown('<div class="chart-title">Raw Scores</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            'Emotion':    [f"{EMOTION_EMOJIS[e]}  {e.capitalize()}" for e in emotions],
            'Confidence': [f"{p * 100:.2f}%" for p in probs],
        }).sort_values('Confidence', ascending=False).reset_index(drop=True)

        st.dataframe(
            prob_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Emotion":    st.column_config.TextColumn("Emotion"),
                "Confidence": st.column_config.TextColumn("Confidence"),
            }
        )

# ─── Footer ───────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    EmoSense · Emotion Detection via LSTM · Deep Learning Final Project
</div>
""", unsafe_allow_html=True)
