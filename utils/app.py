import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="VidMind",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tiempos+Text:ital,wght@0,400;0,500;1,400&family=Söhne:wght@300;400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Claude palette ── */
:root {
    --cream:    #f5f0e8;
    --warm-0:   #faf8f5;
    --warm-1:   #f5f0e8;
    --warm-2:   #ede8de;
    --warm-3:   #e0d9cd;
    --warm-4:   #cdc5b6;
    --warm-5:   #b5ab9a;
    --brown-1:  #6b5e4e;
    --brown-2:  #4a3f33;
    --brown-3:  #2d2520;
    --ink:      #1a1410;
    --claude:   #d97757;
    --claude-l: #e8956d;
    --claude-d: #c4623e;
    --claude-bg: rgba(217,119,87,0.08);
    --claude-border: rgba(217,119,87,0.2);
    --sage:     #7a9e87;
    --sage-bg:  rgba(122,158,135,0.1);
    --muted:    #8a7d70;
    --border:   #ddd6c8;
    --border-l: #ede8de;
    --shadow:   rgba(26,20,16,0.06);
    --shadow-m: rgba(26,20,16,0.12);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--warm-0) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--warm-0) !important;
    min-height: 100vh;
}

/* Hide chrome */
[data-testid="stSidebar"]   { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer    { display: none !important; }

/* ── Main container width ── */
.block-container {
    max-width: 900px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.75rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3.5rem;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Lora', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.topbar-gem {
    width: 22px; height: 22px;
    background: var(--claude);
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
}
.topbar-nav {
    display: flex;
    gap: 1.75rem;
    font-size: 0.82rem;
    color: var(--muted);
    font-weight: 400;
}

/* ── Hero ── */
.hero {
    padding: 0.5rem 0 3rem;
    text-align: center;
}
.hero-h {
    font-family: 'Lora', serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.2;
    color: var(--ink);
    margin-bottom: 0.9rem;
}
.hero-h em {
    font-style: italic;
    color: var(--claude);
}
.hero-p {
    font-size: 0.95rem;
    color: var(--muted);
    line-height: 1.7;
    max-width: 460px;
    margin: 0 auto;
    font-weight: 300;
}

/* ── Input panel ── */
.input-panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin: 0 auto 1rem;
    box-shadow: 0 2px 16px var(--shadow);
}
.input-panel-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--warm-5);
    margin-bottom: 0.6rem;
    display: block;
}

/* Streamlit input overrides */
.stTextInput > div > div > input {
    background: var(--warm-0) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--claude) !important;
    box-shadow: 0 0 0 3px var(--claude-border) !important;
    background: white !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--warm-5) !important;
    font-family: 'DM Mono', monospace !important;
}
label[data-testid="stWidgetLabel"] { display: none !important; }

.stSelectbox > div > div {
    background: var(--warm-0) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-size: 0.85rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--claude) !important;
    box-shadow: 0 0 0 3px var(--claude-border) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--claude) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.7rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    box-shadow: 0 1px 3px rgba(217,119,87,0.3) !important;
}
.stButton > button:hover {
    background: var(--claude-d) !important;
    box-shadow: 0 4px 12px rgba(217,119,87,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary (ghost) button override via data attribute trick */
.btn-ghost > button {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--brown-2) !important;
    box-shadow: none !important;
}
.btn-ghost > button:hover {
    border-color: var(--claude) !important;
    color: var(--claude) !important;
    box-shadow: none !important;
    background: var(--claude-bg) !important;
}

/* ── Input hints ── */
.input-hints {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-l);
}
.hint {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--warm-5);
}
.hint-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--claude);
    opacity: 0.5;
    flex-shrink: 0;
}

/* ── Progress ── */
.prog-wrap {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 2px 16px var(--shadow);
}
.prog-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 1.25rem;
}
.prog-title {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink);
}
.prog-pct {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 400;
    color: var(--claude);
    line-height: 1;
}
.prog-pct small { font-size: 0.9rem; color: var(--warm-4); }
.prog-track {
    height: 3px;
    background: var(--warm-2);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.prog-bar {
    height: 100%;
    background: var(--claude);
    border-radius: 999px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.steps-list { display: flex; flex-direction: column; gap: 0.4rem; }
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    font-size: 0.83rem;
    border: 1px solid transparent;
    transition: all 0.2s;
}
.step-row.pending { color: var(--warm-4); }
.step-row.active  { background: var(--claude-bg); border-color: var(--claude-border); color: var(--ink); }
.step-row.done    { background: var(--sage-bg); color: var(--sage); }

.sdot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sdot.pending { background: var(--warm-3); }
.sdot.active  { background: var(--claude); box-shadow: 0 0 6px rgba(217,119,87,0.5); animation: blink 1.2s infinite; }
.sdot.done    { background: var(--sage); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.stag {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}
.stag.pending { background: var(--warm-2); color: var(--warm-4); }
.stag.active  { background: var(--claude-bg); color: var(--claude); }
.stag.done    { background: var(--sage-bg); color: var(--sage); }

/* ── Divider ── */
.rule {
    height: 1px;
    background: var(--border-l);
    margin: 2.5rem 0;
}

/* ── Section label ── */
.sec {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}
.sec-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    background: var(--warm-2);
    color: var(--warm-5);
    letter-spacing: 0.04em;
}
.sec-name {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.sec-rule { flex: 1; height: 1px; background: var(--border-l); }

/* ── Title banner ── */
.title-band {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 8px var(--shadow);
}
.title-band-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--claude);
    margin-bottom: 0.5rem;
}
.title-band-text {
    font-family: 'Lora', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.3;
    letter-spacing: -0.02em;
    margin-bottom: 0.85rem;
}
.title-band-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.65rem;
    border-radius: 5px;
    background: var(--warm-1);
    border: 1px solid var(--border);
    color: var(--muted);
}
.chip.active { background: var(--claude-bg); border-color: var(--claude-border); color: var(--claude-d); }

/* ── Info cards ── */
.icard {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.35rem 1.5rem;
    height: 100%;
    box-shadow: 0 1px 8px var(--shadow);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.icard:hover { box-shadow: 0 4px 20px var(--shadow-m); border-color: var(--warm-4); }
.icard-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-l);
}
.icard-icon {
    width: 28px; height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    background: var(--warm-1);
    border: 1px solid var(--border-l);
    flex-shrink: 0;
}
.icard-lbl {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}
.icard-body {
    font-size: 0.875rem;
    line-height: 1.75;
    color: var(--brown-1);
    font-weight: 300;
}

/* Stats card */
.stats-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 1px 8px var(--shadow);
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem 1rem;
}
.stat-n {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 400;
    color: var(--claude);
    line-height: 1;
}
.stat-l {
    font-size: 0.68rem;
    color: var(--warm-5);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Transcript */
.tx-box {
    background: var(--warm-0);
    border: 1px solid var(--border-l);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    line-height: 1.95;
    max-height: 260px;
    overflow-y: auto;
    color: var(--muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Chat ── */
.chat-shell {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 16px var(--shadow);
    margin-bottom: 0.75rem;
}
.chat-bar {
    background: var(--warm-0);
    border-bottom: 1px solid var(--border-l);
    padding: 0.9rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-bar-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--ink);
}
.live {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--sage);
    box-shadow: 0 0 6px var(--sage);
    animation: blink 2.5s infinite;
}
.chat-bar-right {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--warm-4);
}
.chat-body {
    padding: 1.5rem;
    min-height: 260px;
    max-height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}
.chat-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.4rem;
    padding: 2rem;
}
.chat-empty-icon { font-size: 2rem; margin-bottom: 0.25rem; opacity: 0.5; }
.chat-empty-h { font-size: 0.9rem; font-weight: 500; color: var(--brown-1); }
.chat-empty-p { font-size: 0.8rem; color: var(--warm-5); }
.suggestions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.85rem;
}
.sug {
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.73rem;
    background: var(--warm-1);
    border: 1px solid var(--border);
    color: var(--muted);
}

.msg { display: flex; flex-direction: column; gap: 0.25rem; }
.msg.u { align-items: flex-end; }
.msg.b { align-items: flex-start; }
.msg-who {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0 0.3rem;
}
.msg-who.u { color: var(--claude); }
.msg-who.b { color: var(--muted); }
.msg-bub {
    max-width: 76%;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    font-size: 0.875rem;
    line-height: 1.65;
    font-weight: 300;
}
.msg-bub.u {
    background: var(--claude-bg);
    border: 1px solid var(--claude-border);
    color: var(--brown-2);
    border-bottom-right-radius: 3px;
}
.msg-bub.b {
    background: var(--warm-1);
    border: 1px solid var(--border-l);
    color: var(--brown-2);
    border-bottom-left-radius: 3px;
}

/* ── Expander overrides ── */
.stExpander {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 8px var(--shadow) !important;
}
.stExpander summary {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    padding: 0.85rem 1.1rem !important;
}

/* ── Misc ── */
.stAlert {
    background: var(--warm-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--brown-2) !important;
    font-size: 0.85rem !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--warm-3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--warm-4); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {
    "result": None, "chat_history": [],
    "pipeline_done": False, "pipeline_steps": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEPS = [
    ("audio",      "🔊", "Audio processing",   15),
    ("transcript", "📝", "Transcription",       40),
    ("title",      "🏷️",  "Title generation",    5),
    ("summary",    "📋", "Summarisation",       15),
    ("extract",    "🔍", "Insight extraction",  15),
    ("rag",        "🧠", "RAG engine",          10),
]

def calc_pct(steps):
    t = 0
    for key, _, _, w in STEPS:
        s = steps.get(key, "pending")
        if s == "done":   t += w
        elif s == "active": t += w // 2
    return min(t, 100)

def sc(steps, key):
    return steps.get(key, "pending")

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-gem">◆</div>
        VidMind
    </div>
    <div class="topbar-nav">
        <span>Transcribe</span>
        <span>Summarise</span>
        <span>Chat</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Landing state ─────────────────────────────────────────────────────────────
if not st.session_state.result:

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-h">Meet your<br><em>meeting intelligence</em></div>
        <div class="hero-p">
            Drop a YouTube link or file path. VidMind transcribes,
            summarises, and lets you have a conversation with your content.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input panel
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<span class="input-panel-label">Video source</span>', unsafe_allow_html=True)

    col_url, col_lang = st.columns([5, 1], gap="medium")
    with col_url:
        source = st.text_input(
            "src", label_visibility="collapsed",
            placeholder="https://youtube.com/watch?v=... or /path/to/video.mp4",
            key="src"
        )
    with col_lang:
        language = st.selectbox("lang", ["english", "hinglish"],
                                label_visibility="collapsed", key="lng")

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    bc, hc = st.columns([1, 3], gap="medium")
    with bc:
        run_btn = st.button("Analyse →", use_container_width=True)
    with hc:
        st.markdown("""
        <div class="input-hints">
            <div class="hint"><div class="hint-dot"></div>YouTube, MP4, WAV, MP3</div>
            <div class="hint"><div class="hint-dot"></div>English &amp; Hinglish</div>
            <div class="hint"><div class="hint-dot"></div>Whisper · Mistral · RAG</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    source, language, run_btn = "", "english", False
    _, nc = st.columns([5, 1])
    with nc:
        if st.button("+ New", use_container_width=True):
            for k in ["result","chat_history","pipeline_done","pipeline_steps"]:
                st.session_state[k] = {} if k=="pipeline_steps" else ([] if k=="chat_history" else (False if k=="pipeline_done" else None))
            st.rerun()

# ── Pipeline runner ───────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.update({
            "pipeline_done": False, "result": None,
            "chat_history": [], "pipeline_steps": {}
        })
        ph = st.empty()

        def upd(k, s): st.session_state.pipeline_steps[k] = s

        def draw():
            pct = calc_pct(st.session_state.pipeline_steps)
            rows = ""
            for k, icon, lbl, _ in STEPS:
                c = sc(st.session_state.pipeline_steps, k)
                tag = {"done":"✓ Done","active":"Running…","pending":"Queued"}.get(c,"Queued")
                rows += f"""
                <div class="step-row {c}">
                    <div class="sdot {c}"></div>
                    <span>{icon}&nbsp; {lbl}</span>
                    <span class="stag {c}">{tag}</span>
                </div>"""
            ph.markdown(f"""
            <div class="prog-wrap">
                <div class="prog-head">
                    <div class="prog-title">Analysing your video</div>
                    <div class="prog-pct">{pct}<small>%</small></div>
                </div>
                <div class="prog-track">
                    <div class="prog-bar" style="width:{pct}%"></div>
                </div>
                <div class="steps-list">{rows}</div>
            </div>""", unsafe_allow_html=True)

        try:
            upd("audio","active");      draw(); chunks     = process_input(source);            upd("audio","done")
            upd("transcript","active"); draw(); transcript = transcribe_all(chunks, language); upd("transcript","done")
            upd("title","active");      draw(); title      = generate_title(transcript);       upd("title","done")
            upd("summary","active");    draw(); summary    = summarize(transcript);             upd("summary","done")
            upd("extract","active");    draw()
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            upd("extract","done")
            upd("rag","active");        draw(); rag_chain  = build_rag_chain(transcript);      upd("rag","done")
            draw()

            st.session_state.result = {
                "title": title, "transcript": transcript,
                "summary": summary, "action_items": action_items,
                "key_decisions": decisions, "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            time.sleep(0.3); ph.empty(); st.rerun()

        except Exception as e:
            for k, *_ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            ph.error(f"Pipeline error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    words = len(r['transcript'].split())

    # Title band
    st.markdown(f"""
    <div class="title-band">
        <div class="title-band-label">Session title</div>
        <div class="title-band-text">{r['title']}</div>
        <div class="title-band-chips">
            <span class="chip active">✓ Transcribed</span>
            <span class="chip active">✓ Summarised</span>
            <span class="chip active">✓ RAG ready</span>
            <span class="chip">{words:,} words</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 01 — Overview
    st.markdown("""
    <div class="sec">
        <span class="sec-tag">01</span>
        <span class="sec-name">Overview</span>
        <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    oc1, oc2 = st.columns([3, 2], gap="medium")
    with oc1:
        st.markdown(f"""
        <div class="icard">
            <div class="icard-head">
                <div class="icard-icon">📋</div>
                <span class="icard-lbl">Summary</span>
            </div>
            <div class="icard-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)
    with oc2:
        a_count = len([x for x in r['action_items'].split('\n') if x.strip()])
        d_count = len([x for x in r['key_decisions'].split('\n') if x.strip()])
        q_count = len([x for x in r['open_questions'].split('\n') if x.strip()])
        st.markdown(f"""
        <div class="stats-card" style="margin-bottom:0.75rem">
            <div class="icard-head">
                <div class="icard-icon">📊</div>
                <span class="icard-lbl">At a glance</span>
            </div>
            <div class="stats-grid">
                <div><div class="stat-n">{words:,}</div><div class="stat-l">Words</div></div>
                <div><div class="stat-n">{a_count}</div><div class="stat-l">Action items</div></div>
                <div><div class="stat-n">{d_count}</div><div class="stat-l">Decisions</div></div>
                <div><div class="stat-n">{q_count}</div><div class="stat-l">Questions</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        with st.expander("📝 Full transcript"):
            st.markdown(f'<div class="tx-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # 02 — Insights
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec">
        <span class="sec-tag">02</span>
        <span class="sec-name">Extracted insights</span>
        <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3, gap="medium")
    with ic1:
        st.markdown(f"""
        <div class="icard">
            <div class="icard-head">
                <div class="icard-icon">✅</div>
                <span class="icard-lbl">Action items</span>
            </div>
            <div class="icard-body">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="icard">
            <div class="icard-head">
                <div class="icard-icon">🔑</div>
                <span class="icard-lbl">Key decisions</span>
            </div>
            <div class="icard-body">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="icard">
            <div class="icard-head">
                <div class="icard-icon">❓</div>
                <span class="icard-lbl">Open questions</span>
            </div>
            <div class="icard-body">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    # 03 — Chat
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec">
        <span class="sec-tag">03</span>
        <span class="sec-name">Chat with your meeting</span>
        <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    msg_count = len(st.session_state.chat_history) // 2
    st.markdown(f"""
    <div class="chat-shell">
        <div class="chat-bar">
            <div class="chat-bar-left">
                <div class="live"></div>
                VidMind Assistant
            </div>
            <div class="chat-bar-right">{msg_count} exchange{"s" if msg_count!=1 else ""} · RAG · Mistral</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.chat_history:
        msgs = '<div class="chat-body">'
        for msg in st.session_state.chat_history:
            c = "u" if msg["role"]=="user" else "b"
            who = "You" if msg["role"]=="user" else "VidMind"
            msgs += f"""
            <div class="msg {c}">
                <span class="msg-who {c}">{who}</span>
                <div class="msg-bub {c}">{msg['content']}</div>
            </div>"""
        msgs += '</div>'
        st.markdown(msgs, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-body">
            <div class="chat-empty">
                <div class="chat-empty-icon">◆</div>
                <div class="chat-empty-h">Ready to help</div>
                <div class="chat-empty-p">Ask anything about your meeting</div>
                <div class="suggestions">
                    <span class="sug">What were the key decisions?</span>
                    <span class="sug">Summarise the action items</span>
                    <span class="sug">What topics came up?</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    qi1, qi2, qi3 = st.columns([5, 1, 1], gap="small")
    with qi1:
        user_q = st.text_input("q", placeholder="Ask anything about this meeting…",
                               label_visibility="collapsed", key="cq")
    with qi2:
        send = st.button("Send", use_container_width=True)
    with qi3:
        if st.session_state.chat_history:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    if send and user_q.strip():
        with st.spinner("Thinking…"):
            ans = ask_question(r["rag_chain"], user_q.strip())
        st.session_state.chat_history += [
            {"role": "user", "content": user_q.strip()},
            {"role": "assistant", "content": ans},
        ]
        st.rerun()