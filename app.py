import streamlit as st
import nltk
import string
import difflib
import datetime
import pandas as pd
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

# ===================== CONFIGURATION & SETUP =====================
st.set_page_config(
    page_title="PlagiCheck Real-Time Document Similarity Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data quietly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ===================== CUSTOM STYLING (CSS) =====================
def load_custom_css():
    st.markdown("""
        <style>
        /* --- FONTS & BASICS --- */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-dark: #0f1117;
            --bg-card: #1e2330;
            --primary: #4f46e5;
            --primary-glow: rgba(79, 70, 229, 0.4);
            --accent: #ec4899;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-main);
        }

        /* --- BACKGROUND (Cleaned) --- */
        .stApp {
            background-color: var(--bg-dark);
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #161a25;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        div[data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }

        /* --- METRIC CARDS (Kept as requested) --- */
        .metric-card {
            background: linear-gradient(145deg, rgba(30, 35, 48, 0.9), rgba(30, 35, 48, 0.6));
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .metric-label {
            font-size: 0.85rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* --- BUTTONS --- */
        .stButton > button {
            background: linear-gradient(90deg, var(--primary) 0%, #4338ca 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
            transform: translateY(-1px);
        }
        
        /* --- CLEAN UPLOADERS --- */
        div[data-testid="stFileUploader"] {
            background-color: rgba(255,255,255,0.03);
            padding: 1rem;
            border-radius: 10px;
        }

        /* --- SCROLL BOXES --- */
        .scroll-box {
            height: 600px;
            overflow-y: auto;
            background-color: #0d1117;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            color: #d1d5db;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.7;
        }
        .scroll-box::-webkit-scrollbar { width: 8px; }
        .scroll-box::-webkit-scrollbar-track { background: #0d1117; }
        .scroll-box::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }

        /* --- HIGHLIGHTS --- */
        mark {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border-bottom: 2px solid #ef4444;
            padding: 0 2px;
            border-radius: 2px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ===================== LOGIC FUNCTIONS =====================
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stop_words = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in stop_words)

def read_file(file):
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return " ".join(p.extract_text() or "" for p in reader.pages)
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""
    return ""

def calculate_metrics(t1, t2, raw1, raw2, threshold):
    vec = TfidfVectorizer()
    try:
        v = vec.fit_transform([t1, t2])
        sim_score = cosine_similarity(v[0], v[1])[0][0] * 100
    except ValueError:
        sim_score = 0.0

    s1 = [s.strip() for s in raw1.split(".") if len(s.strip()) > 10]
    s2 = [s.strip() for s in raw2.split(".") if len(s.strip()) > 10]

    rows, m1, m2 = [], [], []
    
    progress_bar = st.progress(0)
    total_sent = len(s1)
    
    for idx, sent in enumerate(s1):
        if idx % 10 == 0:
            progress_bar.progress(min(idx / total_sent, 1.0))
            
        match = difflib.get_close_matches(sent, s2, n=1, cutoff=threshold)
        if match:
            score = difflib.SequenceMatcher(None, sent, match[0]).ratio() * 100
            rows.append([sent, match[0], round(score, 2)])
            m1.append(sent)
            m2.append(match[0])
            
    progress_bar.empty()

    df = pd.DataFrame(
        rows,
        columns=["Source Segment", "Matched Reference Segment", "Similarity %"]
    )
    return sim_score, df, m1, m2

def highlight_text(text, sentences):
    processed_text = text
    for s in sorted(set(sentences), key=len, reverse=True):
        if len(s) < 5: continue
        processed_text = processed_text.replace(
            s,
            f"||MARK||{s}||ENDMARK||"
        )
    processed_text = processed_text.replace("||MARK||", "<mark>").replace("||ENDMARK||", "</mark>")
    processed_text = processed_text.replace("\n", "<br>")
    return processed_text

def get_risk_level(score):
    if score >= 75: return "CRITICAL RISK", "#ef4444"
    elif score >= 45: return "MODERATE RISK", "#f59e0b"
    return "SAFE CONTENT", "#10b981"

# ===================== SESSION STATE =====================
if "data" not in st.session_state:
    st.session_state.data = None
if "history" not in st.session_state:
    st.session_state.history = []

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2490/2490315.png", width=60)
    st.markdown("""
        <div style='margin-top: 10px; margin-bottom: 20px;'>
            <h2 style='margin:0; font-size: 1.5rem; color: #fff;'>PlagiCheck</h2>
            <p style='margin:0; font-size: 0.8rem; color: #94a3b8; letter-spacing: 1px;'>REAL-TIME DETECTION</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["Dashboard", "Analysis Detail", "Full Reports"],
        index=0,
        format_func=lambda x: f" {x.upper()}"
    )
    
    st.markdown("---")
    st.markdown("**🔍 ANALYSIS SETTINGS**")
    threshold = st.slider(
        "Sensitivity Level",
        min_value=0.4, max_value=0.9, value=0.5, step=0.05
    )
    st.caption("Higher values require closer matches.")
    
    st.markdown("---")
    st.caption("System Online • v3.1.0")

# ===================== PAGES =====================

# 1. DASHBOARD
if page == "Dashboard":
    # Clean Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 4rem; font-weight: 800; margin-bottom: 0.5rem; background: linear-gradient(to right, #4f46e5, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PlagiCheck
            </h1>
            <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 500;">
                Real-Time Document Similarity Detection System
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Clean Upload Area
    st.markdown("#### 📂 Document Upload")
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown("**📄 Suspect Document**")
        f1 = st.file_uploader("", type=["txt", "pdf", "docx"], key="f1", label_visibility="collapsed")
        
    with c2:
        st.markdown("**📚 Reference Corpus**")
        f2 = st.file_uploader("", type=["txt", "pdf", "docx"], key="f2", label_visibility="collapsed")

    # Processing Trigger
    st.markdown("<br>", unsafe_allow_html=True)
    col_centered = st.columns([1, 2, 1])
    with col_centered[1]:
        if f1 and f2:
            if st.button("INITIATE DEEP SCAN ⚡", type="primary", use_container_width=True):
                with st.spinner("Processing linguistic tokens..."):
                    start_time = time.time()
                    raw1 = read_file(f1)
                    raw2 = read_file(f2)
                    clean1 = preprocess_text(raw1)
                    clean2 = preprocess_text(raw2)
                    sim, sent_df, m1, m2 = calculate_metrics(clean1, clean2, raw1, raw2, threshold)
                    risk_label, risk_color = get_risk_level(sim)
                    
                    st.session_state.data = {
                        "raw1": raw1, "raw2": raw2,
                        "sim": sim, "risk": risk_label, "color": risk_color,
                        "sent_df": sent_df, "m1": m1, "m2": m2,
                        "f1_name": f1.name, "f2_name": f2.name
                    }
                    st.session_state.history.append({
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Source": f1.name, "Reference": f2.name,
                        "Similarity": f"{sim:.2f}%", "Risk": risk_label
                    })
                    time.sleep(0.5)
                    st.toast("Scan Complete.", icon="🚀")

    # Metrics Display
    if st.session_state.data:
        d = st.session_state.data
        st.markdown("---")
        st.markdown("#### 📊 Analysis Results")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{d['sim']:.1f}%</div>
                <div class="metric-label">Similarity Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {d['color']}; -webkit-text-fill-color: {d['color']};">{d['risk'].split()[0]}</div>
                <div class="metric-label">{d['risk'].split()[1]} LEVEL</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(d['sent_df'])}</div>
                <div class="metric-label">Overlaps Found</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(d['raw1'].split())}</div>
                <div class="metric-label">Word Count</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"Audit completed for **{d['f1_name']}** vs **{d['f2_name']}**. Proceed to 'Analysis Detail' for evidence.")

# 2. ANALYSIS DETAIL
elif page == "Analysis Detail":
    st.markdown("#### 📝 Evidence Visualizer")
    st.caption("Interactive forensic text comparison. Red highlights indicate detected matches.")
    
    if not st.session_state.data:
        st.warning("⚠️ No active session data. Please run a scan from the Dashboard.")
        st.stop()
        
    d = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**SOURCE: {d['f1_name']}**")
        html_content_1 = highlight_text(d['raw1'], d['m1'])
        st.markdown(f'<div class="scroll-box">{html_content_1}</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"**REFERENCE: {d['f2_name']}**")
        html_content_2 = highlight_text(d['raw2'], d['m2'])
        st.markdown(f'<div class="scroll-box">{html_content_2}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📂 View Granular Match Table", expanded=True):
        st.dataframe(d['sent_df'], use_container_width=True, hide_index=True)

# 3. REPORTS
elif page == "Full Reports":
    st.markdown("#### 📁 Audit History")
    
    if not st.session_state.history:
        st.info("No audit logs available.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            search = st.text_input("Search Logs", placeholder="Filter by filename...")
        
        if search:
            df_hist = df_hist[
                df_hist['Source'].str.contains(search, case=False) | 
                df_hist['Reference'].str.contains(search, case=False)
            ]
            
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Similarity": st.column_config.ProgressColumn(
                    "Similarity Score",
                    format="%s",
                    min_value=0,
                    max_value=100,
                ),
                "Risk": st.column_config.TextColumn("Risk Assessment")
            }
        )
        
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv,
            file_name='plagicheck_audit_log.csv',
            mime='text/csv'
        )
