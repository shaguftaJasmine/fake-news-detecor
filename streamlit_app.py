"""
FAKE NEWS DETECTOR - PROFESSIONAL UI UPGRADE
Created by Shagufta
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fake News Detector - by Shagufta",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================
st.markdown("""
<style>
    /* Main background and fonts */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #666;
        font-size: 1.2rem;
    }
    
    .creator-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input section */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        background: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .fake-result {
        border-left: 5px solid #ff4757;
        background: linear-gradient(145deg, #fff5f5, white);
    }
    
    .real-result {
        border-left: 5px solid #00d25b;
        background: linear-gradient(145deg, #f0fff4, white);
    }
    
    .uncertain-result {
        border-left: 5px solid #ffa502;
        background: linear-gradient(145deg, #fff9f0, white);
    }
    
    /* Probability meters */
    .meter-container {
        margin: 1.5rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 15px;
    }
    
    .meter-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .meter-fake .meter-label span {
        color: #ff4757;
    }
    
    .meter-real .meter-label span {
        color: #00d25b;
    }
    
    .meter-bar {
        height: 25px;
        background: #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
        position: relative;
    }
    
    .meter-fill {
        height: 100%;
        width: 0%;
        transition: width 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .meter-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            rgba(255,255,255,0.3) 0%,
            rgba(255,255,255,0.1) 25%,
            transparent 50%,
            rgba(255,255,255,0.1) 75%,
            rgba(255,255,255,0.3) 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .meter-fake .meter-fill {
        background: linear-gradient(90deg, #ff4757, #ff6b81);
    }
    
    .meter-real .meter-fill {
        background: linear-gradient(90deg, #00d25b, #28a745);
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        transition: all 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* History items */
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid;
        transition: transform 0.2s;
    }
    
    .history-item:hover {
        transform: translateX(5px);
    }
    
    .history-fake {
        border-left-color: #ff4757;
        background: #fff5f5;
    }
    
    .history-real {
        border-left-color: #00d25b;
        background: #f0fff4;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: #2c3e50;
        color: white;
        text-align: center;
        padding: 0.5rem;
        border-radius: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Signature */
    .signature {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 1rem;
        padding: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    .signature strong {
        color: white;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'fake_count' not in st.session_state:
    st.session_state.fake_count = 0
if 'real_count' not in st.session_state:
    st.session_state.real_count = 0

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    try:
        with st.spinner('🤖 Loading AI models...'):
            time.sleep(1)  # Dramatic effect
            model = joblib.load('fakenews_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            return model, vectorizer
    except FileNotFoundError:
        st.error("""
        ❌ **Model files not found!**
        
        Please run `train_model.py` first to create the model files.
        
        In your terminal, type:
        ```
        python train_model.py
        ```
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

model, vectorizer = load_models()

# ============================================
# HEADER SECTION - WITH YOUR NAME
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🔍 Fake News Detector AI</h1>
    <p>Advanced machine learning system to identify misleading information with 99% accuracy</p>
    <div class="creator-badge">
        👨‍💻 Created by <strong>Shagufta</strong>
    </div>
    <div style="margin-top: 1.5rem;">
        <span class="badge" style="background: #667eea; color: white; padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">⚡ Real-time Analysis</span>
        <span class="badge" style="background: #764ba2; color: white; padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">🎯 99.36% Accuracy</span>
        <span class="badge" style="background: #00d25b; color: white; padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">📊 68K+ Articles Trained</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# MAIN LAYOUT
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    # Input Section
    st.markdown("""
    <div class="input-container">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">📝 Enter News Text</h3>
    """, unsafe_allow_html=True)
    
    # Text input with character count
    news_text = st.text_area(
        label="",
        placeholder="Paste a news headline or article here... (e.g., 'Scientists discover breakthrough in cancer research')",
        height=150,
        key="news_input"
    )
    
    # Character counter
    if news_text:
        char_count = len(news_text)
        word_count = len(news_text.split())
        st.caption(f"📊 {char_count} characters | {word_count} words")
    
    # Settings expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher values make the detector more conservative (only flags very clear fake news)"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            show_confidence = st.checkbox("Show Confidence Meter", value=True)
        with col_b:
            save_history = st.checkbox("Save to History", value=True)
    
    # Analyze button with animation
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_btn = st.button(
            "🔍 ANALYZE CREDIBILITY",
            type="primary",
            use_container_width=True,
            help="Click to analyze the news text"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Stats Dashboard
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">📊 Statistics</h3>
    """, unsafe_allow_html=True)
    
    # Stats cards
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-value">{st.session_state.analysis_count}</div>
        <div class="stat-label">Total Analyses</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🚨</div>
            <div class="stat-value" style="color: #ff4757;">{st.session_state.fake_count}</div>
            <div class="stat-label">Fake Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">✅</div>
            <div class="stat-value" style="color: #00d25b;">{st.session_state.real_count}</div>
            <div class="stat-label">Real Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model info
    st.markdown("""
    <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            <span class="tooltip">🤖 Model Info
                <span class="tooltiptext">TF-IDF + Logistic Regression<br>Trained on 68,380 articles<br>99.36% accuracy<br>Created by Chagoofta</span>
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# ANALYSIS RESULTS
# ============================================
if analyze_btn and news_text:
    with st.spinner("🧠 Analyzing with AI..."):
        # Simulate thinking
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Transform text and predict
        features = vectorizer.transform([news_text.lower()])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        fake_prob = proba[0]  # Probability of FAKE
        real_prob = proba[1]  # Probability of REAL
        
        # Determine result
        is_fake = fake_prob >= threshold
        
        # Update stats
        st.session_state.analysis_count += 1
        if is_fake:
            st.session_state.fake_count += 1
        else:
            st.session_state.real_count += 1
        
        # Save to history
        if save_history:
            st.session_state.history.append({
                'text': news_text[:50] + '...' if len(news_text) > 50 else news_text,
                'result': 'FAKE' if is_fake else 'REAL',
                'confidence': max(fake_prob, real_prob),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            # Keep only last 10
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[-10:]
    
    # Display Results
    st.markdown("---")
    
    # Result header
    if is_fake:
        st.markdown("""
        <div class="result-card fake-result">
            <h1 style="color: #ff4757; font-size: 3rem; margin-bottom: 0.5rem;">🚨 FAKE NEWS DETECTED</h1>
            <p style="color: #666; font-size: 1.2rem;">This content appears to be misleading or fabricated.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card real-result">
            <h1 style="color: #00d25b; font-size: 3rem; margin-bottom: 0.5rem;">✅ LIKELY REAL NEWS</h1>
            <p style="color: #666; font-size: 1.2rem;">This content appears to be credible and factual.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability meters
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown(f"""
        <div class="meter-container meter-fake">
            <div class="meter-label">
                <span>🚨 FAKE Probability</span>
                <span>{fake_prob:.1%}</span>
            </div>
            <div class="meter-bar">
                <div class="meter-fill" style="width: {fake_prob*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="meter-container meter-real">
            <div class="meter-label">
                <span>✅ REAL Probability</span>
                <span>{real_prob:.1%}</span>
            </div>
            <div class="meter-bar">
                <div class="meter-fill" style="width: {real_prob*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.metric(
            label="Confidence",
            value=f"{max(fake_prob, real_prob):.1%}",
            delta="High" if max(fake_prob, real_prob) > 0.8 else "Moderate"
        )
    
    with col_d2:
        st.metric(
            label="Decision Threshold",
            value=f"{threshold:.1%}",
            delta="Strict" if threshold > 0.7 else "Balanced"
        )
    
    with col_d3:
        st.metric(
            label="Text Length",
            value=f"{len(news_text)} chars",
            delta=f"{len(news_text.split())} words"
        )
    
    with col_d4:
        st.metric(
            label="Analysis Time",
            value=datetime.now().strftime("%H:%M:%S"),
            delta="Now"
        )
    
    # Text analysis
    with st.expander("🔍 Detailed Text Analysis", expanded=False):
        st.markdown("**Original Text:**")
        st.info(news_text)
        
        # Show text statistics
        words = news_text.split()
        st.markdown("**Text Statistics:**")
        st.json({
            "Total Characters": len(news_text),
            "Total Words": len(words),
            "Average Word Length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
            "Unique Words": len(set(w.lower() for w in words))
        })

elif analyze_btn and not news_text:
    st.warning("⚠️ Please enter some text to analyze.")

# ============================================
# HISTORY SECTION
# ============================================
if st.session_state.history:
    st.markdown("---")
    st.markdown("""
    <h3 style="color: white; margin-bottom: 1rem;">📜 Recent Analyses</h3>
    """, unsafe_allow_html=True)
    
    # Create history cards
    for item in reversed(st.session_state.history):
        history_class = "history-fake" if item['result'] == 'FAKE' else "history-real"
        icon = "🚨" if item['result'] == 'FAKE' else "✅"
        
        st.markdown(f"""
        <div class="history-item {history_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                    <span style="font-weight: 600;">{item['result']}</span>
                    <span style="color: #666; margin-left: 1rem;">{item['text']}</span>
                </div>
                <div>
                    <span style="background: {'#ff4757' if item['result'] == 'FAKE' else '#00d25b'}; 
                                 color: white; padding: 0.2rem 0.5rem; border-radius: 5px;">
                        {item['confidence']:.1%}
                    </span>
                    <span style="color: #999; margin-left: 1rem; font-size: 0.9rem;">
                        {item['timestamp']}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_count = 0
        st.session_state.fake_count = 0
        st.session_state.real_count = 0
        st.rerun()

# ============================================
# FOOTER - WITH YOUR NAME
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); padding: 2rem;">
    <p style="margin-bottom: 0.5rem;">
        🚀 Built with Streamlit | 🤖 Machine Learning | 📊 99.36% Accuracy
    </p>
    <div class="signature">
        👨‍💻 Created with ❤️ by <strong>Shagufta</strong> | © 2026 All Rights Reserved
    </div>
    <p style="font-size: 0.9rem; opacity: 0.7; margin-top: 1rem;">
        ⚠️ Disclaimer: This tool is for educational purposes. Always verify news from multiple reliable sources.
    </p>
</div>
""", unsafe_allow_html=True)