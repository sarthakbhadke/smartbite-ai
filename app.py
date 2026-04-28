import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download
import nltk

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="SmartBite AI",
    page_icon="🍽️",
    layout="wide"
)

# ── Download NLTK data silently ───────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Load model and vectorizer ────────────────────────────
@st.cache_resource
def load_models():
    model      = joblib.load('models/fake_review_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    explainer  = joblib.load('models/shap_explainer.pkl')
    return model, vectorizer, explainer

model, vectorizer, explainer = load_models()

# ── Text cleaning function ────────────────────────────────
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer    = PorterStemmer()
    text       = text.lower()
    text       = re.sub(r'[^a-z\s]', '', text)
    words      = text.split()
    words      = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ── Header ────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#E05C2A;'>
        🍽️ SmartBite AI
    </h1>
    <p style='text-align:center; color:gray; font-size:17px;'>
        AI-Powered Fake Review Detection for Food-Tech Platforms
    </p>
    <p style='text-align:center; color:#888; font-size:13px;'>
        Built for Zomato · NLP + Machine Learning · Explainable AI (SHAP)
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Detect Fake Review",
    "📊 SHAP Explanation",
    "📈 Data Insights"
])

# ════════════════════════════════════════════════════════
# TAB 1 — DETECTOR
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter a restaurant review")
    
    # Example buttons
    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    
    example_text = ""
    with col1:
        if st.button("🟢 Likely Genuine"):
            example_text = "I visited this place last Sunday with my family. The dal makhani was rich and creamy, portions were generous and the staff was warm and attentive throughout our meal."
    with col2:
        if st.button("🔴 Likely Fake"):
            example_text = "Best restaurant ever! Amazing food! Must visit!"
    with col3:
        if st.button("⚪ Neutral"):
            example_text = "The food was okay. Service was average. Nothing special."

    review_input = st.text_area(
        "Type or paste a review here:",
        value=example_text,
        height=150,
        placeholder="e.g. The biryani here is absolutely fantastic, best in the city!"
    )

    if st.button("🔎 Analyse Review", type="primary"):
        if review_input.strip() == "":
            st.warning("Please enter a review first.")
        else:
            clean  = clean_text(review_input)
            vec    = vectorizer.transform([clean])
            pred   = model.predict(vec)[0]
            proba  = model.predict_proba(vec)[0]
            conf   = round(max(proba) * 100, 1)
            
            st.markdown("---")
            
            if pred == 1:
                st.error(f"🚨 **FAKE REVIEW DETECTED** — {conf}% confidence")
                st.markdown("> This review shows patterns commonly associated with fake or spam reviews.")
            else:
                st.success(f"✅ **GENUINE REVIEW** — {conf}% confidence")
                st.markdown("> This review shows patterns associated with authentic customer experiences.")
            
            # Metrics row
            word_count = len(review_input.split())
            m1, m2, m3 = st.columns(3)
            m1.metric("Word Count",     word_count)
            m2.metric("Confidence",     f"{conf}%")
            m3.metric("Verdict",        "Fake" if pred == 1 else "Genuine")
            
            # Store in session for Tab 2
            st.session_state['last_review'] = review_input
            st.session_state['last_clean']  = clean
            st.session_state['last_pred']   = pred
            st.session_state['last_conf']   = conf
            
            st.info("👉 Go to the **SHAP Explanation** tab to see WHY the model made this decision.")

# ════════════════════════════════════════════════════════
# TAB 2 — SHAP
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Why did the model make this decision?")
    
    if 'last_clean' not in st.session_state:
        st.info("👈 First analyse a review in the **Detect Fake Review** tab.")
    else:
        review_display = st.session_state['last_review']
        clean          = st.session_state['last_clean']
        pred           = st.session_state['last_pred']
        conf           = st.session_state['last_conf']
        
        verdict = "🚨 FAKE" if pred == 1 else "✅ GENUINE"
        st.markdown(f"**Review:** *{review_display[:120]}...*")
        st.markdown(f"**Verdict:** {verdict} ({conf}% confident)")
        st.markdown("---")
        
        # Generate SHAP for this review
        vec         = vectorizer.transform([clean])
        single_shap = explainer.shap_values(vec)[0]
        
        feature_names = vectorizer.get_feature_names_out()
        
        word_shap = []
        for word in clean.split():
            if word in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[word]
                word_shap.append((word, single_shap[idx]))
        
        if word_shap:
            word_shap   = sorted(word_shap, key=lambda x: abs(x[1]), reverse=True)[:12]
            words_list  = [w[0] for w in word_shap]
            values_list = [w[1] for w in word_shap]
            colors      = ['tomato' if v > 0 else 'steelblue' for v in values_list]
            
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(words_list, values_list, color=colors)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_title(
                f'Word-level explanation\n"{review_display[:55]}..."',
                fontsize=11, pad=10
            )
            ax.set_xlabel('SHAP Value  (red = pushes toward FAKE, blue = toward GENUINE)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("**How to read this chart:**")
            st.markdown("- 🔴 **Red bars** — these words pushed the model toward **FAKE**")
            st.markdown("- 🔵 **Blue bars** — these words pushed the model toward **GENUINE**")
            st.markdown("- Longer bar = stronger influence on the decision")
        else:
            st.warning("No matching words found in vocabulary for this review.")

# ════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Review Length: Genuine vs Suspicious**")
        try:
            st.image('data/review_length_chart.png', width=600)
        except:
            st.info("Chart not found. Run Day 3 notebook first.")
    
    with col2:
        st.markdown("**Top Words: FAKE vs GENUINE (SHAP)**")
        try:
            st.image('data/shap_top_words.png', width=600)
        except:
            st.info("Chart not found. Run Day 5 notebook first.")
    
    st.markdown("---")
    st.markdown("### Key Findings")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews Analysed", "22,498")
    c2.metric("Model Accuracy",         "83.38%")
    c3.metric("F1-Score (Fake)",        "0.84")
    
    st.markdown("""
    - 📌 Genuine reviews are on average **55% longer** than suspicious ones
    - 📌 Words like *amazing, best, must* are strong fake indicators
    - 📌 Words like *however, meal, appetizer* indicate genuine reviews  
    - 📌 Model trained on **balanced dataset** (50% fake / 50% genuine)
    - 📌 **SHAP explainability** added for transparent predictions
    """)

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "SmartBite AI · Built by SARTHAK BHADKE · MCA Student · Amity University Jaipur"
    "</p>",
    unsafe_allow_html=True
)