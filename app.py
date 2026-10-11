import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from datetime import datetime
from pathlib import Path

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD TRAINED AI MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    model, vectorizer = load_model()

except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()

# =====================================
# SESSION STATE
# =====================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =====================================
# MODERN DASHBOARD STYLING
# =====================================

st.markdown("""
<style>

.stApp {
    background: #0b1220;
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg, #2563eb, #7c3aed
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background: #1d4ed8;
    color: white;
}

.stTextArea textarea {
    background: #17243a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🛡️ TruthLens AI")
    st.caption("AI-Powered News Analysis")

    st.divider()

    st.subheader("👨‍💻 Developer")
    st.write("Sidharth")

    st.divider()

    st.subheader("⚙️ Technologies")
    st.write("🐍 Python")
    st.write("🤖 Machine Learning")
    st.write("📊 Scikit-learn")
    st.write("🌐 Streamlit")
    st.write("📈 Plotly")
    st.write("📰 TF-IDF")
    st.write("🧠 Logistic Regression")

    st.divider()

    st.info(
        "TruthLens AI uses machine learning "
        "to classify news articles."
    )

    st.caption(
        "Predictions are not verified fact-checks."
    )

# =====================================
# HEADER
# =====================================

st.title("🛡️ TruthLens AI")

st.subheader(
    "Intelligent Fake News Detection System"
)

st.caption("Developed by Sidharth")

st.divider()

# =====================================
# ANALYZE NEWS
# =====================================

st.header("🔍 Analyze News")

st.write(
    "Paste a news article below. "
    "The AI model will classify its text "
    "as resembling Fake or Real news "
    "from its training dataset."
)

news = st.text_area(
    "📰 Enter News Article",
    placeholder="Paste your news article here...",
    height=220
)

analyze = st.button(
    "🚀 Analyze News",
    type="primary",
    use_container_width=True
)

if analyze:

    if not news.strip():

        st.warning(
            "⚠️ Please enter a news article."
        )

    else:

        # Convert text into numerical features
        features = vectorizer.transform([news])

        # Predict news class
        prediction = int(
            model.predict(features)[0]
        )

        # Calculate confidence
        probabilities = model.predict_proba(
            features
        )[0]

        confidence = float(
            max(probabilities)
        )

        result = (
            "FAKE" if prediction == 0
            else "REAL"
        )

        # Save latest result
        st.session_state.last_result = {
            "prediction": result,
            "confidence": confidence
        }

        # Save prediction history
        st.session_state.history.append({
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "News": news[:150],
            "Prediction": result,
            "Confidence": round(
                confidence * 100, 2
            )
        })

# =====================================
# LATEST PREDICTION RESULT
# =====================================

if st.session_state.last_result:

    latest = st.session_state.last_result

    st.divider()
    st.header("📊 AI Analysis Results")

    col1, col2 = st.columns(2)

    with col1:

        if latest["prediction"] == "FAKE":

            st.error(
                "🚨 Prediction: FAKE NEWS"
            )

        else:

            st.success(
                "✅ Prediction: REAL NEWS"
            )

    with col2:

        st.metric(
            "AI Model Confidence",
            f"{latest['confidence'] * 100:.2f}%"
        )

    st.subheader("📈 Confidence Level")

    st.progress(
        latest["confidence"]
    )

    st.info(
        "This is a machine-learning prediction, "
        "not independent fact-checking. "
        "High model confidence does not "
        "guarantee factual correctness."
    )

# =====================================
# PREDICTION STATISTICS
# =====================================

st.divider()
st.header("📊 Prediction Statistics")

history = st.session_state.history

total_count = len(history)

fake_count = sum(
    item["Prediction"] == "FAKE"
    for item in history
)

real_count = sum(
    item["Prediction"] == "REAL"
    for item in history
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📰 Total Predictions",
        total_count
    )

with col2:

    st.metric(
        "🚨 Fake Predictions",
        fake_count
    )

with col3:

    st.metric(
        "✅ Real Predictions",
        real_count
    )

# =====================================
# INTERACTIVE CHARTS
# =====================================

st.divider()
st.header("📈 Interactive Analytics")

if total_count > 0:

    chart_data = pd.DataFrame({
        "Category": ["Fake News", "Real News"],
        "Count": [fake_count, real_count]
    })

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🥧 Prediction Distribution")

        pie_chart = px.pie(
            chart_data,
            names="Category",
            values="Count",
            hole=0.45,
            color="Category",
            color_discrete_map={
                "Fake News": "#ef4444",
                "Real News": "#22c55e"
            }
        )

        pie_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            font_color="white"
        )

        st.plotly_chart(
            pie_chart,
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Fake vs Real")

        bar_chart = px.bar(
            chart_data,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            color_discrete_map={
                "Fake News": "#ef4444",
                "Real News": "#22c55e"
            }
        )

        bar_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font_color="white",
            showlegend=False
        )

        st.plotly_chart(
            bar_chart,
            use_container_width=True
        )

else:

    st.info(
        "Analyze news articles to "
        "display interactive charts."
    )

# =====================================
# PREDICTION HISTORY
# =====================================

st.divider()
st.header("📜 Prediction History")

if history:

    history_df = pd.DataFrame(
        history[::-1]
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.NumberColumn(
                "Confidence (%)",
                format="%.2f%%"
            )
        }
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="📥 Download History CSV",
            data=csv_data,
            file_name="truthlens_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):

            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

else:

    st.info(
        "No predictions yet. "
        "Analyze an article to get started."
    )

# =====================================
# ABOUT PROJECT
# =====================================

st.divider()
st.header("💡 About TruthLens AI")

st.write(
    "TruthLens AI is an AI-powered "
    "fake news classification mini-project "
    "developed by Sidharth. "
    "It uses Natural Language Processing "
    "(NLP), TF-IDF vectorization, and "
    "Logistic Regression to classify "
    "news articles."
)

st.write(
    "The dashboard provides AI predictions, "
    "model confidence scores, prediction "
    "history, interactive charts, "
    "statistics, and CSV downloads."
)

st.warning(
    "Important: The model does not verify "
    "claims against live news sources. "
    "It may make incorrect predictions, "
    "especially on new topics."
)

# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <h3>🛡️ TruthLens AI</h3>
        <p>Developed by <b>Sidharth</b></p>
        <p>Python | Scikit-learn | Streamlit | Plotly</p>
        <p>© 2026 TruthLens AI Project</p>
    </div>
    """,
    unsafe_allow_html=True
)