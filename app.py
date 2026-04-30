import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
except Exception:
    accuracy_score = None
    balanced_accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    roc_auc_score = None


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Earnings Call Sentiment & Stock Reaction Predictor",
    page_icon="📈",
    layout="wide",
)

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "master_dataset.csv"
MODEL_DIR = APP_DIR / "models"

MODEL_PATHS = {
    "XGBoost": MODEL_DIR / "xgb_model.pkl",
    "Random Forest": MODEL_DIR / "rf_model.pkl",
    "Logistic Regression": MODEL_DIR / "lr_model.pkl",
}
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"

BASE_NUMERIC_FEATURES = [
    "pct_positive",
    "pct_negative",
    "pct_neutral",
    "sentiment_score",
    "non_neutral_rate",
    "pos_to_neg_ratio",
    "neutral_adjusted_score",
    "sentiment_score_neg2",
    "sentiment_score_neg3",
    "uncertainty_rate",
    "risk_language_rate",
    "CAR-1",
    "CAR-5",
    "CAR-10",
]

CAR_WINDOWS = ["CAR+1", "CAR+3", "CAR+5", "CAR+10"]
EVENT_WINDOWS = ["CAR-10", "CAR-5", "CAR-1", "CAR0", "CAR+1", "CAR+3", "CAR+5", "CAR+10"]

POSITIVE_WORDS = [
    "growth", "strong", "improve", "improved", "improvement", "increase", "increased",
    "record", "positive", "opportunity", "opportunities", "momentum", "demand", "profit",
    "profitable", "exceed", "exceeded", "beat", "successful", "success", "expand",
    "expanded", "expansion", "resilient", "confidence", "confident", "accelerate",
]

NEGATIVE_WORDS = [
    "decline", "declined", "decrease", "decreased", "weak", "weakness", "loss", "losses",
    "risk", "risks", "challenge", "challenging", "pressure", "headwind", "headwinds",
    "slow", "slowdown", "miss", "missed", "negative", "concern", "concerns", "uncertain",
    "uncertainty", "volatile", "volatility", "inflation", "recession", "softness",
]

UNCERTAINTY_WORDS = [
    "may", "might", "could", "approximately", "roughly", "around", "uncertain", "uncertainty",
    "potential", "possibly", "likely", "expect", "expects", "expected", "guidance", "estimate",
    "estimated", "forecast", "volatility", "volatile", "risk", "risks", "cautious", "headwind",
    "headwinds", "challenging", "unclear",
]

RISK_WORDS = [
    "risk", "risks", "challenging", "challenge", "pressure", "headwind", "headwinds", "decline",
    "weakness", "uncertain", "uncertainty", "volatility", "volatile", "slowdown", "recession",
    "inflation", "supply chain", "litigation", "competition", "regulatory", "impairment",
]


# =========================================================
# Data/model loading
# =========================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    for col in ["date", "call_date", "actual_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in BASE_NUMERIC_FEATURES + CAR_WINDOWS + EVENT_WINDOWS + ["label", "label_10d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quarter" not in df.columns:
        date_col = "date" if "date" in df.columns else "call_date"
        if date_col in df.columns:
            df["quarter"] = df[date_col].dt.to_period("Q").astype(str)

    if "sector" not in df.columns:
        df["sector"] = "Unknown"

    return df


@st.cache_resource
def load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_resource
def load_models():
    models = {name: load_pickle(path) for name, path in MODEL_PATHS.items()}
    scaler = load_pickle(SCALER_PATH)

    json_features = None
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r") as f:
                json_features = json.load(f)
        except Exception:
            json_features = None

    return models, scaler, json_features


def get_model_features(model, json_features, df: pd.DataFrame):
    """Return the exact feature names/order expected by the trained model when possible."""
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if json_features and isinstance(json_features, list):
        return json_features

    sector_dummies = []
    if "sector" in df.columns:
        sectors = sorted(df["sector"].dropna().unique())
        if len(sectors) > 1:
            sector_dummies = [f"sector_{s}" for s in sectors[1:]]

    return [f for f in BASE_NUMERIC_FEATURES if f in df.columns] + sector_dummies


# =========================================================
# Text / PDF helpers
# =========================================================
def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        if PdfReader is None:
            st.warning("PDF upload requires pypdf. Add `pypdf` to requirements.txt, or paste the transcript text instead.")
            return ""
        try:
            reader = PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
        except Exception as e:
            st.warning(f"Could not read PDF: {e}")
            return ""

    return ""


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]


def count_terms(text: str, terms: list[str]) -> int:
    text_lower = text.lower()
    total = 0
    for term in terms:
        if " " in term:
            total += text_lower.count(term)
        else:
            total += len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
    return total


def build_transcript_features(text: str) -> dict:
    """
    Lightweight transcript feature extractor for the deployed app.
    The notebook uses the original training pipeline; this app converts pasted/uploaded text
    into a compatible feature row for prediction.
    """
    sentences = split_sentences(text)
    word_tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_count = max(len(word_tokens), 1)

    sentence_labels = []
    for sentence in sentences:
        pos = count_terms(sentence, POSITIVE_WORDS)
        neg = count_terms(sentence, NEGATIVE_WORDS)
        if pos > neg:
            sentence_labels.append("positive")
        elif neg > pos:
            sentence_labels.append("negative")
        else:
            sentence_labels.append("neutral")

    if not sentence_labels:
        sentence_labels = ["neutral"]

    n = len(sentence_labels)
    pct_positive = sentence_labels.count("positive") / n
    pct_negative = sentence_labels.count("negative") / n
    pct_neutral = sentence_labels.count("neutral") / n
    sentiment_score = pct_positive - pct_negative
    non_neutral_rate = pct_positive + pct_negative
    pos_to_neg_ratio = (pct_positive + 1e-6) / (pct_negative + 1e-6)
    neutral_adjusted_score = sentiment_score / max(non_neutral_rate, 1e-6)

    uncertainty_count = count_terms(text, UNCERTAINTY_WORDS)
    risk_count = count_terms(text, RISK_WORDS)

    return {
        "pct_positive": pct_positive,
        "pct_negative": pct_negative,
        "pct_neutral": pct_neutral,
        "sentiment_score": sentiment_score,
        "non_neutral_rate": non_neutral_rate,
        "pos_to_neg_ratio": pos_to_neg_ratio,
        "neutral_adjusted_score": neutral_adjusted_score,
        "sentiment_score_neg2": sentiment_score * pct_negative,
        "sentiment_score_neg3": sentiment_score * (pct_negative ** 2),
        "uncertainty_rate": uncertainty_count / word_count,
        "risk_language_rate": risk_count / word_count,
        "word_count": word_count,
        "sentence_count": n,
    }


# =========================================================
# Market features and prediction helpers
# =========================================================
@st.cache_data(show_spinner=False)
def get_pre_event_abnormal_returns(ticker: str, call_date, lookback_days: int = 35):
    """Estimate pre-event abnormal returns using SPY as benchmark."""
    if yf is None or not ticker or pd.isna(call_date):
        return {"CAR-1": 0.0, "CAR-5": 0.0, "CAR-10": 0.0}

    call_date = pd.Timestamp(call_date)
    start = call_date - pd.Timedelta(days=lookback_days)
    end = call_date + pd.Timedelta(days=3)

    try:
        stock = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)

        if stock.empty or spy.empty:
            return {"CAR-1": 0.0, "CAR-5": 0.0, "CAR-10": 0.0}

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        common = stock.index.intersection(spy.index)
        stock = stock.loc[common]
        spy = spy.loc[common]

        base_idx = stock.index.searchsorted(call_date)
        base_idx = min(max(base_idx, 1), len(stock) - 1)

        out = {}
        for window in [1, 5, 10]:
            start_idx = max(base_idx - window, 0)
            stock_return = float(stock["Close"].iloc[base_idx] / stock["Close"].iloc[start_idx] - 1)
            spy_return = float(spy["Close"].iloc[base_idx] / spy["Close"].iloc[start_idx] - 1)
            out[f"CAR-{window}"] = stock_return - spy_return
        return out
    except Exception:
        return {"CAR-1": 0.0, "CAR-5": 0.0, "CAR-10": 0.0}


@st.cache_data(show_spinner=False)
def get_price_history(ticker: str, call_date, days_before: int = 20, days_after: int = 20):
    if yf is None or not ticker or pd.isna(call_date):
        return pd.DataFrame()

    call_date = pd.Timestamp(call_date)
    start = call_date - pd.Timedelta(days=days_before)
    end = call_date + pd.Timedelta(days=days_after)

    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.reset_index()
    except Exception:
        return pd.DataFrame()


def build_model_row(features: dict, sector: str, feature_names: list[str]) -> pd.DataFrame:
    row = {}
    for f in feature_names:
        if f in features:
            row[f] = features[f]
        elif f.startswith("sector_"):
            row[f] = 1 if f == f"sector_{sector}" else 0
        else:
            row[f] = 0
    return pd.DataFrame([row], columns=feature_names)


def predict_with_model(model_name: str, models: dict, scaler, features: dict, sector: str, feature_names: list[str]):
    model = models.get(model_name)
    if model is None:
        return None, "Model file not found."

    X = build_model_row(features, sector, feature_names)

    try:
        if model_name == "Logistic Regression" and scaler is not None:
            X_eval = scaler.transform(X)
        else:
            X_eval = X

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_eval)[0, 1])
        else:
            pred = float(model.predict(X_eval)[0])
            prob = pred
        return prob, None
    except Exception as e:
        return None, str(e)


def probability_label(prob: float):
    if prob >= 0.60:
        return "Positive expected reaction", "Lower downside risk", "The model expects a higher probability of positive post-call abnormal return."
    if prob <= 0.40:
        return "Negative expected reaction", "Higher downside risk", "The model expects a higher probability of negative post-call abnormal return."
    return "Mixed / uncertain reaction", "Medium risk", "The signal is not strong enough to clearly classify the expected post-call reaction."


def make_sentiment_gauge(score: float):
    """Gauge-style sentiment display with a marker at the score instead of a filled bar from -1."""
    score = float(np.clip(score, -1, 1))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f", "font": {"size": 58}},
            title={"text": "Sentiment Score", "font": {"size": 22}},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar": {"color": "rgba(0,0,0,0)", "thickness": 0.01},
                "steps": [
                    {"range": [-1, -0.15], "color": "#f7c9c9"},
                    {"range": [-0.15, 0.15], "color": "#fff0b3"},
                    {"range": [0.15, 1], "color": "#cfe8da"},
                ],
                "threshold": {
                    "line": {"color": "#1f2937", "width": 5},
                    "thickness": 0.9,
                    "value": score,
                },
            },
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.03,
        text="Negative  ←  Neutral  →  Positive",
        showarrow=False,
        xref="paper",
        yref="paper",
        font={"size": 13, "color": "#6b7280"},
    )

    fig.update_layout(height=310, margin=dict(l=20, r=20, t=45, b=25))
    return fig


def make_sentiment_pie(features: dict):
    fig = go.Figure(
        go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[
                features.get("pct_positive", 0),
                features.get("pct_negative", 0),
                features.get("pct_neutral", 0),
            ],
            hole=0.45,
            marker={"colors": ["#0b6fc6", "#ff4b4b", "#7fc7ff"]},
            textinfo="percent",
            sort=False,
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        ),
    )
    return fig


def format_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def prediction_card(label: str, value: str, helper: str | None = None):
    helper_html = f"<div style='font-size:0.85rem;color:#6b7280;margin-top:0.25rem;'>{helper}</div>" if helper else ""
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 20px;background:#ffffff;min-height:120px;">
            <div style="font-size:0.95rem;color:#4b5563;margin-bottom:0.55rem;">{label}</div>
            <div style="font-size:1.75rem;font-weight:650;color:#1f2937;line-height:1.15;white-space:normal;word-break:normal;">
                {value}
            </div>
            {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Load data and models
# =========================================================
df = load_data()
models, scaler, json_features = load_models()

available_model_names = [name for name, m in models.items() if m is not None]
if not available_model_names:
    available_model_names = ["XGBoost"]

reference_model = models.get("XGBoost") or models.get("Random Forest") or models.get("Logistic Regression")
model_features = get_model_features(reference_model, json_features, df)


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Controls")

if df.empty:
    ticker_options = ["AAPL", "MSFT", "NVDA"]
    sector_options = ["Technology", "Consumer Cyclical", "Healthcare", "Financial Services", "Industrials", "Energy", "Unknown"]
else:
    ticker_options = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else ["AAPL", "MSFT", "NVDA"]
    sector_options = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else ["Unknown"]

st.sidebar.markdown("### Historical Analysis Controls")
selected_ticker = st.sidebar.selectbox(
    "Company for historical view",
    ticker_options,
    help="Used as the default company in the historical charts and company tracker.",
)
selected_window = st.sidebar.selectbox(
    "Historical return window",
    [w for w in ["CAR+5", "CAR+10", "CAR+1", "CAR+3"] if df.empty or w in df.columns],
    index=0,
    help="Controls the return window shown in historical backtesting charts. This does not change the prediction target.",
)

st.sidebar.markdown("### Prediction Controls")
selected_model_name = st.sidebar.selectbox(
    "Prediction model",
    available_model_names,
    index=0,
    help="Used only when analyzing a new uploaded or pasted transcript.",
)
st.sidebar.caption("Prediction target: Positive CAR+5")

st.sidebar.markdown("---")
st.sidebar.caption("Model files")
st.sidebar.write(f"XGBoost: {'✅' if models.get('XGBoost') is not None else '—'}")
st.sidebar.write(f"Random Forest: {'✅' if models.get('Random Forest') is not None else '—'}")
st.sidebar.write(f"Logistic Regression: {'✅' if models.get('Logistic Regression') is not None else '—'}")
st.sidebar.write(f"Features loaded: {len(model_features)}")


# =========================================================
# Header
# =========================================================
st.title("📈 Earnings Call Sentiment & Stock Reaction Predictor")
st.markdown(
    "This app evaluates whether management tone in earnings call transcripts can help predict positive or negative abnormal stock returns after the call."
)

if df.empty:
    st.error("Could not find `data/master_dataset.csv`. Please check your GitHub file path.")
    st.stop()


# =========================================================
# Top KPIs
# =========================================================
call_count = len(df)
company_count = df["ticker"].nunique() if "ticker" in df.columns else 0
sector_count = df["sector"].nunique() if "sector" in df.columns else 0
positive_rate = df["label"].mean() if "label" in df.columns else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Earnings Calls", f"{call_count:,}")
k2.metric("Companies", f"{company_count:,}")
k3.metric("Sectors", f"{sector_count:,}")
k4.metric("Positive CAR+5 Rate", format_pct(positive_rate))

with st.expander("View dataset preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)


# =========================================================
# Tabs
# =========================================================
tab_home, tab_predict, tab_backtest, tab_tracker = st.tabs([
    "Home",
    "Analyze New Transcript",
    "Historical Backtesting",
    "Company Tracker",
])


# =========================================================
# Home
# =========================================================
with tab_home:
    st.subheader("Financial Analytics Problem")
    st.write(
        "Can earnings call sentiment predict abnormal stock returns in the days following the call? "
        "The app uses transcript tone features and trained machine learning models to estimate whether a call is likely to be followed by positive CAR+5."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### App workflow")
        st.markdown(
            "1. Upload or paste an earnings call transcript.\n"
            "2. Convert the transcript into sentiment, uncertainty, and risk-language features.\n"
            "3. Combine transcript signals with optional pre-call market movement.\n"
            "4. Predict the probability of a positive post-call abnormal return.\n"
            "5. Compare the prediction with historical backtesting evidence."
        )
    with c2:
        available_cols = [
            c for c in [
                "sentiment_score",
                "pct_positive",
                "pct_negative",
                "uncertainty_rate",
                "risk_language_rate",
                selected_window,
            ]
            if c in df.columns
        ]
        if available_cols:
            summary = df[available_cols].describe().T[["mean", "std", "min", "max"]]
            st.markdown("#### Historical feature summary")
            st.dataframe(summary, use_container_width=True)

    st.info(
        "For the real-time analyzer, PDF upload is supported, but pasting transcript text is usually the most stable option. "
        "The original training pipeline used PDF transcripts; the deployed app only needs the extracted transcript text."
    )


# =========================================================
# Analyze New Transcript
# =========================================================
with tab_predict:
    st.subheader("Analyze a New Earnings Call Transcript")
    st.caption("Use this page as the actual prediction tool: paste text or upload a PDF/TXT transcript.")

    input_col, setting_col = st.columns([1.3, 0.7])

    with setting_col:
        user_ticker = st.text_input("Ticker", value=selected_ticker)
        user_sector = st.selectbox(
            "Sector",
            sector_options,
            index=sector_options.index("Technology") if "Technology" in sector_options else 0,
        )
        call_date = st.date_input("Earnings call date")
        use_market_features = st.checkbox(
            "Estimate pre-call market features with yfinance",
            value=True,
            help="If checked, the app estimates CAR-1, CAR-5, and CAR-10 automatically using yfinance. If unchecked, the app uses the manual values below.",
        )
        st.caption(
            "Checked: auto-estimate CAR-1/CAR-5/CAR-10 from ticker and call date.  "
            "Unchecked: manually enter pre-call CAR values."
        )

        if not use_market_features:
            manual_car1 = st.number_input("Manual CAR-1", value=0.0, step=0.01, format="%.4f")
            manual_car5 = st.number_input("Manual CAR-5", value=0.0, step=0.01, format="%.4f")
            manual_car10 = st.number_input("Manual CAR-10", value=0.0, step=0.01, format="%.4f")
        else:
            manual_car1 = 0.0
            manual_car5 = 0.0
            manual_car10 = 0.0

    with input_col:
        uploaded = st.file_uploader("Upload transcript file (PDF or TXT)", type=["pdf", "txt"])
        uploaded_text = extract_text_from_upload(uploaded)

        transcript_text = st.text_area(
            "Or paste transcript text here",
            value=uploaded_text,
            height=320,
            placeholder="Paste CEO/CFO prepared remarks or the full earnings call transcript...",
        )

    analyze = st.button("Run Sentiment Analysis & Prediction", type="primary")

    if analyze:
        if not transcript_text.strip():
            st.error("Please upload a transcript file or paste transcript text first.")
        else:
            with st.spinner("Extracting features and running model..."):
                transcript_features = build_transcript_features(transcript_text)

                if use_market_features:
                    market_features = get_pre_event_abnormal_returns(user_ticker, pd.Timestamp(call_date))
                else:
                    market_features = {"CAR-1": manual_car1, "CAR-5": manual_car5, "CAR-10": manual_car10}

                full_features = {**transcript_features, **market_features}
                prob, err = predict_with_model(
                    selected_model_name,
                    models,
                    scaler,
                    full_features,
                    user_sector,
                    model_features,
                )

            st.markdown("### Transcript Sentiment Signals")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Positive", f"{full_features['pct_positive']:.1%}")
            m2.metric("Negative", f"{full_features['pct_negative']:.1%}")
            m3.metric("Neutral", f"{full_features['pct_neutral']:.1%}")
            m4.metric("Uncertainty Rate", f"{full_features['uncertainty_rate']:.2%}")

            chart1, chart2 = st.columns(2)
            with chart1:
                st.plotly_chart(make_sentiment_gauge(full_features["sentiment_score"]), use_container_width=True)
            with chart2:
                st.plotly_chart(make_sentiment_pie(full_features), use_container_width=True)

            st.markdown("### Model Prediction")
            if err:
                st.error(f"Prediction failed: {err}")
                st.caption("This usually means features.json does not match the trained model. Re-export model features from the notebook if needed.")
            else:
                signal, risk, explanation = probability_label(prob)
                p1, p2, p3 = st.columns(3)
                with p1:
                    prediction_card("Probability of Positive CAR+5", f"{prob:.1%}", "Prediction target")
                with p2:
                    prediction_card("Expected Reaction", signal, "Direction of expected post-call reaction")
                with p3:
                    prediction_card("Risk Level", risk, "Based on model confidence")
                st.write(explanation)
                st.caption(
                    f"Model: {selected_model_name}. Prediction is based on patterns learned from the historical earnings call sample."
                )

            with st.expander("View generated model features"):
                feature_df = pd.DataFrame([full_features]).T.reset_index()
                feature_df.columns = ["feature", "value"]
                st.dataframe(feature_df, use_container_width=True)


# =========================================================
# Historical Backtesting
# =========================================================
with tab_backtest:
    st.subheader("Historical Backtesting Evidence")

    sector_filter = st.selectbox("Filter by sector", ["All"] + sector_options, key="backtest_sector")
    filtered = df.copy()
    if sector_filter != "All" and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"] == sector_filter]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Observations", f"{len(filtered):,}")
    b2.metric(f"Average {selected_window}", format_pct(filtered[selected_window].mean()) if selected_window in filtered.columns else "N/A")
    b3.metric("Average Sentiment", f"{filtered['sentiment_score'].mean():.3f}" if "sentiment_score" in filtered.columns else "N/A")
    b4.metric("Positive CAR+5 Rate", format_pct(filtered["label"].mean()) if "label" in filtered.columns else "N/A")

    if "sentiment_score" in filtered.columns and selected_window in filtered.columns:
        fig = px.scatter(
            filtered,
            x="sentiment_score",
            y=selected_window,
            color="sector" if "sector" in filtered.columns else None,
            hover_data=[c for c in ["ticker", "date", "actual_date", "sector"] if c in filtered.columns],
            trendline="ols",
            title=f"Sentiment Score vs. {selected_window}",
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    available_windows = [w for w in EVENT_WINDOWS if w in filtered.columns]
    if available_windows:
        avg_car = filtered[available_windows].mean().reset_index()
        avg_car.columns = ["Event Window", "Average CAR"]
        fig_bar = px.bar(avg_car, x="Event Window", y="Average CAR", title="Average CAR Around Earnings Call Date")
        fig_bar.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Model Performance on Historical Sample")
    if "label" in filtered.columns and len(filtered) > 0:
        perf_rows = []
        for name, model in models.items():
            if model is None:
                continue

            features_for_model = get_model_features(model, json_features, df)
            rows = []
            for _, row in filtered.iterrows():
                rows.append(build_model_row(row.to_dict(), row.get("sector", "Unknown"), features_for_model))
            X_eval = pd.concat(rows, ignore_index=True)
            y_true = filtered["label"].astype(int)

            try:
                if name == "Logistic Regression" and scaler is not None:
                    X_model = scaler.transform(X_eval)
                else:
                    X_model = X_eval

                y_prob = model.predict_proba(X_model)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                perf_rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_true, y_pred) if accuracy_score else np.nan,
                    "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred) if balanced_accuracy_score else np.nan,
                    "Precision": precision_score(y_true, y_pred, zero_division=0) if precision_score else np.nan,
                    "Recall": recall_score(y_true, y_pred, zero_division=0) if recall_score else np.nan,
                    "F1": f1_score(y_true, y_pred, zero_division=0) if f1_score else np.nan,
                    "ROC-AUC": roc_auc_score(y_true, y_prob) if roc_auc_score and len(set(y_true)) > 1 else np.nan,
                })
            except Exception:
                continue

        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            st.dataframe(
                perf_df.style.format({c: "{:.3f}" for c in perf_df.columns if c != "Model"}),
                use_container_width=True,
            )
            st.caption("Displayed metrics are calculated on the available historical app dataset and are mainly for dashboard interpretation.")
        else:
            st.info("Model performance could not be calculated. Check whether features.json matches the saved models.")

    xgb_model = models.get("XGBoost")
    if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
        feat_names = get_model_features(xgb_model, json_features, df)
        importance = pd.DataFrame({"feature": feat_names, "importance": xgb_model.feature_importances_})
        importance = importance.sort_values("importance", ascending=False).head(12)
        fig_imp = px.bar(importance, x="importance", y="feature", orientation="h", title="XGBoost Feature Importance")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# Company Tracker
# =========================================================
with tab_tracker:
    st.subheader("Multi-Company Sentiment Tracker")

    tracker_tickers = st.multiselect(
        "Select companies",
        ticker_options,
        default=[selected_ticker] if selected_ticker in ticker_options else ticker_options[:1],
    )

    tracker = df[df["ticker"].isin(tracker_tickers)].copy() if "ticker" in df.columns else df.copy()
    date_col = "date" if "date" in tracker.columns else "call_date"

    if not tracker.empty and date_col in tracker.columns and "sentiment_score" in tracker.columns:
        fig_trend = px.line(
            tracker.sort_values(date_col),
            x=date_col,
            y="sentiment_score",
            color="ticker",
            markers=True,
            title="Sentiment Score Over Time",
        )
        fig_trend.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_trend, use_container_width=True)

    heatmap_options = [
        c for c in [
            "sentiment_score",
            "pct_positive",
            "pct_negative",
            "uncertainty_rate",
            "risk_language_rate",
            "CAR+5",
            "CAR+10",
        ]
        if c in tracker.columns
    ]

    if heatmap_options:
        heatmap_metric = st.selectbox("Heatmap metric", heatmap_options)
        if "quarter" in tracker.columns:
            pivot = tracker.pivot_table(index="ticker", columns="quarter", values=heatmap_metric, aggfunc="mean")
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                title=f"Quarterly Heatmap: {heatmap_metric}",
                labels={"x": "Quarter", "y": "Ticker", "color": heatmap_metric},
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("View selected company data"):
        st.dataframe(tracker, use_container_width=True)


st.markdown("---")
st.caption(
    "BA870/AC820 Financial Analytics Streamlit prototype. Training data and model artifacts are produced in the project notebook; this app loads the exported dataset and model files from GitHub."
)
