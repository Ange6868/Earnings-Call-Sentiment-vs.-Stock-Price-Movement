import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")

st.title("Earnings Call Sentiment vs. Stock Price Movement")

df = pd.read_csv("data/master_dataset.csv")

st.write("Dataset preview:")
st.dataframe(df.head())

ticker = st.sidebar.selectbox("Select Company", sorted(df["ticker"].unique()))
company_df = df[df["ticker"] == ticker]

st.subheader(f"{ticker} Earnings Call Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Average Sentiment", round(company_df["sentiment_score"].mean(), 3))

if "CAR+5" in company_df.columns:
    col2.metric("Average CAR+5", f"{company_df['CAR+5'].mean():.2%}")
else:
    col2.metric("Average CAR+5", "N/A")

if "CAR+10" in company_df.columns:
    col3.metric("Average CAR+10", f"{company_df['CAR+10'].mean():.2%}")
else:
    col3.metric("Average CAR+10", "N/A")

tab1, tab2, tab3 = st.tabs(["Overview", "Sentiment Trend", "Return Analysis"])

with tab1:
    st.dataframe(company_df, use_container_width=True)

with tab2:
    date_col = "date" if "date" in company_df.columns else "call_date"
    fig = px.line(
        company_df.sort_values(date_col),
        x=date_col,
        y="sentiment_score",
        title="Sentiment Score Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "CAR+10" in df.columns:
        fig = px.scatter(
            df,
            x="sentiment_score",
            y="CAR+10",
            color="ticker",
            title="Sentiment Score vs. CAR+10"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("CAR+10 column is not available in the dataset.")
