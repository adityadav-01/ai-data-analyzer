import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("AI Data Analyzer Application")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Upload Data", "Data Overview", "Visualization", "Forecasting"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if uploaded_file is not None:

    # --------------------------------------------------
    # SAFE DATA READING (Handles Encoding Errors)
    # --------------------------------------------------
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding="latin1")
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file)

    # --------------------------------------------------
    # FIX DUPLICATE COLUMN NAMES
    # --------------------------------------------------
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            if i != 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols

    # --------------------------------------------------
    # SESSION STATE FOR CLEANED DATA
    # --------------------------------------------------
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = df.copy()

    df_cleaned = st.session_state.cleaned_df

    # --------------------------------------------------
    # SECTION 1 — Upload
    # --------------------------------------------------
    if section == "Upload Data":
        st.subheader("Dataset Preview")
        st.dataframe(df_cleaned.head())
        st.write("Shape:", df_cleaned.shape)
        st.write("Columns:", df_cleaned.columns.tolist())

    # --------------------------------------------------
    # SECTION 2 — Overview
    # --------------------------------------------------
    if section == "Data Overview":

        st.subheader("Data Types")
        st.write(df_cleaned.dtypes)

        st.subheader("Statistical Summary")
        st.write(df_cleaned.describe())

        st.subheader("Missing Values")
        st.dataframe(
            df_cleaned.isnull().sum().reset_index()
            .rename(columns={"index": "Column", 0: "Missing Values"})
        )

        if st.button("Clean Missing Values"):
            df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
            df_cleaned = df_cleaned.fillna("Unknown")
            st.session_state.cleaned_df = df_cleaned
            st.success("Missing values cleaned successfully")

        cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    # --------------------------------------------------
    # SECTION 3 — Visualization
    # --------------------------------------------------
    if section == "Visualization":

        numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox("Select Numeric Column", numeric_cols)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", round(df_cleaned[selected_col].mean(), 2))
            col2.metric("Max", round(df_cleaned[selected_col].max(), 2))
            col3.metric("Min", round(df_cleaned[selected_col].min(), 2))
            col4.metric("Std Dev", round(df_cleaned[selected_col].std(), 2))

            fig_line = px.line(df_cleaned, y=selected_col, title="Trend")
            st.plotly_chart(fig_line, use_container_width=True)

            fig_hist = px.histogram(df_cleaned, x=selected_col)
            st.plotly_chart(fig_hist, use_container_width=True)

        if len(numeric_cols) > 1:

            st.subheader("Scatter Plot")

            x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y")

            if x_col != y_col:
                temp_df = df_cleaned[[x_col, y_col]].dropna()

                if len(temp_df) > 0:
                    fig_scatter = px.scatter(
                        temp_df,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col} vs {y_col}"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Not enough valid data for scatter plot.")
            else:
                st.info("Select two different columns.")

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = df_cleaned[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
            st.pyplot(fig)

        if len(categorical_cols) > 0:
            st.subheader("Categorical Distribution")
            cat_col = st.selectbox("Categorical Column", categorical_cols)
            fig_pie = px.pie(df_cleaned, names=cat_col)
            st.plotly_chart(fig_pie, use_container_width=True)

    # --------------------------------------------------
    # SECTION 4 — Forecasting
    # --------------------------------------------------
    if section == "Forecasting":

        numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns

        date_cols = []
        for col in df_cleaned.columns:
            try:
                pd.to_datetime(df_cleaned[col])
                date_cols.append(col)
            except:
                continue

        if len(date_cols) > 0 and len(numeric_cols) > 0:

            st.subheader("Time Series Forecasting")

            date_col = st.selectbox("Date Column", date_cols)
            target_col = st.selectbox("Target Column", numeric_cols)

            if st.button("Generate 2 Year Forecast"):

                df_forecast = df_cleaned[[date_col, target_col]].copy()
                df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors="coerce")
                df_forecast = df_forecast.dropna()

                df_forecast = df_forecast.rename(columns={
                    date_col: "ds",
                    target_col: "y"
                }).sort_values("ds")

                model = Prophet()
                model.fit(df_forecast)

                future = model.make_future_dataframe(periods=730)
                forecast = model.predict(future)

                fig = px.line(forecast, x="ds", y="yhat", title="Forecast Trend")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Trend and Seasonality")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                forecast_csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                csv = forecast_csv.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name="forecast_results.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No valid time series structure found.")

else:
    st.info("Upload dataset to begin.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("Website made by Aditya Yadav")
st.markdown("Contact Number: 6306512207")
st.markdown("Email: adityadav757@gmail.com")
