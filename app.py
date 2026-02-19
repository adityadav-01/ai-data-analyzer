import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    # ---------------- SAFE DATA READING ----------------
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        df = pd.read_excel(uploaded_file)

    # ---------------- FIX DUPLICATE COLUMNS ----------------
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            if i != 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = df.copy()

    df_cleaned = st.session_state.cleaned_df

    # ==================================================
    # SECTION 1 — Upload
    # ==================================================
    if section == "Upload Data":
        st.subheader("Dataset Preview")
        st.dataframe(df_cleaned.head())
        st.write("Shape:", df_cleaned.shape)

    # ==================================================
    # SECTION 2 — Data Overview + Cleaning
    # ==================================================
    if section == "Data Overview":

        st.subheader("Statistical Summary")
        st.write(df_cleaned.describe())

        st.subheader("Missing Values")

        missing = df_cleaned.isnull().sum()
        missing_percent = (missing / len(df_cleaned)) * 100

        missing_df = pd.DataFrame({
            "Column": df_cleaned.columns,
            "Missing Values": missing.values,
            "Missing %": missing_percent.values.round(2)
        })

        st.dataframe(missing_df)

        clean_option = st.selectbox(
            "Missing Value Handling",
            ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"]
        )

        if st.button("Apply Cleaning"):

            if clean_option == "Fill with Mean":
                df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
                df_cleaned = df_cleaned.fillna("Unknown")

            elif clean_option == "Fill with Median":
                df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
                df_cleaned = df_cleaned.fillna("Unknown")

            elif clean_option == "Fill with Mode":
                for col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

            elif clean_option == "Drop Rows":
                df_cleaned = df_cleaned.dropna()

            st.session_state.cleaned_df = df_cleaned
            st.success("Cleaning Applied Successfully")

    # ==================================================
    # SECTION 3 — Visualization
    # ==================================================
    if section == "Visualization":

        numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Numeric Column", numeric_cols)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", round(df_cleaned[selected_col].mean(), 2))
            col2.metric("Max", round(df_cleaned[selected_col].max(), 2))
            col3.metric("Min", round(df_cleaned[selected_col].min(), 2))
            col4.metric("Std Dev", round(df_cleaned[selected_col].std(), 2))

            fig_line = px.line(df_cleaned, y=selected_col, title="Trend")
            st.plotly_chart(fig_line, use_container_width=True)

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = df_cleaned[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
            st.pyplot(fig)

    # ==================================================
    # SECTION 4 — Forecasting + Accuracy
    # ==================================================
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

            if st.button("Generate Forecast"):

                df_forecast = df_cleaned[[date_col, target_col]].copy()
                df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors="coerce")
                df_forecast = df_forecast.dropna()

                df_forecast = df_forecast.rename(columns={
                    date_col: "ds",
                    target_col: "y"
                }).sort_values("ds")

                # ---------------- Train Test Split ----------------
                split_index = int(len(df_forecast) * 0.8)
                train = df_forecast.iloc[:split_index]
                test = df_forecast.iloc[split_index:]

                model = Prophet()
                model.fit(train)

                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)

                predicted = forecast.iloc[-len(test):]["yhat"].values
                actual = test["y"].values

                # ---------------- Metrics ----------------
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100

                # R2 Calculation
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                st.subheader("Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", round(mae, 2))
                col2.metric("RMSE", round(rmse, 2))
                col3.metric("MAPE (%)", round(mape, 2))
                col4.metric("R² Score", round(r2, 3))

                # ---------------- Model Rating ----------------
                if r2 >= 0.9:
                    rating = "Excellent Model"
                elif r2 >= 0.75:
                    rating = "Good Model"
                elif r2 >= 0.5:
                    rating = "Moderate Model"
                elif r2 >= 0:
                    rating = "Weak Model"
                else:
                    rating = "Poor Model"

                st.success(f"Model Quality: {rating}")

                # ---------------- Forecast Plot ----------------
                fig = px.line(
                    forecast,
                    x="ds",
                    y=["yhat", "yhat_lower", "yhat_upper"],
                    title="Forecast with Confidence Interval"
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

        else:
            st.warning("No valid time series structure found.")


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; font-size:15px; padding:10px;'>
        <b>Aditya Yadav</b> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
        6306512207 &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
        adityadav757@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)
