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

# --------------------------------------------------
# GLOBAL STYLING + ANIMATION
# --------------------------------------------------
st.markdown("""
<style>

/* Animated Welcome */
.welcome-title {
    font-size: 44px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeSlide 1.5s ease-in-out;
}

.welcome-sub {
    text-align: center;
    font-size: 18px;
    color: #CCCCCC;
    margin-bottom: 30px;
    animation: fadeSlide 2s ease-in-out;
}

@keyframes fadeSlide {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Section Title */
.section-title {
    font-size:22px;
    margin-top:20px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER (Always Visible)
# --------------------------------------------------
st.markdown('<div class="welcome-title">Welcome to Our AI Data Analyzer</div>', unsafe_allow_html=True)


st.markdown("---")

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
# IF NO FILE UPLOADED
# --------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a dataset to activate analysis features.")
else:

    # -------------------------------
    # SAFE FILE READING
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        df = pd.read_excel(uploaded_file)

    # -------------------------------
    # REMOVE DUPLICATE COLUMNS
    # -------------------------------
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
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df_cleaned.head())
        st.write("Shape:", df_cleaned.shape)
        st.write("Columns:", df_cleaned.columns.tolist())

    # ==================================================
    # SECTION 2 — Data Overview
    # ==================================================
    if section == "Data Overview":

        st.markdown('<div class="section-title">Data Types</div>', unsafe_allow_html=True)
        st.write(df_cleaned.dtypes)

        st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
        st.write(df_cleaned.describe())

        missing = df_cleaned.isnull().sum()
        missing_percent = (missing / len(df_cleaned)) * 100

        missing_df = pd.DataFrame({
            "Column": df_cleaned.columns,
            "Missing Values": missing.values,
            "Missing %": missing_percent.values.round(2)
        })

        st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
        st.dataframe(missing_df)

        clean_option = st.selectbox(
            "Missing Value Handling",
            ["Fill Mean", "Fill Median", "Fill Mode", "Drop Rows"]
        )

        if st.button("Apply Cleaning"):

            if clean_option == "Fill Mean":
                df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
            elif clean_option == "Fill Median":
                df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
            elif clean_option == "Fill Mode":
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

            fig_line = px.line(df_cleaned, y=selected_col)
            st.plotly_chart(fig_line, use_container_width=True)

            fig_hist = px.histogram(df_cleaned, x=selected_col)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==================================================
    # SECTION 4 — Forecasting
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

                split_index = int(len(df_forecast) * 0.8)
                train = df_forecast.iloc[:split_index]
                test = df_forecast.iloc[split_index:]

                freq = pd.infer_freq(train["ds"])
                if freq is None:
                    freq = "D"

                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(train)

                future = model.make_future_dataframe(periods=len(test), freq=freq)
                forecast = model.predict(future)

                predicted = forecast.iloc[-len(test):]["yhat"].values
                actual = test["y"].values

                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                r2 = r2_score(actual, predicted)

                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", round(mae, 2))
                col2.metric("RMSE", round(rmse, 2))
                col3.metric("R² Score", round(r2, 3))

                if r2 >= 0.75:
                    st.success("Model Performance: Good")
                elif r2 >= 0:
                    st.warning("Model Performance: Moderate")
                else:
                    st.error("Model Performance: Poor")

                future_full = model.make_future_dataframe(periods=365, freq=freq)
                forecast_full = model.predict(future_full)

                fig = px.line(forecast_full, x="ds", y="yhat", title="Forecast Trend")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No valid time series structure found.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Aditya Yadav | 6306512207 | adityadav757@gmail.com</div>",
    unsafe_allow_html=True
)
