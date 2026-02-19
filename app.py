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

st.markdown("""
    <style>
    .main-title {
        font-size:32px;
        font-weight:600;
        margin-bottom:10px;
    }
    .section-title {
        font-size:22px;
        margin-top:20px;
        margin-bottom:10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Data Analyzer Application</div>', unsafe_allow_html=True)

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

    # -------------------------------
    # SAFE DATA READING
    # -------------------------------
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

    # -------------------------------
    # FIX DUPLICATE COLUMNS
    # -------------------------------
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            if i != 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols

    # -------------------------------
    # SESSION STATE
    # -------------------------------
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
    # SECTION 2 — Data Overview + Cleaning
    # ==================================================
    if section == "Data Overview":

        st.markdown('<div class="section-title">Data Types</div>', unsafe_allow_html=True)
        st.write(df_cleaned.dtypes)

        st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
        st.write(df_cleaned.describe())

        # ---------------- Missing Values ----------------
        st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)

        missing = df_cleaned.isnull().sum()
        missing_percent = (missing / len(df_cleaned)) * 100

        missing_df = pd.DataFrame({
            "Column": df_cleaned.columns,
            "Missing Values": missing.values,
            "Missing %": missing_percent.values.round(2)
        })

        st.dataframe(missing_df)

        clean_option = st.selectbox(
            "Select Missing Value Handling Method",
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

        # Download cleaned data
        cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    # ==================================================
    # SECTION 3 — Visualization
    # ==================================================
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

            fig_line = px.line(df_cleaned, y=selected_col, title="Trend Analysis")
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
                    fig_scatter = px.scatter(temp_df, x=x_col, y=y_col)
                    st.plotly_chart(fig_scatter, use_container_width=True)

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

                # ---------------- Train Test Split ----------------
                split_index = int(len(df_forecast) * 0.8)
                train = df_forecast.iloc[:split_index]
                test = df_forecast.iloc[split_index:]

                model = Prophet()
                model.fit(train)

                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)

                # ---------------- Performance Calculation ----------------
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                predicted = forecast.iloc[-len(test):]["yhat"].values
                actual = test["y"].values

                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                r2 = r2_score(actual, predicted)

                st.subheader("Model Accuracy")

                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", round(mae, 2))
                col2.metric("RMSE", round(rmse, 2))
                col3.metric("R² Score", round(r2, 3))

                # ---------------- Short Performance Message ----------------
                if r2 >= 0.9:
                    message = "Your model is Excellent"
                elif r2 >= 0.75:
                    message = "Your model is Good"
                elif r2 >= 0.5:
                    message = "Your model is Moderate"
                elif r2 >= 0:
                    message = "Your model is Weak"
                else:
                    message = "Your model is Poor"

                st.success(message)

                # ---------------- Final 2 Year Forecast ----------------
                future_full = model.make_future_dataframe(periods=730)
                forecast_full = model.predict(future_full)

                fig = px.line(forecast_full, x="ds", y="yhat", title="Forecast Trend")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = model.plot_components(forecast_full)
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
        📞 6306512207 &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
        ✉ adityadav757@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)
