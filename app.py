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
st.markdown('<div class="welcome-title"> AI Data Analyzer</div>', unsafe_allow_html=True)
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
    # SECTION 2 — Data Overview + Cleaning
    # ==================================================
    elif section == "Data Overview":

        st.info(f"Current Dataset Shape: {df_cleaned.shape}")

        st.markdown('<div class="section-title">Current Data Overview</div>', unsafe_allow_html=True)

        st.subheader("Data Types")
        st.write(df_cleaned.dtypes)

        st.subheader("Statistical Summary")
        st.write(df_cleaned.describe())

        missing = df_cleaned.isnull().sum()
        missing_percent = (missing / len(df_cleaned)) * 100

        missing_df = pd.DataFrame({
            "Column": df_cleaned.columns,
            "Missing Values": missing.values,
            "Missing %": missing_percent.values.round(2)
        })

        st.subheader("Missing Values")
        st.dataframe(missing_df)

        st.markdown("---")
        st.subheader("Apply Data Cleaning")

        clean_option = st.selectbox(
            "Select Missing Value Handling Method",
            ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"]
        )

        if st.button("Apply Cleaning"):

            temp_df = df_cleaned.copy()

            if clean_option == "Fill with Mean":
                temp_df = temp_df.fillna(temp_df.mean(numeric_only=True))
            elif clean_option == "Fill with Median":
                temp_df = temp_df.fillna(temp_df.median(numeric_only=True))
            elif clean_option == "Fill with Mode":
                for col in temp_df.columns:
                    temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])
            elif clean_option == "Drop Rows":
                temp_df = temp_df.dropna()

            st.session_state.cleaned_df = temp_df
            df_cleaned = temp_df

            st.success("Cleaning Applied Successfully!")

            st.markdown("## Updated Cleaned Data Overview")

            st.subheader("Updated Data Types")
            st.write(df_cleaned.dtypes)

            st.subheader("Updated Statistical Summary")
            st.write(df_cleaned.describe())

            updated_missing = df_cleaned.isnull().sum()
            updated_missing_percent = (updated_missing / len(df_cleaned)) * 100

            updated_missing_df = pd.DataFrame({
                "Column": df_cleaned.columns,
                "Missing Values": updated_missing.values,
                "Missing %": updated_missing_percent.values.round(2)
            })

            st.subheader("Updated Missing Values")
            st.dataframe(updated_missing_df)

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
    elif section == "Visualization":

        st.markdown("## Advanced Data Visualization")

        numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns

        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for visualization.")
        else:

            st.subheader("1️ Line Chart (Trend)")
            line_col = st.selectbox("Select Column for Line Chart", numeric_cols, key="line")
            fig_line = px.line(df_cleaned, y=line_col)
            st.plotly_chart(fig_line, use_container_width=True)

            if len(categorical_cols) > 0:
                st.subheader("2️ Bar Chart (Category vs Value)")
                cat_col = st.selectbox("Select Category Column", categorical_cols, key="bar_cat")
                num_col = st.selectbox("Select Value Column", numeric_cols, key="bar_num")
                grouped = df_cleaned.groupby(cat_col)[num_col].mean().reset_index()
                fig_bar = px.bar(grouped, x=cat_col, y=num_col)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("3️ Histogram (Distribution)")
            hist_col = st.selectbox("Select Column for Histogram", numeric_cols, key="hist")
            fig_hist = px.histogram(df_cleaned, x=hist_col)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("4️ Box Plot (Outlier Detection)")
            box_col = st.selectbox("Select Column for Box Plot", numeric_cols, key="box")
            fig_box = px.box(df_cleaned, y=box_col)
            st.plotly_chart(fig_box, use_container_width=True)

            if len(numeric_cols) > 1:
                st.subheader("5️ Scatter Plot (Relationship)")
                x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y")
                if x_col != y_col:
                    fig_scatter = px.scatter(df_cleaned, x=x_col, y=y_col)
                    st.plotly_chart(fig_scatter, use_container_width=True)

            if len(numeric_cols) > 1:
                st.subheader("6️ Correlation Heatmap")
                corr = df_cleaned[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
                st.pyplot(fig)

            if len(categorical_cols) > 0:
                st.subheader("7️ Pie Chart (Composition)")
                pie_col = st.selectbox("Select Column for Pie Chart", categorical_cols, key="pie")
                fig_pie = px.pie(df_cleaned, names=pie_col)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("8️ Area Chart")
            area_col = st.selectbox("Select Column for Area Chart", numeric_cols, key="area")
            fig_area = px.area(df_cleaned, y=area_col)
            st.plotly_chart(fig_area, use_container_width=True)

            st.subheader("9️ Moving Average (Trend Smoothing)")
            ma_col = st.selectbox("Select Column for Moving Average", numeric_cols, key="ma")
            window = st.slider("Select Moving Average Window", 2, 30, 7)
            df_ma = df_cleaned.copy()
            df_ma["Moving Average"] = df_ma[ma_col].rolling(window=window).mean()
            fig_ma = px.line(df_ma, y=[ma_col, "Moving Average"])
            st.plotly_chart(fig_ma, use_container_width=True)

            if len(categorical_cols) > 0:
                st.subheader("10 Treemap (Hierarchical View)")
                tree_cat = st.selectbox("Select Category for Treemap", categorical_cols, key="tree_cat")
                tree_val = st.selectbox("Select Value for Treemap", numeric_cols, key="tree_val")
                grouped_tree = df_cleaned.groupby(tree_cat)[tree_val].sum().reset_index()
                fig_tree = px.treemap(grouped_tree, path=[tree_cat], values=tree_val)
                st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("---")
        st.markdown("## 💬 Visualization Doubt Chat")

        user_doubt = st.text_area("Ask your visualization doubt here:")

        if st.button("Get Answer"):
            if user_doubt.strip() == "":
                st.warning("Please enter your question.")
            else:
                st.success("Here is guidance:")
                st.write("""
• Use Line Chart for trends over time  
• Use Bar Chart for comparing categories  
• Use Histogram for distribution  
• Use Box Plot for outlier detection  
• Use Scatter Plot for relationship between variables  
• Use Heatmap for correlation analysis  
• Use Pie Chart for proportions  
• Use Moving Average for smoothing noisy data  
• Use Treemap for hierarchical category breakdown  
""")

    # ==================================================
    # SECTION 4 — Forecasting
    # ==================================================
    elif section == "Forecasting":

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

                df_forecast = df_forecast.rename(columns={date_col: "ds", target_col: "y"}).sort_values("ds")

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
