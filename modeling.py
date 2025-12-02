import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from data_utils import load_data

REG_FEATURES = [
    "AQI_Lag1", "AQI_Lag7",
    "PM2.5", "PM10", "CO", "NO2", "O3", "SO2", "NO",
    "Region_West", "Region_East", "Region_North",
    "Region_Northeast", "Region_South",
    "Season_Spring", "Month", "Is_Weekend"
]

CLASS_FEATURES = REG_FEATURES  # same set, predicting AQI_Bucket


def app():
    st.title("🤖 Modeling & Evaluation")
    st.caption("Multiple Linear Regression for AQI and Decision Tree for AQI category.")

    df = load_data()

    task = st.radio("Select Task:", ["AQI Regression (Linear Regression)", "AQI Category Classification (Decision Tree)"])

    if task.startswith("AQI Regression"):
        run_regression(df)
    else:
        run_classification(df)


def run_regression(df: pd.DataFrame):
    st.markdown("### 📈 Multiple Linear Regression — Predicting AQI")

    cols = [c for c in REG_FEATURES if c in df.columns]
    if "AQI" not in df.columns or len(cols) == 0:
        st.error("Required columns for regression are missing. Please verify preprocessing.")
        return

    model_df = df.dropna(subset=cols + ["AQI"]).copy()
    X = model_df[cols]
    y = model_df["AQI"]

    test_size = st.slider("Test set size (%):", 10, 40, 20, step=5) / 100.0
    random_state = st.number_input("Random state:", min_value=0, value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R²", f"{r2:.4f}")

    st.markdown("These should be close to the values you reported in the notebook (MAE ≈ 22.79, RMSE ≈ 46.11, R² ≈ 0.889).")

    # Plot actual vs predicted
    import plotly.express as px

    res_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": y_pred})
    fig = px.scatter(
        res_df, x="Actual AQI", y="Predicted AQI",
        title="Actual vs Predicted AQI",
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show coefficients
    coef_df = pd.DataFrame({"Feature": cols, "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
    st.markdown("### 🔍 Feature Influence (Coefficients)")
    st.dataframe(coef_df)

def run_classification(df: pd.DataFrame):
    st.markdown("### 🌈 Decision Tree Classification — AQI Buckets")

    if "AQI_Bucket" not in df.columns:
        st.error("Column 'AQI_Bucket' not found. Please ensure classification labels are prepared.")
        return

    cols = [c for c in CLASS_FEATURES if c in df.columns]
    model_df = df.dropna(subset=cols + ["AQI_Bucket"]).copy()

    X = model_df[cols]
    y = model_df["AQI_Bucket"]

    test_size = st.slider("Test set size (%):", 10, 40, 20, step=5, key="cls_size") / 100.0
    random_state = st.number_input("Random state (classifier):", min_value=0, value=42, step=1, key="cls_rs")
    max_depth = st.slider("Max tree depth (0 = None):", 0, 20, 8, step=1)

    if max_depth == 0:
        max_depth = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=int(random_state))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")

    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df)

    st.markdown("These values should be close to your notebook’s results (overall accuracy ≈ 0.783, class-wise F1 scores).")
