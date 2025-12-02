import streamlit as st
from data_utils import load_data

def app():
    df = load_data()

    st.markdown(
        """
        <h1 style="color:#0f4c75;">India Air Quality Intelligence 🌫️</h1>
        <p style="font-size:1.05rem;">
        An interactive end-to-end platform for exploring, modelling, and visualizing India's air quality
        between 2015–2020. Built as part of the CMP7005 Programming for Data Analysis module.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔍 Project Goals")
    st.markdown(
        """
        - Monitor and analyse pollution patterns across Indian cities  
        - Predict AQI using **Multiple Linear Regression**  
        - Classify AQI categories using a **Decision Tree classifier**  
        - Provide an interactive dashboard for stakeholders and decision makers  
        """
    )

    # KPI cards
    st.markdown("### 📌 Quick Snapshot")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Cities", df["City"].nunique() if "City" in df.columns else "N/A")
    with col3:
        st.metric("Date Range", f"{df['Date'].min()} → {df['Date'].max()}" if "Date" in df.columns else "N/A")
    with col4:
        st.metric("Target Variable", "AQI & AQI Category")

    st.markdown("---")
    st.info(
        "Use the sidebar navigation to explore data loading, preprocessing, visual insights, "
        "and the modelling workflow."
    )
