import streamlit as st
from multiapp import MultiApp

import home
import data_loading
import data_preprocessing
import data_visualization
import modeling
import dashboard

# Global page configuration (only here)
st.set_page_config(
    page_title="India Air Quality Intelligence",
    page_icon="🌫️",
    layout="wide"
)

app = MultiApp()

app.add_app("🏠 Home", home.app)
app.add_app("📂 Data Loading", data_loading.app)
app.add_app("🧹 Data Preprocessing", data_preprocessing.app)
app.add_app("📊 Data Visualization", data_visualization.app)
app.add_app("🤖 Modeling & Evaluation", modeling.app)
app.add_app("📈 Insights Dashboard", dashboard.app)

app.run()
