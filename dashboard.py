import streamlit as st
import plotly.express as px
from data_utils import load_data

def app():
    st.title("📈 Insights Dashboard")
    st.caption("High-level overview of air quality patterns across cities and time.")

    df = load_data()

    # Top KPIs
    col1, col2, col3 = st.columns(3)
    if "AQI" in df.columns:
        col1.metric("Average AQI", f"{df['AQI'].mean():.1f}")
        col2.metric("Max AQI", f"{df['AQI'].max():.0f}")
        col3.metric("Min AQI", f"{df['AQI'].min():.0f}")

    st.markdown("---")

    # City + time filters
    if "City" in df.columns and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        cities = ["All"] + sorted(df["City"].unique().tolist())
        city = st.selectbox("Select City:", cities)

        if city != "All":
            df_plot = df[df["City"] == city].copy()
        else:
            df_plot = df.copy()

        if "AQI" in df_plot.columns:
            fig = px.line(
                df_plot.sort_values("Date"),
                x="Date", y="AQI", color="City" if city == "All" else None,
                title="AQI Trend Over Time"
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if "AQI_Bucket" in df.columns and "City" in df.columns:
        st.markdown("### 🌈 AQI Category Distribution by City")
        fig = px.histogram(
            df, x="City", color="AQI_Bucket", barmode="group",
            title="AQI Category Counts per City"
        )
        fig.update_layout(xaxis_tickangle=-45, height=500, margin=dict(l=20, r=20, t=40, b=100))
        st.plotly_chart(fig, use_container_width=True)
