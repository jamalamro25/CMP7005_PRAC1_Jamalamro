import streamlit as st
from data_utils import load_data

def app():
    st.title("🧹 Data Preprocessing")
    st.caption("Overview of cleaning steps and engineered features used in modelling.")

    df = load_data()

    st.markdown("### 🧼 Cleaning Summary")
    st.markdown(
        """
        **Key preprocessing steps applied in the notebook (mirrored here conceptually):**
        - Handled missing values using forward fill / interpolation where appropriate  
        - Removed obvious outliers based on domain thresholds  
        - Converted `Date` to proper datetime and extracted **Year, Month, Day, Season**  
        - Created **lag features**: `AQI_Lag1`, `AQI_Lag7`  
        - Created **region dummies**: `Region_West`, `Region_East`, ...  
        - Encoded **AQI_Bucket** for classification  
        """
    )

    st.markdown("### 🧬 Engineered Feature Columns (Preview)")
    model_features = [
        "AQI", "AQI_Lag1", "AQI_Lag7",
        "PM2.5", "PM10", "CO", "NO2", "O3", "SO2", "NO",
        "Region_West", "Region_East", "Region_North",
        "Region_Northeast", "Region_South",
        "Season_Spring", "Month", "Is_Weekend", "AQI_Bucket"
    ]

    existing = [c for c in model_features if c in df.columns]
    st.write(f"Total engineered / modelling features found: **{len(existing)}**")
    st.dataframe(df[existing].head(10))

    st.markdown("### 🔎 Data Types Check")
    st.dataframe(df[existing].dtypes.to_frame("dtype"))
