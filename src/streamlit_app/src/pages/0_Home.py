import streamlit as st

st.set_page_config(
    page_title="Rent Price Prediction in India",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Rent Price Prediction in India")
st.markdown("---")

# Introducción
st.subheader("📌 Context")
st.markdown("""
India's rental housing market is highly diverse, ranging from historic palaces to rural homes and modern city apartments.  
Access to adequate housing remains a challenge, with just **60.9%** of the population living in acceptable conditions.

This project leverages **Machine Learning** to help tenants and property owners make informed decisions about rental pricing.
""")

# Objetivo general
st.subheader("🎯 Project Goal")
st.markdown("""
To develop a predictive model using the **House Rent Prediction Dataset** (Kaggle) and integrate it into an interactive dashboard that enables:
- Rental price forecasting based on property features.
- Visualization of market trends across Indian cities.
""")

# Funcionalidades
st.subheader("🧰 Key Features")
st.markdown("""
- **Exploratory Data Analysis (EDA)**: Understand relationships and trends.
- **Modeling**: Train and evaluate models (Linear Regression, Random Forest, XGBoost).
- **Dashboard**: Real-time prediction with a friendly interface.
""")

# Tecnologías
st.subheader("🛠️ Tech Stack")
st.markdown("""
- **Backend**: Python (Pandas, Polars, Scikit-learn, XGBoost, etc.)
- **Frontend**: Streamlit
- **Visualization**: Seaborn, Matplotlib
""")

# Footer
st.markdown("---")
st.markdown(
    "📘 Developed by Nicolás J. Pietrocola | TFM – Master in Big Data & AI, Universidad de Málaga"
)
