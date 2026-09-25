import streamlit as st

st.set_page_config(
    page_title="Seasonal Crop Prediction",
    page_icon="🌱",
    layout="wide"
)

# SIDEBAR
with st.sidebar:
    st.title("🌱 Crop Prediction")
    st.markdown("---")

    st.subheader("System Status")
    st.success("ML Model Ready")
    st.error("ESP32 Disconnected")

    st.markdown("---")
    st.info(
        """
        **Project:**
        Seasonal Crop Prediction
         
        **Input:**
        Soil Parameters
        
        **Output:**
        Recommended Crop
        """
    )

# MAIN HEADER
st.title("🌱 Seasonal Crop Prediction System")
st.caption("Real-Time Soil Analysis & Crop Recommendation")

st.markdown("---")

# SENSOR SECTION
st.subheader("Soil Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💧 Moisture", "--")

with col2:
    st.metric("🧪 pH", "--")

with col3:
    st.metric("🌿 Nitrogen", "--")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("🌾 Phosphorus", "--")

with col5:
    st.metric("🍃 Potassium", "--")

with col6:
    st.metric("💦 WHC", "--")

st.markdown("---")

# SEASON
season = st.selectbox(
    "Select Season",
    ["Summer", "Winter", "Monsoon"]
)

st.markdown("---")

# PREDICTION AREA
st.subheader("Crop Recommendation")

if st.button("Predict Crop"):
    st.warning("Waiting for sensor data...")

st.info("Prediction output will appear here.")

st.markdown("---")

# CHART SECTION
st.subheader("NPK Analysis")

st.bar_chart({
    "NPK": [0, 0, 0]
})