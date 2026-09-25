import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Seasonal Crop Prediction",
    page_icon="🌱",
    layout="wide"
)

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# SIDEBAR
with st.sidebar:
    st.title("Crop Prediction")
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
st.markdown("""
<div class="hero">

<div class="hero-title">
Seasonal Crop Prediction System
</div>

<div class="hero-subtitle">
Smart Agriculture using IoT & Machine Learning
</div>

</div>
""", unsafe_allow_html=True)

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
    st.metric("💧 WHC", "--")

st.markdown("---")

# SEASON
season = st.selectbox(
    "Select Season",
    ["Summer", "Winter", "Monsoon"]
)

st.markdown("---")

# PREDICTION AREA

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("🟫 Soil Type")
    st.info("Waiting for soil classification...")

with col2:
    st.subheader("🌾 Recommended Crop")
    st.success("Prediction will appear here after clicking Predict Crop.")

with col3:
    st.subheader(" ")

    predict = st.button("Predict Crop")

    if predict:
        st.success("Prediction Started...")

# CHART SECTION
st.subheader("NPK Analysis")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=["Nitrogen", "Phosphorus", "Potassium"],
        y=[0, 0, 0],
        text=[0, 0, 0],
        textposition="outside",
        marker_color=[
            "#43A047",
            "#F9A825",
            "#1E88E5"
        ]
    )
)

fig.update_layout(
    template="plotly_white",
    font=dict(
        color="#263238"
    ),
    xaxis=dict(
        title="Nutrients",
        color="#263238"
    ),
    yaxis=dict(
        title="Value",
        color="#263238"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

#CROP INFORMATION

st.markdown("---")

st.subheader("Crop Information")

st.markdown("""

<div class="info-card blue">
<div class="info-title"> <!--
category: Nature
tags: [nature, plant, tree, autumn, fall, greenery, flower, forest, garden]
version: "1.29"
unicode: "ed4f"
-->
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="32"
  height="32"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#000000"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M5 21c.5 -4.5 2.5 -8 7 -10" />
  <path d="M9 18c6.218 0 10.5 -3.288 11 -12v-2h-4.014c-9 0 -11.986 4 -12 9c0 1 0 3 2 5h3z" />
</svg>
 Crop Name</div>
<div class="info-value">Not Available</div>
</div>

<div class="info-card green">
<div class="info-title"><!--
category: Design
tags: [water, rain, liquid]
version: "1.0"
unicode: "ea97"
-->
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="32"
  height="32"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#000000"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M7.502 19.423c2.602 2.105 6.395 2.105 8.996 0c2.602 -2.105 3.262 -5.708 1.566 -8.546l-4.89 -7.26c-.42 -.625 -1.287 -.803 -1.936 -.397a1.376 1.376 0 0 0 -.41 .397l-4.893 7.26c-1.695 2.838 -1.035 6.441 1.567 8.546z" />
</svg>
  Water Requirement</div>
<div class="info-value">Not Available</div>
</div>

<div class="info-card orange">
<div class="info-title"><!--
tags: [time-calendar, schedule-clock, appointment, date-time, event-schedule, timing, calendar-clock, agenda, time-management, calendar-event]
version: "2.41"
unicode: "fd2e"
-->
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="32"
  height="32"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#000000"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M10.5 21h-4.5a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2h12a2 2 0 0 1 2 2v3" />
  <path d="M16 3v4" />
  <path d="M8 3v4" />
  <path d="M4 11h10" />
  <path d="M18 18m-4 0a4 4 0 1 0 8 0a4 4 0 1 0 -8 0" />
  <path d="M18 16.5v1.5l.5 .5" />
</svg>
  Growth Duration</div>
<div class="info-value">Not Available</div>
</div>

<div class="info-card purple">
<div class="info-title"> <!--
category: Weather
version: "2.10"
unicode: "f84f"
-->
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="28"
  height="28"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#000000"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M12 18.004h-5.343c-2.572 -.004 -4.657 -2.011 -4.657 -4.487c0 -2.475 2.085 -4.482 4.657 -4.482c.393 -1.762 1.794 -3.2 3.675 -3.773c1.88 -.572 3.956 -.193 5.444 1c1.488 1.19 2.162 3.007 1.77 4.769h.99c.956 0 1.822 .39 2.449 1.02" />
  <path d="M19.001 19m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" />
  <path d="M19.001 15.5v1.5" />
  <path d="M19.001 21v1.5" />
  <path d="M22.032 17.25l-1.299 .75" />
  <path d="M17.27 20l-1.3 .75" />
  <path d="M15.97 17.25l1.3 .75" />
  <path d="M20.733 20l1.3 .75" />
</svg>
  Suitable Season</div>
<div class="info-value">Not Available</div>
</div>

<div class="info-card red">
<div class="info-title"><!--
tags: [sample, color, flask, liquid, container, glass, chemistry, test, laboratory, experimental, beta]
version: "1.0"
unicode: "eb3a"
-->
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="28"
  height="28"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#000000"
  stroke-width="1.25"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <path d="M20 8.04l-12.122 12.124a2.857 2.857 0 1 1 -4.041 -4.04l12.122 -12.124" />
  <path d="M7 13h8" />
  <path d="M19 15l1.5 1.6a2 2 0 1 1 -3 0l1.5 -1.6z" />
  <path d="M15 3l6 6" />
</svg>
  Ideal Soil pH</div>
<div class="info-value">Not Available</div>
</div>

""",unsafe_allow_html=True)












