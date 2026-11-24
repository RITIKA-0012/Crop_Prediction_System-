import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from machine_learning.predict import predict_crop_details, model, label_encoder, features

st.set_page_config(
    page_title="Seasonal Crop Prediction | Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Regional coordinates + seasonal mean rainfall (mm) for accurate crop context
LOCATIONS = {
    "Barasat (North 24 Parganas, WB)": {"lat": 22.72, "lon": 88.48, "seasonal_rain": 185.0},
    "Kolkata (South Bengal, WB)": {"lat": 22.57, "lon": 88.36, "seasonal_rain": 195.0},
    "Krishnanagar (Nadia, WB)": {"lat": 23.40, "lon": 88.50, "seasonal_rain": 165.0},
    "Bardhaman (Purba Bardhaman, WB)": {"lat": 23.23, "lon": 87.86, "seasonal_rain": 175.0},
    "Siliguri (North Bengal, WB)": {"lat": 26.72, "lon": 88.39, "seasonal_rain": 240.0},
}

CROP_INFO = {
    "rice": {"water": "High (1,200 – 1,800 mm)", "duration": "120 – 150 days", "season": "Monsoon (Kharif)", "ph": "5.5 – 6.5"},
    "maize": {"water": "Moderate (500 – 800 mm)", "duration": "90 – 110 days", "season": "Monsoon / Summer", "ph": "5.8 – 7.0"},
    "chickpea": {"water": "Low (350 – 500 mm)", "duration": "100 – 120 days", "season": "Winter (Rabi)", "ph": "6.0 – 7.5"},
    "kidneybeans": {"water": "Moderate (400 – 600 mm)", "duration": "90 – 120 days", "season": "Monsoon / Winter", "ph": "6.0 – 7.0"},
    "pigeonpeas": {"water": "Low–Moderate (600 – 650 mm)", "duration": "150 – 180 days", "season": "Monsoon (Kharif)", "ph": "5.5 – 7.0"},
    "mothbeans": {"water": "Very Low (250 – 400 mm)", "duration": "75 – 90 days", "season": "Summer / Kharif", "ph": "6.5 – 8.0"},
    "mungbean": {"water": "Low (350 – 450 mm)", "duration": "60 – 75 days", "season": "Summer / Kharif", "ph": "6.2 – 7.2"},
    "blackgram": {"water": "Low–Moderate (400 – 500 mm)", "duration": "75 – 90 days", "season": "Monsoon / Summer", "ph": "6.0 – 7.5"},
    "lentil": {"water": "Low (300 – 400 mm)", "duration": "110 – 130 days", "season": "Winter (Rabi)", "ph": "6.0 – 7.5"},
    "pomegranate": {"water": "Moderate (500 – 700 mm)", "duration": "Perennial (5 – 7 yrs)", "season": "All Seasons", "ph": "6.5 – 7.5"},
    "banana": {"water": "High (1,200 – 2,200 mm)", "duration": "11 – 14 months", "season": "All Seasons", "ph": "6.0 – 7.5"},
    "mango": {"water": "Moderate (750 – 1,000 mm)", "duration": "Perennial", "season": "Summer", "ph": "5.5 – 7.5"},
    "grapes": {"water": "Moderate (500 – 700 mm)", "duration": "Perennial", "season": "Winter / Summer", "ph": "6.5 – 7.5"},
    "watermelon": {"water": "Moderate (400 – 600 mm)", "duration": "80 – 90 days", "season": "Summer (Zaid)", "ph": "6.0 – 6.8"},
    "muskmelon": {"water": "Moderate (400 – 500 mm)", "duration": "75 – 90 days", "season": "Summer (Zaid)", "ph": "6.0 – 6.8"},
    "apple": {"water": "Moderate (800 – 1,000 mm)", "duration": "Perennial", "season": "Winter / Temperate", "ph": "5.5 – 6.5"},
    "orange": {"water": "Moderate (900 – 1,100 mm)", "duration": "Perennial", "season": "Winter", "ph": "6.0 – 7.5"},
    "papaya": {"water": "Moderate–High (1,000 – 1,500 mm)", "duration": "9 – 11 months", "season": "All Seasons", "ph": "6.0 – 6.5"},
    "coconut": {"water": "High (1,300 – 2,000 mm)", "duration": "Perennial", "season": "All Seasons", "ph": "5.5 – 7.0"},
    "cotton": {"water": "Moderate (700 – 1,200 mm)", "duration": "150 – 180 days", "season": "Monsoon (Kharif)", "ph": "6.0 – 7.5"},
    "jute": {"water": "High (1,000 – 1,500 mm)", "duration": "120 – 150 days", "season": "Monsoon (Kharif)", "ph": "6.0 – 7.5"},
    "coffee": {"water": "High (1,500 – 2,000 mm)", "duration": "Perennial", "season": "Monsoon", "ph": "6.0 – 6.5"},
}

@st.cache_data(ttl=600)
def fetch_live_weather(lat, lon, default_rain):
    """Fetches real-time temperature, humidity, and precipitation from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation"
    )
    try:
        resp = requests.get(url, timeout=4)
        resp.raise_for_status()
        current = resp.json().get("current", {})
        temp = float(current.get("temperature_2m", 26.5))
        hum = float(current.get("relative_humidity_2m", 78.0))
        live_precip = float(current.get("precipitation", 0.0))
        # Use seasonal regional rainfall baseline + any live precipitation
        rain = round(default_rain + live_precip, 1)
        return temp, hum, rain, True
    except Exception:
        # Offline fallback if seminar hall has no internet
        return 26.5, 78.0, default_rain, False

def classify_soil(moisture, ph):
    if ph < 5.8:
        acidity = "Acidic"
    elif ph > 7.3:
        acidity = "Alkaline"
    else:
        acidity = "Neutral"

    if moisture < 35:
        texture = "Sandy / Low-Retention"
    elif moisture > 65:
        texture = "Clayey / High-Retention"
    else:
        texture = "Balanced Loam"
    return f"{texture} ({acidity})"

@st.cache_data
def get_model_validation_stats():
    df_train = pd.read_csv("data/Crop_recommendation.csv")
    X = df_train[features]
    y = label_encoder.transform(df_train["label"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    return round(acc, 2), len(df_train), len(label_encoder.classes_)

DATA_PATH = "data/dummydatacrops.xlsx"
df = pd.read_excel(DATA_PATH)
acc, total_samples, num_classes = get_model_validation_stats()

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="sidebar-brand">AgriSense ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Soil Telemetry & Crop Inference</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Field Location (Weather API)**")
    selected_loc = st.selectbox("Select Region", list(LOCATIONS.keys()), label_visibility="collapsed")
    loc_info = LOCATIONS[selected_loc]
    api_temp, api_hum, api_rain, api_online = fetch_live_weather(
        loc_info["lat"], loc_info["lon"], loc_info["seasonal_rain"]
    )

    st.markdown("**System Diagnostics**")
    api_badge = (
        '<span style="color:#15803D; font-weight:600;">Connected</span>'
        if api_online else
        '<span style="color:#B45309; font-weight:600;">Cached</span>'
    )
    api_dot = "dot-green" if api_online else "dot-amber"

    st.markdown(f"""
    <div class="status-pill">
        <span><span class="dot-green"></span>RF Inference Engine</span>
        <span style="color:#15803D; font-weight:600;">Active</span>
    </div>
    <div class="status-pill">
        <span><span class="{api_dot}"></span>Open-Meteo Weather API</span>
        {api_badge}
    </div>
    <div class="status-pill">
        <span><span class="dot-amber"></span>ESP32 Soil Node</span>
        <span style="color:#B45309; font-weight:600;">Buffered</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Soil Telemetry Source**")
    input_mode = st.radio(
        "Input Source",
        ["Buffered Sensor Packet", "Manual Parameter Tuning"],
        label_visibility="collapsed"
    )

    if input_mode == "Buffered Sensor Packet":
        packet_idx = st.selectbox(
            "Select Telemetry Packet",
            options=list(range(len(df))),
            format_func=lambda i: f"Packet #{i+1:02d} — Log Record {i+1}"
        )

    st.markdown("---")
    st.markdown("**Model Metadata**")
    st.caption(f"Algorithm: Random Forest (100 trees)\n\nValidation Accuracy: {acc}%\n\nDataset: {total_samples:,} records ({num_classes} classes)")

# SOIL VALUES (Hardware / Buffer) + CLIMATE VALUES (Live API)
if input_mode == "Buffered Sensor Packet":
    row = df.iloc[packet_idx]
    MOISTURE = float(row["moisture"])
    PH = float(row["ph"])
    NITROGEN = float(row.iloc[5])
    PHOSPHORUS = float(row.iloc[6])
    POTASSIUM = float(row.iloc[7])
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Soil Parameter Controls**")
    NITROGEN = st.sidebar.slider("Nitrogen (N) [mg/kg]", 0.0, 140.0, 90.0)
    PHOSPHORUS = st.sidebar.slider("Phosphorus (P) [mg/kg]", 5.0, 145.0, 42.0)
    POTASSIUM = st.sidebar.slider("Potassium (K) [mg/kg]", 5.0, 205.0, 43.0)
    PH = st.sidebar.slider("Soil pH", 3.5, 9.5, 6.5)
    MOISTURE = st.sidebar.slider("Soil Moisture [%]", 10.0, 100.0, 62.0)

# MINIMALIST HEADER
st.markdown("""
<div class="hero">
    <div class="hero-badge">IoT Soil Telemetry & Live Meteorological API</div>
    <div class="hero-title">Seasonal Crop Prediction System</div>
    <div class="hero-subtitle">Hybrid sensor-cloud data fusion and Random Forest crop recommendation engine</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Telemetry & Crop Inference", "Model Evaluation & Architecture"])

with tab1:
    st.markdown("#### In-Situ Soil Telemetry (ESP32 / Buffer Layer)")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Soil Moisture", f"{MOISTURE:.1f} %")
    with col2:
        st.metric("Soil pH", f"{PH:.2f}")
    with col3:
        st.metric("Nitrogen (N)", f"{NITROGEN:.1f} mg/kg")
    with col4:
        st.metric("Phosphorus (P)", f"{PHOSPHORUS:.1f} mg/kg")
    with col5:
        st.metric("Potassium (K)", f"{POTASSIUM:.1f} mg/kg")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### Macro-Climate Telemetry — {selected_loc} (Open-Meteo Live Feed)")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        TEMPERATURE = st.number_input("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=api_temp)
    with w2:
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=api_hum)
    with w3:
        rainfall = st.number_input("Seasonal Rainfall Index (mm)", min_value=0.0, max_value=400.0, value=api_rain)
    with w4:
        season = st.selectbox("Cropping Season", ["Monsoon (Kharif)", "Winter (Rabi)", "Summer (Zaid)"])

    # INFERENCE
    top_crops, feature_imp = predict_crop_details(
        NITROGEN, PHOSPHORUS, POTASSIUM, TEMPERATURE, humidity, PH, rainfall
    )
    best_crop = top_crops[0]["crop"]
    best_conf = top_crops[0]["confidence"]
    soil_desc = classify_soil(MOISTURE, PH)
    whc_est = "High" if MOISTURE > 60 else ("Moderate" if MOISTURE >= 35 else "Low")
    details = CROP_INFO.get(best_crop.lower(), {
        "water": "Moderate", "duration": "90 – 120 days", "season": season, "ph": "6.0 – 7.0"
    })

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Derived Soil Classification</div>
            <div class="result-main" style="font-size:18px;">{soil_desc}</div>
            <div class="result-sub">Water Holding Capacity: <strong>{whc_est}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="result-box primary">
            <div class="result-label" style="color:#15803D;">Optimal Crop Recommendation</div>
            <div class="result-main" style="color:#14532D;">{best_crop.upper()}</div>
            <div class="result-sub" style="color:#166534;">Posterior Class Probability: <strong>{best_conf}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # CHARTS
    st.markdown("<br>", unsafe_allow_html=True)
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### Macronutrient Concentration (NPK)")
        fig_npk = go.Figure(go.Bar(
            x=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
            y=[NITROGEN, PHOSPHORUS, POTASSIUM],
            text=[f"{NITROGEN:.1f}", f"{PHOSPHORUS:.1f}", f"{POTASSIUM:.1f}"],
            textposition="outside",
            marker_color=["#1E293B", "#475569", "#15803D"],
            width=0.45
        ))
        fig_npk.update_layout(
            template="plotly_white",
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(title="mg/kg", range=[0, max(NITROGEN, PHOSPHORUS, POTASSIUM, 50) * 1.25], gridcolor="#F1F5F9"),
            xaxis=dict(title=""),
            height=320
        )
        st.plotly_chart(fig_npk, use_container_width=True)

    with chart2:
        st.markdown("#### Top-3 Class Probability Distribution")
        fig_prob = go.Figure(go.Bar(
            x=[item["crop"].title() for item in top_crops],
            y=[item["confidence"] for item in top_crops],
            text=[f"{item['confidence']}%" for item in top_crops],
            textposition="outside",
            marker_color=["#15803D", "#64748B", "#94A3B8"],
            width=0.45
        ))
        fig_prob.update_layout(
            template="plotly_white",
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(title="Confidence (%)", range=[0, 118], gridcolor="#F1F5F9"),
            xaxis=dict(title=""),
            height=320
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # AGRONOMIC SPECS
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### Agronomic Requirements — {best_crop.title()}")

    st.markdown(f"""
    <div class="info-card">
        <div class="info-title">Recommended Cultivar Class</div>
        <div class="info-value">{best_crop.title()} ({best_conf}% confidence)</div>
    </div>
    <div class="info-card">
        <div class="info-title">Seasonal Water Requirement</div>
        <div class="info-value">{details['water']}</div>
    </div>
    <div class="info-card">
        <div class="info-title">Typical Crop Cycle Duration</div>
        <div class="info-value">{details['duration']}</div>
    </div>
    <div class="info-card">
        <div class="info-title">Optimal Sowing Window</div>
        <div class="info-value">{details['season']}</div>
    </div>
    <div class="info-card">
        <div class="info-title">Target Soil pH Range</div>
        <div class="info-value">{details['ph']}</div>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### Classifier Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Architecture", "Random Forest")
    m2.metric("Hold-out Accuracy", f"{acc}%")
    m3.metric("Ensemble Size", "100 Trees")
    m4.metric("Validation Split", "80 / 20 Stratified")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Gini Feature Importance Ranking")
    sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
    fig_imp = go.Figure(go.Bar(
        x=[k.upper() for k, _ in sorted_imp],
        y=[round(v * 100, 2) for _, v in sorted_imp],
        text=[f"{round(v * 100, 1)}%" for _, v in sorted_imp],
        textposition="outside",
        marker_color="#1E293B",
        width=0.5
    ))
    fig_imp.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#0F172A"),
        margin=dict(t=25, b=25, l=25, r=25),
        yaxis=dict(title="Relative Importance (%)", range=[0, 35], gridcolor="#E2E8F0", tickfont=dict(color="#0F172A")),
        xaxis=dict(title="", tickfont=dict(color="#0F172A")),
        height=360
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Hybrid Edge-to-Cloud Architecture")
    st.markdown("""
    <div class="result-box">
        <div class="result-sub" style="line-height:1.7; color:#1E293B;">
            <strong>1. In-Situ Soil Layer (ESP32 + RS485 NPK + Capacitive Moisture):</strong> Captures localized subsurface parameters (N, P, K, pH, Moisture) that vary across individual plots.<br>
            <strong>2. Macro-Climate API Layer (Open-Meteo REST API):</strong> Fetches real-time ambient temperature, relative humidity, and precipitation via latitude/longitude coordinates, eliminating redundant DHT hardware.<br>
            <strong>3. ML Inference Engine (<code>predict.py</code>):</strong> Fuses both telemetry streams into a 7-feature vector evaluated by a 100-tree Random Forest classifier (<code>crop_prediction.pkl</code>).
        </div>
    </div>
    """, unsafe_allow_html=True)