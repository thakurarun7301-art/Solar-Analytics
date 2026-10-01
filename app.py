import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Solar Power Analytics Suite",
    page_icon="☀️",
    layout="wide"
)


reg_model = joblib.load("models/regression_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
nb_model = joblib.load("models/nb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")


theme_mode = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])

if theme_mode == "Dark":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_color = "#1E1E1E"
    border_color = "#2E2E2E"
    accent = "#F39C12"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    card_color = "#F5F7FA"
    border_color = "#D9D9D9"
    accent = "#1f77b4"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {card_color} !important;
        border-right: 1px solid {border_color};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    .custom-card {{
        background-color: {card_color};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}

    .hero-card {{
        background: linear-gradient(135deg, {card_color}, {bg_color});
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 18px;
    }}

    .section-title {{
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
        color: {text_color};
    }}

    .section-subtitle {{
        font-size: 15px;
        opacity: 0.85;
        margin-bottom: 0;
        color: {text_color};
    }}

    div[data-testid="stMetric"] {{
        background-color: {card_color};
        padding: 16px;
        border-radius: 16px;
        border: 1px solid {border_color};
    }}

    div[data-baseweb="select"] > div {{
        background-color: {card_color} !important;
        color: {text_color} !important;
    }}

    div[data-baseweb="radio"] * {{
        color: {text_color} !important;
    }}

    div[data-baseweb="slider"] * {{
        color: {text_color} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {card_color};
        border-radius: 12px 12px 0 0;
        padding: 10px 18px;
        border: 1px solid {border_color};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {accent} !important;
        color: white !important;
    }}

    .stButton > button {{
        background-color: {accent};
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 600;
    }}

    .stDownloadButton > button {{
        background-color: {accent};
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 600;
    }}

    h1, h2, h3, h4, h5, h6, p, label, div {{
        color: {text_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# DATA SOURCE
# -----------------------------
st.sidebar.header("Dataset Settings")
data_source = st.sidebar.radio("Choose Dataset Source", ["Default Dataset", "Upload CSV"])

if data_source == "Default Dataset":
    df = pd.read_csv("data/solar_data.csv")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.warning("No file uploaded. Using default dataset.")
        df = pd.read_csv("data/solar_data.csv")

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("Prediction Inputs")
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 28.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
irradiance = st.sidebar.slider("Irradiance", 0.0, 1200.0, 700.0)
wind_speed = st.sidebar.slider("Wind Speed", 0.0, 20.0, 5.0)
panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 0.0, 100.0, 85.0)

st.sidebar.header("Model Selection")
prediction_mode = st.sidebar.selectbox(
    "Choose Prediction Mode",
    ["Power Prediction", "Panel Health Prediction"]
)

health_model_choice = st.sidebar.selectbox(
    "Choose Classification Model",
    ["SVM", "Naive Bayes"]
)

input_df = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "irradiance": irradiance,
    "wind_speed": wind_speed,
    "panel_efficiency": panel_efficiency
}])

# -----------------------------
# PREDICT
# -----------------------------
predict_clicked = st.sidebar.button("Run Prediction")

predicted_power = reg_model.predict(input_df)[0]
scaled_input = scaler.transform(input_df)
svm_status = encoder.inverse_transform(svm_model.predict(scaled_input))[0]
nb_status = encoder.inverse_transform(nb_model.predict(scaled_input))[0]

if health_model_choice == "SVM":
    selected_status = svm_status
else:
    selected_status = nb_status

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    f"""
    <div class="hero-card">
        <div class="section-title">☀️ Solar Power Analytics Suite</div>
        <div class="section-subtitle">
            Interactive ML dashboard with theme switch, dataset upload, model selection, and prediction insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# QUICK STATUS
# -----------------------------
top1, top2, top3, top4 = st.columns(4)
top1.metric("Temperature", f"{temperature:.1f} °C")
top2.metric("Humidity", f"{humidity:.1f} %")
top3.metric("Irradiance", f"{irradiance:.1f}")
top4.metric("Efficiency", f"{panel_efficiency:.1f} %")

if predict_clicked:
    st.success("Prediction updated successfully.")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Prediction Lab",
    "Dataset Explorer",
    "Model Comparison"
])

# -----------------------------
# TAB 1: OVERVIEW
# -----------------------------
with tab1:
    st.subheader("Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Power Output", f"{predicted_power:.2f}")
    col2.metric("SVM Status", svm_status)
    col3.metric("Naive Bayes Status", nb_status)

    insight1, insight2 = st.columns(2)

    with insight1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write("### Live Input Summary")
        st.write(f"- Temperature: **{temperature:.1f} °C**")
        st.write(f"- Humidity: **{humidity:.1f} %**")
        st.write(f"- Irradiance: **{irradiance:.1f}**")
        st.write(f"- Wind Speed: **{wind_speed:.1f}**")
        st.write(f"- Panel Efficiency: **{panel_efficiency:.1f} %**")
        st.markdown('</div>', unsafe_allow_html=True)

    with insight2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write("### Smart Insight")
        if predicted_power > 500:
            st.success("High expected solar output under current conditions.")
        elif predicted_power > 300:
            st.info("Moderate solar output predicted.")
        else:
            st.warning("Low output predicted. Conditions may be suboptimal.")

        if selected_status == "Optimal":
            st.success("Panel health looks strong.")
        elif selected_status == "Maintenance Required":
            st.warning("Panel health suggests maintenance may be needed.")
        else:
            st.error("Panel health suggests critical attention is needed.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# TAB 2: PREDICTION LAB
# -----------------------------
with tab2:
    st.subheader("Prediction Lab")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write("### Selected Prediction Mode")

        if prediction_mode == "Power Prediction":
            st.metric("Power Output", f"{predicted_power:.2f}")
            st.progress(min(max(predicted_power / 700, 0.0), 1.0))
        else:
            st.metric("Predicted Panel Health", selected_status)
            if selected_status == "Optimal":
                st.progress(0.9)
            elif selected_status == "Maintenance Required":
                st.progress(0.55)
            else:
                st.progress(0.2)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write("### All Model Outputs")
        a, b, c = st.columns(3)
        a.metric("Linear Regression", f"{predicted_power:.2f}")
        b.metric("SVM", svm_status)
        c.metric("Naive Bayes", nb_status)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write("### Download Current Input")
        download_df = input_df.copy()
        download_df["predicted_power_output"] = predicted_power
        download_df["svm_status"] = svm_status
        download_df["naive_bayes_status"] = nb_status
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prediction CSV",
            data=csv_data,
            file_name="solar_prediction_output.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# TAB 3: DATASET EXPLORER
# -----------------------------
with tab3:
    st.subheader("Dataset Explorer")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.write("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    required_columns = ["temperature", "humidity", "irradiance", "power_output", "status"]

    if all(col in df.columns for col in required_columns):
        c1, c2 = st.columns(2)

        with c1:
            st.write("### Irradiance vs Power Output")
            st.scatter_chart(df, x="irradiance", y="power_output")

            st.write("### Humidity vs Power Output")
            st.scatter_chart(df, x="humidity", y="power_output")

        with c2:
            st.write("### Temperature vs Power Output")
            st.scatter_chart(df, x="temperature", y="power_output")

            st.write("### Panel Status Distribution")
            status_counts = df["status"].value_counts()
            st.bar_chart(status_counts)
    else:
        st.error("Uploaded dataset does not contain the required columns:")
        st.code("temperature, humidity, irradiance, power_output, status")

# -----------------------------
# TAB 4: MODEL COMPARISON
# -----------------------------
with tab4:
    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "SVM", "Naive Bayes"],
        "Task": ["Regression", "Classification", "Classification"],
        "Output": ["Power Output", "Panel Health", "Panel Health"],
        "Used In App": ["Yes", "Yes", "Yes"]
    })

    st.table(comparison_df)

    with st.expander("Why use multiple models?"):
        st.write("- Linear Regression is used for continuous power prediction.")
        st.write("- SVM is used for panel health classification.")
        st.write("- Naive Bayes gives a second classification view for comparison.")

    with st.expander("Project Insights"):
        st.write("- Higher irradiance usually improves solar output.")
        st.write("- Higher humidity can reduce performance.")
        st.write("- Lower panel efficiency increases maintenance risk.")
        st.write("- Multiple model outputs improve interpretability.")