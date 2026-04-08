import streamlit as st
import requests
import matplotlib.pyplot as plt

st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #f6f8fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
        color: #111827;
    }

    /* ---------- Cards ---------- */
    .section-card {
        background: white;
        padding: 1.2rem 1.3rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    /* ---------- Risk badges ---------- */
    .risk-high {
        background: #fee2e2;
        color: #b91c1c;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #fecaca;
        display: inline-block;
    }

    .risk-low {
        background: #dcfce7;
        color: #166534;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #bbf7d0;
        display: inline-block;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e5e7eb;
    }

    /* ---------- Button ---------- */
    div.stButton > button {
        border-radius: 10px;
        height: 2.8rem;
        font-weight: 600;
        border: none;
        background: #2563eb;
        color: white;
    }

    div.stButton > button:hover {
        background: #1d4ed8;
    }

    /* ---------- Progress bar ---------- */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #2563eb;
    }

</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown("## Patient Data")
st.sidebar.caption("Enter the patient's clinical information below.")

age = st.sidebar.slider("Age", 1, 120, 55)
sex = st.sidebar.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)
cp = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic"
    }[x]
)
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 50, 250, 130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
restecg = st.sidebar.selectbox(
    "Resting ECG Results",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "Left Ventricular Hypertrophy"
    }[x]
)
thalach = st.sidebar.slider("Max Heart Rate Achieved", 50, 250, 150)
exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }[x]
)
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox(
    "Thalassemia",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x]
)

predict = st.sidebar.button("Predict Risk", use_container_width=True)

# ---------- Default landing state ----------
if not predict:
    st.title("🫀 Heart Disease Risk Predictor")
    st.markdown(
      "Enter patient clinical data to get a heart disease risk prediction "
      "powered by a Logistic Regression model trained on the UCI Heart Disease dataset."
    )
    st.divider()
    st.info("Fill in the patient data from the sidebar and click **Predict Risk**.")
    st.stop()

# ---------- Prediction request ----------
payload = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

with st.spinner("Running prediction..."):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

prob = result["probability"]
prediction_label = result["prediction_label"]
prediction = result["prediction"]

# ---------- Top summary ----------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction Summary")

    if prediction == 1:
        st.markdown(
            f'<div class="risk-high">High Risk • {prediction_label}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="risk-low">Low Risk • {prediction_label}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Predicted Probability", f"{prob:.1%}")
    st.progress(float(prob))
    st.caption("This score represents the model’s estimated probability of heart disease.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Confidence Distribution")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(["No Disease", "Disease"], [1 - prob, prob], color=["#60a5fa", "#ef4444"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Input Summary ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Patient Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("Resting BP", trestbps)
c3.metric("Cholesterol", chol)
c4.metric("Max Heart Rate", thalach)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Sex", "Male" if sex == 1 else "Female")
c6.metric("Exercise Angina", "Yes" if exang == 1 else "No")
c7.metric("Fasting Sugar", "High" if fbs == 1 else "Normal")
c8.metric("Major Vessels", ca)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- SHAP Explanation ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Feature Contribution (SHAP)")
st.caption("Positive values push the prediction toward Disease. Negative values push it toward No Disease.")

shap_vals = result["shap_values"]
sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
features, values = zip(*sorted_shap)
colors = ["#ff6b6b" if v > 0 else "#4f8bf9" for v in values]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.barh(features[::-1], values[::-1], color=colors[::-1])
ax2.axvline(0, color="white", linewidth=0.8)
ax2.set_xlabel("SHAP Value")
ax2.set_title("Top 10 Feature Contributions")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Raw response ----------
with st.expander("View raw API response"):
    st.json(result)