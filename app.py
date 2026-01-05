import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# Optional charts: if plotly not installed, app still runs (no crash)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Autoimmune Clinical Decision Support",
    page_icon="🧬",
    layout="wide"
)

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "stage1"))
sys.path.append(os.path.join(BASE_DIR, "stage2"))

from predict1 import predict_disease_from_10
from predict2 import predict_response_from_10


# ---------------- SESSION STATE ----------------
if "screening_done" not in st.session_state:
    st.session_state.screening_done = False

if "disease_detected" not in st.session_state:
    st.session_state.disease_detected = False

if "disease_label" not in st.session_state:
    st.session_state.disease_label = ""

if "disease_conf" not in st.session_state:
    st.session_state.disease_conf = 0.0

if "disease_probs" not in st.session_state:
    st.session_state.disease_probs = None


# ---------------- NO-DISEASE GATE ----------------
def is_no_disease_by_rules(wbc, crp, esr, ana, rf, dsdna, il6, tnf):
    """
    Your stage-1 model has only disease classes (no Healthy class),
    so we apply a rule-gate to allow true 'No disease'.
    """
    # Normal/low inflammation ranges
    if not (4000 <= wbc <= 11000):
        return False
    if crp > 3.0:
        return False
    if esr > 15.0:
        return False

    # Autoimmune markers must be negative
    if ana == 1 or rf == 1 or dsdna == 1:
        return False

    # Cytokines low/normal
    if il6 > 5.0:
        return False
    if tnf > 6.0:
        return False

    return True


# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
    .main-title{
        text-align:center;
        font-size:40px;
        font-weight:800;
        background: linear-gradient(90deg,#00c6ff,#0072ff,#00ff88);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        margin-bottom:0px;
    }
    .sub-title{
        text-align:center;
        font-size:16px;
        color:#d9d9d9;
        margin-top:0px;
        margin-bottom:20px;
        opacity:0.9;
    }
    .card{
        border-radius:16px;
        padding:18px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }
    .mini{
        font-size:13px;
        opacity:0.85;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>🧬 Autoimmune Clinical Decision Support</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Disease screening and treatment outcome prediction using deep learning</div>", unsafe_allow_html=True)


# ==============================
# SECTION 1: SCREENING
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 🩺 Disease Screening")
st.markdown("<div class='mini'>Enter key clinical and immune markers</div>", unsafe_allow_html=True)
st.write("")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", min_value=1, max_value=100, value=22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wbc = st.number_input("WBC Count", value=6200.0)

with c2:
    crp = st.number_input("CRP (mg/L)", value=0.5)
    esr = st.number_input("ESR (mm/hr)", value=5.0)
    il6 = st.number_input("IL-6", value=1.8)

with c3:
    ana = st.selectbox("ANA (0/1)", [0, 1], index=0)
    rf = st.selectbox("Rheumatoid Factor (0/1)", [0, 1], index=0)
    dsdna = st.selectbox("Anti-dsDNA (0/1)", [0, 1], index=0)
    tnf = st.number_input("TNF-α", value=2.2)

gender_val = 0 if gender == "Male" else 1

inputs10 = [
    age, gender_val, wbc, crp, esr,
    ana, rf, dsdna, il6, tnf
]

st.write("")
run = st.button("🔍 Run Screening", type="primary", use_container_width=True)

if run:
    with st.spinner("Analyzing..."):
        time.sleep(0.6)

        # Rule gate first (true No Disease)
        if is_no_disease_by_rules(wbc, crp, esr, ana, rf, dsdna, il6, tnf):
            st.session_state.screening_done = True
            st.session_state.disease_detected = False
            st.session_state.disease_label = "No Autoimmune Disease"
            st.session_state.disease_conf = 1.0
            st.session_state.disease_probs = None
        else:
            # Model prediction
            pred_id, pred_label, confidence, probs_dict = predict_disease_from_10(inputs10)

            st.session_state.screening_done = True
            st.session_state.disease_detected = True
            st.session_state.disease_label = str(pred_label)
            st.session_state.disease_conf = float(confidence)
            st.session_state.disease_probs = probs_dict

# RESULT DISPLAY
if st.session_state.screening_done:
    st.write("")
    if not st.session_state.disease_detected:
        st.success("🟢 No autoimmune disease detected.")
        st.info("Treatment prediction is not required.")
    else:
        st.warning(f"⚠️ Autoimmune disease detected: **{st.session_state.disease_label}**")
        st.caption(f"Confidence: {st.session_state.disease_conf:.2f}")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")


# ==============================
# DISEASE GRAPH (ONLY IF DISEASE DETECTED)
# ==============================
if st.session_state.screening_done and st.session_state.disease_detected and st.session_state.disease_probs:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"## ✅ Predicted Disease: **{st.session_state.disease_label}**")
    st.caption("Probability distribution across disease classes")
    probs_dict = st.session_state.disease_probs

    # Convert dict to dataframe
    df = pd.DataFrame({
        "Disease": list(probs_dict.keys()),
        "Probability": list(probs_dict.values())
    }).sort_values("Probability", ascending=False)

    if PLOTLY_OK:
        fig = px.bar(df, x="Disease", y="Probability", title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index("Disease"))

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")


# ==============================
# SECTION 2: TREATMENT OUTCOME (ONLY IF DISEASE DETECTED)
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 💊 Treatment Outcome")
if not (st.session_state.screening_done and st.session_state.disease_detected):
    st.info("This section becomes available only when a disease is detected.")
else:
    st.success(f"Enabled for: **{st.session_state.disease_label}**")
    st.write("")

    st.markdown("<div class='mini'>Enter gene expression values</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    gene10 = []
    defaults = [0.10] * 10

    for i in range(10):
        with cols[i % 5]:
            gene10.append(st.number_input(f"GENE_{i}", value=float(defaults[i])))

    st.write("")
    run2 = st.button("💉 Predict Treatment Response", type="primary", use_container_width=True)

    if run2:
        with st.spinner("Predicting..."):
            time.sleep(0.6)

            out = predict_response_from_10(gene10)

            # supports (pred,p0,p1,debug) OR (pred,p0,p1)
            if len(out) == 4:
                pred, p0, p1, _ = out
            else:
                pred, p0, p1 = out

            pred = int(pred)
            p0 = float(p0)
            p1 = float(p1)

            # Mapping: response=0 => Not Respond, response=1 => Respond
            label = "Responds to Treatment ✅" if pred == 1 else "Does Not Respond ❌"

            st.write("")
            if pred == 1:
                st.success(label)
            else:
                st.error(label)

            # Simple probability display
            st.metric("Respond probability", f"{p1:.2f}")
            st.metric("Not-Respond probability", f"{p0:.2f}")

            # Optional chart
            if PLOTLY_OK:
                df2 = pd.DataFrame({
                    "Outcome": ["Not Respond", "Respond"],
                    "Probability": [p0, p1]
                })
                fig2 = px.pie(df2, names="Outcome", values="Probability", title="")
                st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")
