import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="⚡Power Plant Energy Predictor", layout="wide")

# ---------- DARK UI ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
}
.header {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}
.subheader {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
}
.result {
    background: #1e293b;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #22c55e;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="header">⚡ Power Plant Energy Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader"></div>', unsafe_allow_html=True)

# ---------- MODEL ----------
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ANN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙ Controls")
st.sidebar.success("Model Loaded ✅")

AT = st.sidebar.slider("Temperature", 0.0, 50.0, 25.0)
V = st.sidebar.slider("Vacuum", 25.0, 80.0, 40.0)
AP = st.sidebar.slider("Pressure", 900.0, 1100.0, 1010.0)
RH = st.sidebar.slider("Humidity", 0.0, 100.0, 60.0)

# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- MAIN ----------
col1, col2 = st.columns(2)

# Prediction
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Prediction")

    if st.button("🚀 Predict"):
        input_data = np.array([[AT, V, AP, RH]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        pred = model(input_tensor).item()

        st.markdown(f'<div class="result">⚡ {pred:.2f} MW</div>', unsafe_allow_html=True)

        st.session_state.history.append({
            "AT": AT, "V": V, "AP": AP, "RH": RH,
            "Prediction": pred
        })

    st.markdown('</div>', unsafe_allow_html=True)

# Batch
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Batch")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if st.button("Run Batch"):
            data = df[['AT', 'V', 'AP', 'RH']]
            scaled = scaler.transform(data)
            tensor = torch.tensor(scaled, dtype=torch.float32)

            df["Predicted"] = model(tensor).detach().numpy()
            st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- COMPACT COMPARISON ----------
if st.session_state.history:
    latest = st.session_state.history[-1]
    hist = pd.DataFrame(st.session_state.history)

    st.markdown("## 📊 Quick Comparison")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temp", latest["AT"])
    m2.metric("Vacuum", latest["V"])
    m3.metric("Pressure", latest["AP"])
    m4.metric("Humidity", latest["RH"])

    st.markdown("---")

    # ONLY 2 CLEAN CHARTS
    c1, c2 = st.columns(2)

    # 🔹 Bar Chart
    with c1:
        st.markdown("**Feature Comparison**")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(["AT","V","AP","RH"],
               [latest["AT"], latest["V"], latest["AP"], latest["RH"]])
        ax.tick_params(labelsize=8)
        st.pyplot(fig)

    # 🔹 Trend Chart
    with c2:
        st.markdown("**Prediction Trend**")
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.plot(hist["Prediction"], marker='o')
        ax2.tick_params(labelsize=8)
        st.pyplot(fig2)

    # Insight
    avg = hist["Prediction"].mean()
    current = hist["Prediction"].iloc[-1]

    st.metric("⚡ Current Output", f"{current:.2f} MW", f"{current-avg:.2f} vs Avg")

# ---------- HISTORY ----------
if st.session_state.history:
    st.markdown("### 🧾 History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# ---------- FOOTER ----------
st.markdown("""
---
<center style='color: #64748b'>
Clean ML Dashboard | Minimal UI 🚀
</center>
""", unsafe_allow_html=True)