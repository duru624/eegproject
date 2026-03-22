import streamlit as st
import os
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data"
st.set_page_config(page_title="NeuroPulse", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "history_TestOnMe" not in st.session_state:
    st.session_state.history_TestOnMe = {}

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# -------------------------
# AUTH
# -------------------------
st.sidebar.title("Account")
username = st.sidebar.text_input("Username")

col1, col2 = st.sidebar.columns(2)

if col1.button("Login"):
    if username in st.session_state.users:
        st.session_state.current_user = username
        st.sidebar.success("Logged in")
    else:
        st.sidebar.error("User not found")

if col2.button("Register"):
    if username and username not in st.session_state.users:
        st.session_state.users[username] = []
        st.session_state.history_TestOnMe[username] = []
        st.session_state.current_user = username
        st.sidebar.success("Account created")
    else:
        st.sidebar.error("Invalid username")

if st.session_state.current_user:
    if st.session_state.current_user not in st.session_state.history_TestOnMe:
        st.session_state.history_TestOnMe[st.session_state.current_user] = []

    if st.sidebar.button("Logout"):
        st.session_state.current_user = None
        st.sidebar.success("Logged out")

# -------------------------
# LOGIN CHECK
# -------------------------
if st.session_state.current_user is None:
    st.title("🧠 NeuroPulse")
    st.write("Mental State Detection Without Words")
    st.stop()

# -------------------------
# MAIN UI
# -------------------------
st.title("🧠 NeuroPulse")
st.write("User:", st.session_state.current_user)

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📝 Test On Me"])

# ===========================
# EEG MODE (FINAL)
# ===========================
with tab1:

    st.header("EEG Analysis (Random Auto Selection)")

    # DATA CHECK
    if not os.path.exists(DATA_PATH):
        st.error("data folder yok!")
        st.stop()

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]

    if len(files) == 0:
        st.error("data klasöründe .edf yok")
        st.stop()

    # BUTTON → RANDOM SEÇİM
    if st.button("🎲 Analyze Random EEG"):
        st.session_state.selected_file = random.choice(files)

    # SEÇİLMEDİYSE
    if st.session_state.selected_file is None:
        st.info("Butona bas → sistem rastgele dosya seçsin")
        st.stop()

    file = st.session_state.selected_file
    path = os.path.join(DATA_PATH, file)

    st.success(f"Seçilen dosya: {file}")

    # EEG OKU
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    signal = raw.get_data()[0]

    # FFT
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/raw.info['sfreq'])

    # BANDLAR (mean → daha stabil)
    delta = np.mean(fft[(freqs >= 0.5) & (freqs < 4)])
    theta = np.mean(fft[(freqs >= 4) & (freqs < 8)])
    alpha = np.mean(fft[(freqs >= 8) & (freqs < 12)])
    beta = np.mean(fft[(freqs >= 12) & (freqs < 30)])

    # NORMALIZE
    total = delta + theta + alpha + beta
    delta /= total
    theta /= total
    alpha /= total
    beta /= total

    # CLASSIFICATION (BALANCED)
    values = {
        "Calm": alpha,
        "Stressed": beta,
        "Drowsy": theta,
        "Deep": delta
    }

    state = max(values, key=values.get)

    # UI
    colors = {
        "Calm": "#4CAF50",
        "Stressed": "#F44336",
        "Drowsy": "#FFC107",
        "Deep": "#2196F3"
    }

    advice = {
        "Calm": "Keep going 🌿",
        "Stressed": "Breathe 🫁",
        "Drowsy": "Rest 😴",
        "Deep": "Very relaxed 🧘"
    }

    st.markdown(f"""
    <div style='background:{colors[state]};
                padding:30px;
                border-radius:20px;
                text-align:center;
                color:white'>
        <h1>{state}</h1>
        <p>{advice[state]}</p>
    </div>
    """, unsafe_allow_html=True)

    # GRAPH
    fig, ax = plt.subplots()
    ax.plot(signal[:2000])
    st.pyplot(fig)

    st.bar_chart({
        "delta":[delta],
        "theta":[theta],
        "alpha":[alpha],
        "beta":[beta]
    })

# ===========================
# TEST ON ME
# ===========================
with tab2:

    st.header("Test On Me")

    hr = st.slider("Heart rate", 50, 120, 70)
    stress = st.slider("Stress", 0, 10, 5)
    sleep = st.slider("Sleep", 0, 10, 7)

    if st.button("Analyze Me"):

        score = stress + (hr/100) - sleep

        if score > 8:
            state = "Stressed"
        elif score < 4:
            state = "Calm"
        else:
            state = "Anxious"

        st.write("Result:", state)

        st.session_state.history_TestOnMe[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": state
        })

    st.subheader("History")

    for h in st.session_state.history_TestOnMe[st.session_state.current_user][::-1]:
        st.write(h)
