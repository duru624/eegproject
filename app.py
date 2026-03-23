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
# SESSION STATE INIT
# -------------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "history_eeg" not in st.session_state:
    st.session_state.history_eeg = {}

if "history_test" not in st.session_state:
    st.session_state.history_test = {}

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

if "last_self_state" not in st.session_state:
    st.session_state.last_self_state = None

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
        st.session_state.history_eeg[username] = []
        st.session_state.history_test[username] = []
        st.session_state.current_user = username
        st.sidebar.success("Account created")
    else:
        st.sidebar.error("Invalid username")

if st.session_state.current_user:
    if st.session_state.current_user not in st.session_state.history_eeg:
        st.session_state.history_eeg[st.session_state.current_user] = []

    if st.session_state.current_user not in st.session_state.history_test:
        st.session_state.history_test[st.session_state.current_user] = []

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

tab1, tab2 = st.tabs(["🧪 EEG Mode", "🧠 Self Analysis"])

# ===========================
# EEG MODE
# ===========================
with tab1:

    st.header("EEG Analysis")

    if not os.path.exists(DATA_PATH):
        st.error("Data folder not found!")
        st.stop()

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]

    if len(files) == 0:
        st.error("No .edf files found")
        st.stop()

    if st.button("🎲 Analyze Random EEG", key="eeg_btn"):
        st.session_state.selected_file = random.choice(files)

    if st.session_state.selected_file is None:
        st.info("Click to analyze a random EEG file")
        st.stop()

    file = st.session_state.selected_file
    path = os.path.join(DATA_PATH, file)
    st.success(f"Selected file: {file}")

    # PSD (çok daha doğru)
psd, freqs = mne.time_frequency.psd_array_welch(
    data,
    sfreq=raw.info['sfreq'],
    fmin=0.5,
    fmax=50,
    n_fft=1024
)

# channel power (frequency-based)
channel_power = np.mean(psd, axis=1)

st.subheader("Channel Spectral Power (first 5)")
st.write([float(x) for x in channel_power[:5]])

    colors = {
        "Calm": "#4CAF50",
        "Stressed": "#F44336",
        "Drowsy": "#FFC107",
        "Deep Relaxation": "#2196F3",
        "Neutral": "#9E9E9E",
        "Low Activity": "#3F51B5",
        "Uncertain": "#607D8B"
    }

    advice = {
        "Calm": "Balanced and stable 🌿",
        "Stressed": "High cognitive load ⚠️",
        "Drowsy": "Low alertness 😴",
        "Deep Relaxation": "Very deep calm 🧘",
        "Neutral": "No dominant state",
        "Low Activity": "Low brain activity detected",
        "Uncertain": "Signal unclear, try another sample"
    }

    st.markdown(f"""
    <div style='background:{colors[state]};
                padding:30px;
                border-radius:20px;
                text-align:center;
                color:white'>
        <h1>{state}</h1>
        <p>{advice[state]}</p>
        <h3>Confidence: {round(float(confidence),2)}</h3>
    </div>
    """, unsafe_allow_html=True)

    # SAVE HISTORY
    st.session_state.history_eeg[st.session_state.current_user].append({
        "time": datetime.now().strftime("%H:%M"),
        "state": state,
        "confidence": float(confidence),
        "file": file
    })

    # VISUALIZATION
    fig, ax = plt.subplots()
    ax.plot(np.mean(data, axis=0)[:2000])
    st.pyplot(fig)
    st.bar_chart({"delta":[delta],"theta":[theta],"alpha":[alpha],"beta":[beta]})

    # HISTORY
    st.subheader("EEG History")
    for h in st.session_state.history_eeg[st.session_state.current_user][::-1]:
        st.markdown(f"""
        <div style='padding:15px;margin:10px 0;border-radius:15px;background:#111;color:white'>
        <b>{h["state"]}</b> ({round(h["confidence"],2)})<br>
        {h["file"]}<br>
        <small>{h["time"]}</small>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# SELF ANALYSIS
# ===========================
with tab2:

    st.header("Self Mental State Analysis")

    col1, col2 = st.columns(2)

    with col1:
        stress = st.slider("Stress Level", 0, 10, 5)
        focus = st.slider("Focus Level", 0, 10, 5)

    with col2:
        energy = st.slider("Energy Level", 0, 10, 5)
        sleep = st.slider("Sleep Quality", 0, 10, 5)

    # BUTTON (KEY)
    if st.button("Analyze My State", key="self_btn"):

        stress_score = stress * 1.5
        fatigue_score = (10 - energy) + (10 - sleep)
        focus_score = focus
        total = stress_score + fatigue_score - focus_score

        if total > 15:
            state = "Highly Stressed"
        elif total > 10:
            state = "Stressed"
        elif total > 5:
            state = "Unstable"
        else:
            state = "Balanced"

        st.session_state.last_self_state = state

        st.session_state.history_test[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": state
        })

    # SHOW CARD
    if st.session_state.last_self_state:

        state = st.session_state.last_self_state

        colors = {
            "Highly Stressed": "#D32F2F",
            "Stressed": "#F57C00",
            "Unstable": "#FFC107",
            "Balanced": "#4CAF50"
        }

        advice = {
            "Highly Stressed": "Immediate rest recommended 🛑",
            "Stressed": "Slow down and breathe 🌬️",
            "Unstable": "Try to rebalance your day ⚖️",
            "Balanced": "You're doing great 🚀"
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

    # HISTORY
    st.subheader("Your Mental History")
    for h in st.session_state.history_test[st.session_state.current_user][::-1]:
        st.markdown(f"""
        <div style='padding:15px;margin:10px 0;border-radius:15px;background:#222;color:white'>
        <b>{h["state"]}</b><br>
        <small>{h["time"]}</small>
        </div>
        """, unsafe_allow_html=True)
