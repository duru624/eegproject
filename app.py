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
    if st.sidebar.button("Logout"):
        st.session_state.current_user = None

if st.session_state.current_user is None:
    st.title("🧠 NeuroPulse")
    st.stop()

st.title("🧠 NeuroPulse")
st.write("User:", st.session_state.current_user)

tab1, tab2 = st.tabs(["🧪 EEG Mode", "🧠 Self Analysis"])

# ===========================
# EEG MODE (FIXED)
# ===========================
with tab1:

    st.header("EEG Analysis")

    if not os.path.exists(DATA_PATH):
        st.error("Data folder not found!")
        st.stop()

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]

    if st.button("🎲 Analyze Random EEG"):
        st.session_state.selected_file = random.choice(files)

    if st.session_state.selected_file is None:
        st.info("Click to analyze EEG")
        st.stop()

    file = st.session_state.selected_file
    path = os.path.join(DATA_PATH, file)

    st.success(f"Selected: {file}")

    # LOAD
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(0.5, 30, verbose=False)

    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # PSD
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=30,
        n_fft=1024
    )

    # BAND POWER
    def band(fmin, fmax):
        return np.mean(np.sum(psd[:, (freqs >= fmin) & (freqs < fmax)], axis=1))

    delta = band(0.5, 4)
    theta = band(4, 8)
    alpha = band(8, 12)
    beta = band(12, 30)

    # RELATIVE
    total = delta + theta + alpha + beta
    delta, theta, alpha, beta = delta/total, theta/total, alpha/total, beta/total

    # SMART LOGIC (FIXED)
    if beta > alpha * 1.2:
        state = "Stressed"
    elif theta > alpha:
        state = "Drowsy"
    elif delta > 0.35:
        state = "Deep Relaxation"
    elif alpha > 0.3:
        state = "Calm"
    else:
        state = "Neutral"

    # UI
    colors = {
        "Calm": "#4CAF50",
        "Stressed": "#F44336",
        "Drowsy": "#FFC107",
        "Deep Relaxation": "#2196F3",
        "Neutral": "#9E9E9E"
    }

    advice = {
        "Calm": "Balanced 🌿",
        "Stressed": "Take a breath 🫁",
        "Drowsy": "You need rest 😴",
        "Deep Relaxation": "Deep calm 🧘",
        "Neutral": "Stable state"
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

    # SAVE HISTORY
    st.session_state.history_eeg.setdefault(st.session_state.current_user, [])
    st.session_state.history_eeg[st.session_state.current_user].append({
        "time": datetime.now().strftime("%H:%M"),
        "state": state
    })

    # GRAPH
    fig, ax = plt.subplots()
    ax.plot(np.mean(data, axis=0)[:2000])
    st.pyplot(fig)

    st.bar_chart({
        "delta":[delta],
        "theta":[theta],
        "alpha":[alpha],
        "beta":[beta]
    })

    # HISTORY (FIXED)
    st.subheader("EEG History")

    history = st.session_state.history_eeg[st.session_state.current_user]

    if len(history) == 0:
        st.info("No history yet")
    else:
        for h in history[::-1]:
            st.markdown(f"""
            <div style='padding:15px;
                        margin:10px 0;
                        border-radius:12px;
                        background:linear-gradient(135deg,#1f1f1f,#2a2a2a);
                        color:white'>
                <b>{h["state"]}</b>
                <div style='font-size:12px;color:#aaa'>{h["time"]}</div>
            </div>
            """, unsafe_allow_html=True)

# ===========================
# SELF ANALYSIS (PRETTY)
# ===========================
with tab2:

    st.header("Self Mental State Analysis")

    col1, col2 = st.columns(2)

    with col1:
        stress = st.slider("Stress", 0, 10, 5)
        focus = st.slider("Focus", 0, 10, 5)

    with col2:
        energy = st.slider("Energy", 0, 10, 5)
        sleep = st.slider("Sleep", 0, 10, 5)

    if st.button("Analyze", key="self_btn"):

        score = stress*1.5 + (10-energy) + (10-sleep) - focus

        if score > 15:
            state = "Highly Stressed"
        elif score > 10:
            state = "Stressed"
        elif score > 5:
            state = "Unstable"
        else:
            state = "Balanced"

        st.session_state.last_self_state = state

        st.session_state.history_test.setdefault(st.session_state.current_user, [])
        st.session_state.history_test[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": state
        })

    if st.session_state.last_self_state:

        state = st.session_state.last_self_state

        colors = {
            "Highly Stressed": "#D32F2F",
            "Stressed": "#F57C00",
            "Unstable": "#FFC107",
            "Balanced": "#4CAF50"
        }

        advice = {
            "Highly Stressed": "Immediate rest 🛑",
            "Stressed": "Slow down",
            "Unstable": "Rebalance",
            "Balanced": "Great 🚀"
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

    # HISTORY (PRETTY FIX)
    st.subheader("Your Mental History")

    history = st.session_state.history_test[st.session_state.current_user]

    if len(history) == 0:
        st.info("No history yet")
    else:
        for h in history[::-1]:
            st.markdown(f"""
            <div style='padding:15px;
                        margin:10px 0;
                        border-radius:15px;
                        background:linear-gradient(135deg,#2c2c2c,#3a3a3a);
                        color:white;
                        display:flex;
                        justify-content:space-between'>
                <span><b>{h["state"]}</b></span>
                <span style='color:#bbb'>{h["time"]}</span>
            </div>
            """, unsafe_allow_html=True)
