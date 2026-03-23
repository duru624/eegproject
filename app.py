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
st.set_page_config(page_title="NeuroPulse AI", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
for key in ["users","current_user","history_eeg","history_test","selected_file","last_self_state","last_eeg_state"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "history" in key or key=="users" else None

# -------------------------
# AUTH
# -------------------------
st.sidebar.title("Account")
username = st.sidebar.text_input("Username")

c1,c2 = st.sidebar.columns(2)

if c1.button("Login"):
    if username in st.session_state.users:
        st.session_state.current_user = username
        st.sidebar.success("Logged in")
    else:
        st.sidebar.error("User not found")

if c2.button("Register"):
    if username and username not in st.session_state.users:
        st.session_state.users[username]=[]
        st.session_state.history_eeg[username]=[]
        st.session_state.history_test[username]=[]
        st.session_state.current_user=username
        st.sidebar.success("Account created")

if st.session_state.current_user is None:
    st.title("🧠 NeuroPulse AI")
    st.write("Mental State Detection Without Words")
    st.stop()

user = st.session_state.current_user

st.title("🧠 NeuroPulse AI")
st.write("User:", user)

tab1,tab2,tab3 = st.tabs(["🧪 EEG","🧠 Self","🤖 AI Fusion"])

# ===========================
# EEG MODE
# ===========================
with tab1:

    st.header("EEG Analysis")

    files=[f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]

    if st.button("🎲 Analyze EEG"):
        st.session_state.selected_file=random.choice(files)

    if st.session_state.selected_file:

        path=os.path.join(DATA_PATH,st.session_state.selected_file)
        raw=mne.io.read_raw_edf(path,preload=True,verbose=False)
        raw.filter(0.5,30,verbose=False)

        data=raw.get_data()
        sfreq=raw.info['sfreq']

        psd,freqs=mne.time_frequency.psd_array_welch(
            data,sfreq=sfreq,fmin=0.5,fmax=30,n_fft=1024)

        def band(f1,f2):
            return np.mean(np.sum(psd[:,(freqs>=f1)&(freqs<f2)],axis=1))

        delta,theta,alpha,beta=band(0.5,4),band(4,8),band(8,12),band(12,30)

        total=delta+theta+alpha+beta
        delta,theta,alpha,beta=delta/total,theta/total,alpha/total,beta/total

        if beta>alpha*1.2: state="Stressed"
        elif theta>alpha: state="Drowsy"
        elif delta>0.35: state="Deep"
        elif alpha>0.3: state="Calm"
        else: state="Neutral"

        st.session_state.last_eeg_state=state

        st.session_state.history_eeg.setdefault(user,[])
        st.session_state.history_eeg[user].append({"time":datetime.now().strftime("%H:%M"),"state":state})

        color={"Calm":"#4CAF50","Stressed":"#F44336","Drowsy":"#FFC107","Deep":"#2196F3","Neutral":"#9E9E9E"}[state]

        st.markdown(f"""
        <div style='background:{color};padding:30px;border-radius:20px;color:white;text-align:center'>
        <h1>{state}</h1>
        </div>
        """,unsafe_allow_html=True)

        st.line_chart(np.mean(data,axis=0)[:2000])

    # HISTORY
    st.subheader("EEG Timeline")
    hist=st.session_state.history_eeg.get(user,[])
    for h in hist[::-1]:
        st.markdown(f"<div style='padding:12px;margin:8px;background:#222;border-radius:12px;color:white'>{h['state']} — {h['time']}</div>",unsafe_allow_html=True)

# ===========================
# SELF ANALYSIS
# ===========================
with tab2:

    st.header("Self Analysis")

    stress=st.slider("Stress",0,10,5)
    focus=st.slider("Focus",0,10,5)
    energy=st.slider("Energy",0,10,5)
    sleep=st.slider("Sleep",0,10,5)

    if st.button("Analyze",key="self"):

        score=stress*1.5+(10-energy)+(10-sleep)-focus

        if score>15: state="Highly Stressed"
        elif score>10: state="Stressed"
        elif score>5: state="Unstable"
        else: state="Balanced"

        st.session_state.last_self_state=state

        st.session_state.history_test.setdefault(user,[])
        st.session_state.history_test[user].append({"time":datetime.now().strftime("%H:%M"),"state":state})

    if st.session_state.last_self_state:

        state=st.session_state.last_self_state

        color={"Highly Stressed":"#D32F2F","Stressed":"#F57C00","Unstable":"#FFC107","Balanced":"#4CAF50"}[state]

        st.markdown(f"""
        <div style='background:{color};padding:30px;border-radius:20px;color:white;text-align:center'>
        <h1>{state}</h1>
        </div>
        """,unsafe_allow_html=True)

    st.subheader("Mental Timeline")
    hist=st.session_state.history_test.get(user,[])
    for h in hist[::-1]:
        st.markdown(f"<div style='padding:12px;margin:8px;background:#333;border-radius:12px;color:white'>{h['state']} — {h['time']}</div>",unsafe_allow_html=True)

# ===========================
# AI FUSION (WINNING PART)
# ===========================
with tab3:

    st.header("AI Mental Fusion")

    eeg=st.session_state.last_eeg_state
    self_s=st.session_state.last_self_state

    if eeg and self_s:

        if eeg=="Stressed" and "Stressed" in self_s:
            result="Critical Stress"
        elif eeg=="Calm" and self_s=="Balanced":
            result="Optimal State"
        elif eeg!=self_s:
            result="Mismatch"
        else:
            result="Moderate"

        st.markdown(f"""
        <div style='background:#111;padding:40px;border-radius:25px;color:white;text-align:center'>
        <h1>{result}</h1>
        <p>Brain vs Self comparison</p>
        </div>
        """,unsafe_allow_html=True)

    else:
        st.info("Run EEG and Self Analysis first")
