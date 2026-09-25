import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import websockets
import asyncio
import json
from datetime import datetime
import threading

st.set_page_config(page_title="Physio Monitor Dashboard", layout="wide")

st.sidebar.title("Settings")
ws_url = st.sidebar.text_input("Backend WS URL", "ws://localhost:8000/ws/dashboard")

if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []

if 'running' not in st.session_state:
    st.session_state.running = False

def ws_thread_func(url):
    async def listen():
        try:
            async with websockets.connect(url) as ws:
                while st.session_state.running:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    st.session_state.data_buffer.append(data)
                    # Keep only last 300
                    if len(st.session_state.data_buffer) > 300:
                        st.session_state.data_buffer = st.session_state.data_buffer[-300:]
        except Exception as e:
            print(f"WS Error: {e}")
            st.session_state.running = False

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen())

col1, col2 = st.sidebar.columns(2)
if col1.button("Start"):
    if not st.session_state.running:
        st.session_state.running = True
        threading.Thread(target=ws_thread_func, args=(ws_url,), daemon=True).start()
if col2.button("Stop"):
    st.session_state.running = False

if st.sidebar.button("Export to CSV"):
    if st.session_state.data_buffer:
        df = pd.json_normalize(st.session_state.data_buffer)
        st.sidebar.download_button(
            "Download CSV",
            df.to_csv(index=False).encode('utf-8'),
            "telemetry.csv",
            "text/csv"
        )

st.title("Physio Monitor Dashboard")

# Top metrics
m1, m2, m3 = st.columns(3)
if st.session_state.data_buffer:
    latest = st.session_state.data_buffer[-1]
    metrics = latest['metrics']
    prev_metrics = st.session_state.data_buffer[-2]['metrics'] if len(st.session_state.data_buffer) > 1 else metrics
    
    m1.metric("Heart Rate", f"{metrics['heart_rate']:.1f} bpm", f"{metrics['heart_rate'] - prev_metrics['heart_rate']:.1f}")
    m2.metric("SpO2", f"{metrics['spo2']:.1f} %", f"{metrics['spo2'] - prev_metrics['spo2']:.1f}")
    m3.metric("Temperature", f"{metrics['temperature']:.1f} °C", f"{metrics['temperature'] - prev_metrics['temperature']:.1f}")
else:
    m1.metric("Heart Rate", "--")
    m2.metric("SpO2", "--")
    m3.metric("Temperature", "--")

# Chart
if st.session_state.data_buffer:
    df = pd.json_normalize(st.session_state.data_buffer)
    # create timestamps for x axis assuming ~1s interval if not provided properly
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['metrics.heart_rate'], mode='lines', name='HR'))
    fig.add_trace(go.Scatter(y=df['metrics.spo2'], mode='lines', name='SpO2'))
    fig.add_trace(go.Scatter(y=df['metrics.temperature'], mode='lines', name='Temp'))
    fig.update_layout(title="Vital Signs Over Time", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Bottom row
b1, b2, b3 = st.columns(3)
with b1:
    st.subheader("Diagnostics")
    if st.session_state.data_buffer:
        diag = latest['diagnostics']
        st.progress(diag['sqa_score'], text=f"SQA Score: {diag['sqa_score']:.2f}")
        st.write(f"Anomaly Score: {diag.get('anomaly_score', 0):.3f}")

with b2:
    st.subheader("Status")
    if st.session_state.data_buffer:
        diag = latest['diagnostics']
        if diag['sqa_status'] != "GOOD":
            st.error("CEK SENSOR")
        elif diag.get('is_anomaly'):
            st.warning("ANOMALI")
        else:
            st.success("NORMAL")

with b3:
    st.subheader("System")
    if st.session_state.data_buffer:
        sys_info = latest['system']
        st.write(f"Device: {sys_info['device_id']}")
        st.write(f"RSSI: {sys_info['rssi']} dBm")
        st.write("Connection: Active" if st.session_state.running else "Inactive")

# Auto-refresh workaround using st_autorefresh or rerun logic
if st.session_state.running:
    import time
    time.sleep(1)
    st.rerun()
