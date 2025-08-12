
import os
import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Auto-refresh every 5 seconds (keeps dashboard "live" for judges)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="auto_refresh")
except Exception:
    # If package missing, ignore auto-refresh (app still works)
    pass

st.set_page_config(page_title='TPPR AI Safety Assistant', layout='wide')

st.title("🚨 TPPR AI Safety Assistant")
st.markdown(
    """
    **Purpose:** Demo of an AI layer that augments Honeywell Touchpoint Pro data with anomaly detection,
    short-term forecasting, alert ranking, and an interactive operator dashboard.
    """
)

# Determine CSV path reliably
CSV_NAME = "tppr_simulated.csv"
CSV_PATH = os.path.join(os.path.dirname(__file__), CSV_NAME)

# Load CSV if present, otherwise generate a realistic fallback dataset
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
else:
    # Fallback synthetic dataset
    now = pd.Timestamp("2025-08-12 14:00:00")
    minutes = 240  # 4 hours
    rows = []
    channels = [
        {"id": 1, "gas": "CH4", "threshold": 100},
        {"id": 2, "gas": "H2S", "threshold": 50},
        {"id": 3, "gas": "CO", "threshold": 200},
    ]
    np.random.seed(42)
    for i in range(minutes):
        ts = now + pd.Timedelta(minutes=i)
        for ch in channels:
            baseline = {"CH4": 25, "H2S": 5, "CO": 2}[ch["gas"]]
            value = baseline + np.random.normal(0, baseline*0.05)
            rows.append({
                "timestamp": ts,
                "channel": ch["id"],
                "gas_type": ch["gas"],
                "gas_level_ppm": round(float(value), 2),
                "alarm_state": 0,
                "fault_state": 0,
                "sensor_status": "OK",
                "calibration_date": "2025-06-10"
            })
    df = pd.DataFrame(rows)
    # inject a slow ramp leak on CH4 and some spikes for demo
    def apply_ramp(df, ch_id, start_min, dur, peak):
        mask = (df['channel']==ch_id)
        idxs = df[mask].index[start_min:start_min+dur]
        for i, idx in enumerate(idxs):
            df.at[idx, 'gas_level_ppm'] += (i/len(idxs))*peak
    apply_ramp(df, 1, 60, 40, 120)
    # inject spikes
    for m in [30, 90, 150]:
        row = df[(df['channel']==1)].iloc[m:m+1]
        if not row.empty:
            idx = row.index[0]
            df.at[idx, 'gas_level_ppm'] += 80
    # set alarm states where threshold exceeded
    thresholds = {1:100, 2:50, 3:200}
    for ch_id, thr in thresholds.items():
        ch_mask = df['channel']==ch_id
        df.loc[ch_mask & (df['gas_level_ppm'] >= thr), 'alarm_state'] = 1

# Ensure timestamp dtype
if df['timestamp'].dtype == object:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar controls
st.sidebar.header("Controls")
danger_threshold = st.sidebar.slider('Danger threshold (ppm)', min_value=0, max_value=500, value=50, step=1)
play_live_feed = st.sidebar.button("▶ Start Live Feed Simulation")
speed = st.sidebar.selectbox("Feed speed (delay sec per step)", [0.1, 0.25, 0.5, 1.0], index=1)

# Prepare layout
col_main, col_side = st.columns([3,1])

# Alert history initialization
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

# Current latest readings
latest_readings = df.tail(1)

# Live AI Alert panel
with col_side:
    alert_placeholder = st.empty()
    numeric_latest = latest_readings.select_dtypes(include=[np.number])
    is_danger = False
    if not numeric_latest.empty and (numeric_latest > danger_threshold).any().any():
        is_danger = True
        alert_placeholder.error(f"🚨 DANGER: Gas above {danger_threshold} ppm detected!")
    else:
        alert_placeholder.success(f"✅ All clear (≤ {danger_threshold} ppm)")

    # Update alert history
    latest_time = latest_readings['timestamp'].iloc[0] if 'timestamp' in latest_readings else 'Unknown time'
    if is_danger:
        st.session_state.alert_history.append(f"🚨 {latest_time} — Gas > {danger_threshold} ppm")
    else:
        st.session_state.alert_history.append(f"✅ {latest_time} — Safe")

    st.sidebar.title("Alert History (most recent)")
    for item in reversed(st.session_state.alert_history[-20:]):
        st.sidebar.write(item)

# Main charts area
with col_main:
    st.subheader("Channel Readings & Forecasts")
    channels = sorted(df['channel'].unique())
    channel_map = {1: "CH4 (methane)", 2: "H2S", 3: "CO"}
    thresholds = {1:100, 2:50, 3:200}

    # Play live feed if requested
    if play_live_feed:
        for i in range(len(df)):
            snapshot = df.iloc[:i+1].copy()
            # create small plotting area
            fig, axs = plt.subplots(len(channels), 1, figsize=(10, 3*len(channels)), sharex=True)
            forecasts = []
            for j, ch in enumerate(channels):
                ch_data = snapshot[snapshot['channel']==ch].copy()
                ch_data = ch_data.set_index('timestamp').resample('1T').mean(numeric_only=True).ffill()
                series = ch_data['gas_level_ppm']
                axs[j].plot(series.index, series.values, label='Measured')
                axs[j].set_ylabel("ppm")
                # mark TPPR alarms
                alarm_points = series[series >= thresholds[ch]]
                if not alarm_points.empty:
                    axs[j].scatter(alarm_points.index, alarm_points.values, marker='x', color='red', zorder=5)
                # anomaly detection (simple z-score)
                is_anom = False
                if len(series) >= 5:
                    window = min(12, len(series))
                    recent = series.iloc[-window:]
                    mean = recent.mean()
                    std = recent.std(ddof=0)
                    if std > 0:
                        z = (series.iloc[-1] - mean) / std
                        if abs(z) >= 3.0:
                            is_anom = True
                            axs[j].axvline(series.index[-1], color='orange', linestyle='--', alpha=0.6)
                # forecast: linear fit on last up to 20 samples
                try:
                    recent = series.dropna().iloc[-20:]
                    if len(recent) >= 3:
                        x = np.arange(len(recent))
                        y = recent.values.astype(float)
                        coef = np.polyfit(x, y, 1)
                        m, b = coef[0], coef[1]
                        future_x = np.arange(len(recent), len(recent)+5)
                        future_y = m*future_x + b
                        last_time = series.index[-1]
                        future_index = [last_time + pd.Timedelta(minutes=int(k)) for k in range(1,6)]
                        axs[j].plot(future_index, future_y, linestyle='--', marker='o', label='Predicted')
                        forecasts.append((ch, channel_map.get(ch, str(ch)), future_index, future_y))
                except Exception:
                    pass
                axs[j].legend(loc='upper left')
            st.pyplot(fig)
            # show forecast table if available
            if forecasts:
                rows = []
                for f in forecasts:
                    for t, v in zip(f[2], f[3]):
                        rows.append({'channel': f[0], 'gas': f[1], 'pred_time': t, 'pred_ppm': round(float(v),2)})
                st.write("**Short-term per-channel forecast (next 5 minutes)**")
                st.dataframe(pd.DataFrame(rows))
            time.sleep(speed)
        st.experimental_rerun()

    # Non-playback (static view of latest snapshot)
    snapshot = df.copy()
    latest_snapshot = snapshot.tail(len(channels)*4)  # recent chunk
    fig, axs = plt.subplots(len(channels), 1, figsize=(10, 3*len(channels)), sharex=True)
    forecasts = []
    for j, ch in enumerate(channels):
        ch_data = snapshot[snapshot['channel']==ch].copy()
        ch_data = ch_data.set_index('timestamp').resample('1T').mean(numeric_only=True).ffill()
        series = ch_data['gas_level_ppm']
        axs[j].plot(series.index, series.values, label='Measured')
        axs[j].set_ylabel("ppm")
        alarm_points = series[series >= thresholds[ch]]
        if not alarm_points.empty:
            axs[j].scatter(alarm_points.index, alarm_points.values, marker='x', color='red', zorder=5)
        # anomaly detection
        if len(series) >= 5:
            window = min(12, len(series))
            recent = series.iloc[-window:]
            mean = recent.mean()
            std = recent.std(ddof=0)
            if std > 0:
                z = (series.iloc[-1] - mean) / std
                if abs(z) >= 3.0:
                    axs[j].axvline(series.index[-1], color='orange', linestyle='--', alpha=0.6)
        # forecast per-channel
        try:
            recent = series.dropna().iloc[-20:]
            if len(recent) >= 3:
                x = np.arange(len(recent))
                y = recent.values.astype(float)
                coef = np.polyfit(x, y, 1)
                m, b = coef[0], coef[1]
                future_x = np.arange(len(recent), len(recent)+5)
                future_y = m*future_x + b
                last_time = series.index[-1]
                future_index = [last_time + pd.Timedelta(minutes=int(k)) for k in range(1,6)]
                axs[j].plot(future_index, future_y, linestyle='--', marker='o', label='Predicted')
                forecasts.append((ch, channel_map.get(ch, str(ch)), future_index, future_y))
        except Exception:
            pass
        axs[j].legend(loc='upper left')
    st.pyplot(fig)
    if forecasts:
        rows = []
        for f in forecasts:
            for t, v in zip(f[2], f[3]):
                rows.append({'channel': f[0], 'gas': f[1], 'pred_time': t, 'pred_ppm': round(float(v),2)})
        st.write("**Short-term per-channel forecast (next 5 minutes)**")
        st.dataframe(pd.DataFrame(rows))

# Footer / notes
st.markdown("---")
st.markdown("**Notes:** This demo runs with a simulated dataset if a CSV isn't present. For full integration,"
            " TPPR would stream Modbus/RS-485 or Modbus/TCP data to the AI unit which would run the same"
            " analytics shown here.")
