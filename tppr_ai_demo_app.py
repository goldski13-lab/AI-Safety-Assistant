
import os, time, io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta
import soundfile as sf
import sounddevice as sd

st.set_page_config(page_title="TPPR AI Assistant — UX Upgrade", layout="wide")
st.title("🚨 TPPR AI Safety Assistant — UX Upgrade")

# Load or simulate data
CSV_NAME = "tppr_simulated.csv"
CSV_PATH = os.path.join(os.path.dirname(__file__), CSV_NAME)
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
else:
    now = pd.Timestamp.now().floor('min')
    minutes = 240
    rows = []
    channels = [{"id":1,"gas":"CH4","thr":100},{"id":2,"gas":"H2S","thr":50},{"id":3,"gas":"CO","thr":200}]
    np.random.seed(42)
    for i in range(minutes):
        ts = now + pd.Timedelta(minutes=i)
        for ch in channels:
            base = {"CH4":25,"H2S":5,"CO":2}[ch["gas"]]
            val = base + np.random.normal(0, base*0.05)
            rows.append({"timestamp":ts,"channel":ch["id"],"gas_type":ch["gas"],"gas_level_ppm":round(float(val),2),"alarm_state":0})
    df = pd.DataFrame(rows)
    # inject demo event
    def ramp(ch_id,start,dur,peak):
        mask = df['channel']==ch_id
        idxs = df[mask].index[start:start+dur]
        for i,idx in enumerate(idxs):
            df.at[idx,'gas_level_ppm'] += (i/len(idxs))*peak
    ramp(1,60,40,120)
    for m in [30,90,150]:
        row = df[(df['channel']==1)].iloc[m:m+1]
        if not row.empty:
            idx = row.index[0]
            df.at[idx,'gas_level_ppm'] += 80
    thr = {1:100,2:50,3:200}
    for ch, t in thr.items():
        chmask = df['channel']==ch
        df.loc[chmask & (df['gas_level_ppm']>=t), 'alarm_state'] = 1

# Ensure timestamp
if df['timestamp'].dtype == object:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar controls (grouped)
st.sidebar.header("Display Settings")
thresholds_global = st.sidebar.slider("Global danger threshold (ppm)", 0, 500, 100)
accessibility = st.sidebar.checkbox("High-contrast (accessibility) mode", value=False)
st.sidebar.markdown("**Simulation Controls**")
play = st.sidebar.button("▶ Start Live Feed")
speed = st.sidebar.selectbox("Feed speed (s per step)", [0.1,0.25,0.5,1.0], index=1)

channel_map = {1:"CH4 (methane)", 2:"H2S", 3:"CO"}
thresholds = {1:100, 2:50, 3:200}

# Latest per-channel values
latest = df.groupby('channel').tail(1).set_index('channel')
channel_latest = {ch: float(latest.loc[ch,'gas_level_ppm']) if ch in latest.index else None for ch in sorted(df['channel'].unique())}

# Compute per-channel status
per_status = {}
worst_status = ('safe', 0, None)
for ch, val in channel_latest.items():
    thr = thresholds.get(ch, thresholds_global)
    if val is None:
        st = ('nodata', 0)
    else:
        pct = val / thr if thr>0 else 0
        if val >= thr:
            st = ('critical', pct)
        elif pct >= 0.75:
            st = ('warning', pct)
        else:
            st = ('safe', pct)
    per_status[ch] = {'value': val, 'status': st[0], 'pct': round(float(st[1]),2), 'threshold': thr}
    if st[1] > worst_status[1]:
        worst_status = (st[0], st[1], ch)

# Top banner
overall = worst_status[0]
if overall == 'critical':
    st.markdown(f"<div style='background:#ff4c4c;padding:12px;border-radius:8px;color:white'><h2>🚨 CRITICAL — {channel_map.get(worst_status[2])} {per_status[worst_status[2]]['value']} ppm (≥ {per_status[worst_status[2]]['threshold']} ppm)</h2></div>", unsafe_allow_html=True)
    sound = True
elif overall == 'warning':
    st.markdown(f"<div style='background:#ffb84d;padding:12px;border-radius:8px;color:#3a2d0f'><h2>🟠 WARNING — {channel_map.get(worst_status[2])} {per_status[worst_status[2]]['value']} ppm (close to {per_status[worst_status[2]]['threshold']} ppm)</h2></div>", unsafe_allow_html=True)
    sound = False
else:
    st.markdown(f"<div style='background:#d4f0d4;padding:12px;border-radius:8px;color:#0f3a14'><h2>✅ SAFE — All channels within normal range</h2></div>", unsafe_allow_html=True)
    sound = False

# Play short beep for critical (using embedded WAV bytes if available)
if overall == 'critical':
    # generate a short sine beep and provide as downloadable audio and play via st.audio
    sr = 22050
    t = np.linspace(0,0.4,int(sr*0.4), False)
    freq = 880.0
    sine = 0.5*np.sin(2*np.pi*freq*t)
    # convert to 16-bit PCM bytes
    import soundfile as sf, io
    buf = io.BytesIO()
    sf.write(buf, sine, sr, format='WAV')
    buf.seek(0)
    st.audio(buf.read())

# Per-channel metric cards
cols = st.columns(len(per_status))
for i, ch in enumerate(sorted(per_status.keys())):
    p = per_status[ch]
    val = p['value'] if p['value'] is not None else '—'
    label = channel_map.get(ch, str(ch))
    delta = 0
    try:
        # compute delta vs previous 1-min
        prev = df[(df['channel']==ch)].tail(2)['gas_level_ppm'].iloc[0]
        delta = round(p['value'] - prev,2) if p['value'] is not None else 0
    except Exception:
        delta = 0
    cols[i].metric(label, f\"{val} ppm\", delta= f\"{delta} ppm\" if delta!=0 else \"—\")
    # small status text
    cols[i].markdown(f\"**Status:** {p['status'].upper()} (Threshold {p['threshold']} ppm)\")

# Help / guided tour
with st.expander(\"Quick guided tour (recommended for non-experts)\", expanded=False):
    st.write(\"1. Top banner shows immediate overall AI assessment.\\n2. Metric cards show each gas numeric value and trend.\\n3. Charts below show history + predictions.\\n4. Use the slider and live feed to simulate behaviour.\")

# Action guidance box
if overall in ['critical','warning']:
    st.markdown(\"### Recommended actions\")
    if overall == 'critical':
        st.warning(\"Immediate actions: 1) Evacuate affected area. 2) Isolate main valve. 3) Notify safety officer.\\nPress 'Acknowledge' when done.\")
    else:
        st.info(\"Precautionary actions: Check ventilation, inspect nearby equipment, stand by.\")
    op = st.text_input(\"Operator name (optional, to log acknowledgement)\", value=\"\")
    ack = st.button(\"Acknowledge\")
    if ack:
        if 'ack_log' not in st.session_state:
            st.session_state.ack_log = []
        st.session_state.ack_log.append({'time':pd.Timestamp.now(), 'operator':op, 'status':overall, 'channel': worst_status[2]})
        st.success(\"Acknowledged and logged.\")

# Main plotting area
st.markdown(\"---\")
st.subheader(\"Channel trends and forecasts\")
# choose channels to show
show = st.multiselect(\"Select channels to display\", options=sorted(df['channel'].unique()), default=sorted(df['channel'].unique()), format_func=lambda x: channel_map.get(x, str(x)))
display_df = df[df['channel'].isin(show)].copy()

# Aggregate for heatmap / alert history visualization
display_df['hour'] = display_df['timestamp'].dt.floor('H')
alert_counts = display_df[display_df['alarm_state']==1].groupby(['hour','channel']).size().unstack(fill_value=0)

# Heatmap of alerts per hour
st.markdown(\"#### Alert heatmap (alerts per hour)\")
if not alert_counts.empty:
    fig_h, axh = plt.subplots(figsize=(8,2))
    im = axh.imshow(alert_counts.T, aspect='auto', cmap='Reds')
    axh.set_yticks(np.arange(len(alert_counts.columns)))
    axh.set_yticklabels([channel_map.get(c) for c in alert_counts.columns])
    axh.set_xticks(np.arange(len(alert_counts.index)))
    axh.set_xticklabels([t.strftime('%H:%M') for t in alert_counts.index], rotation=45)
    plt.tight_layout()
    st.pyplot(fig_h)
else:
    st.write(\"No alarms in selected range.\")

# For each channel, plot recent trend and short forecast
for ch in show:
    ch_data = display_df[display_df['channel']==ch].set_index('timestamp').resample('1T').mean(numeric_only=True).ffill()
    if ch_data.empty:
        continue
    series = ch_data['gas_level_ppm']
    fig, ax = plt.subplots(figsize=(8,2.5))
    ax.plot(series.index, series.values, label='Measured')
    # shaded zones
    thr = thresholds.get(ch, thresholds_global)
    ax.fill_between(series.index, thr*0.75, thr, color='orange', alpha=0.1, label='Warning zone')
    ax.fill_between(series.index, thr, thr*2, color='red', alpha=0.06, label='Critical zone')
    # anomaly markers
    if len(series) >= 5:
        window = min(12, len(series))
        recent = series.iloc[-window:]
        m = recent.mean(); s = recent.std(ddof=0)
        if s > 0 and abs(series.iloc[-1]-m)/s >= 3.0:
            ax.axvline(series.index[-1], color='purple', linestyle='--', label='Anomaly')
    # forecast linear
    try:
        recent = series.dropna().iloc[-20:]
        if len(recent) >= 3:
            x = np.arange(len(recent)); y = recent.values.astype(float)
            coef = np.polyfit(x,y,1); m,b = coef[0],coef[1]
            future_x = np.arange(len(recent), len(recent)+5)
            future_y = m*future_x + b
            last_t = series.index[-1]
            future_idx = [last_t + pd.Timedelta(minutes=int(k)) for k in range(1,6)]
            ax.plot(future_idx, future_y, linestyle='--', marker='o', label='Predicted')
            # calculate minutes to threshold estimate
            if m>0:
                mins_to_thr = (thresholds.get(ch, thresholds_global) - series.iloc[-1]) / m
            else:
                mins_to_thr = None
        else:
            mins_to_thr = None
    except Exception:
        mins_to_thr = None
    ax.set_title(f\"{channel_map.get(ch)} — {series.iloc[-1]:.1f} ppm\" )
    ax.set_ylabel('ppm')
    ax.legend(loc='upper left', fontsize=8)
    st.pyplot(fig)
    if mins_to_thr is not None and mins_to_thr>0 and mins_to_thr<9999:
        st.write(f\"Estimated time to threshold: ~{max(0,int(round(mins_to_thr)))} minutes (approx)\")

# Export PDF report of current figures & alert history
st.markdown('---')
st.subheader('Export / Report')
buffer = io.BytesIO()
with PdfPages(buffer) as pdf:
    # simple: write a page with metrics snapshot and a chart image for each channel
    # metrics snapshot
    fig_m, axm = plt.subplots(figsize=(8,2))
    text = 'Metrics snapshot:\\n' + '\\n'.join([f\"{channel_map[c]}: {per_status[c]['value']} ppm ({per_status[c]['status']})\" for c in per_status])
    axm.text(0.01,0.5, text, fontsize=12)
    axm.axis('off')
    pdf.savefig(fig_m)
    plt.close(fig_m)
    # small chart per channel
    for ch in sorted(df['channel'].unique()):
        chd = df[df['channel']==ch].set_index('timestamp').resample('1T').mean(numeric_only=True).ffill()
        if chd.empty: 
            continue
        figc, axc = plt.subplots(figsize=(8,2))
        axc.plot(chd.index, chd['gas_level_ppm'])
        axc.set_title(channel_map.get(ch))
        pdf.savefig(figc)
        plt.close(figc)
buffer.seek(0)
st.download_button('Download incident report (PDF)', data=buffer, file_name='tppr_report.pdf', mime='application/pdf')

st.markdown('---')
st.write('Operator acknowledgement log:')
st.write(st.session_state.get('ack_log', []))
