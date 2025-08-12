
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import time, random

# Page config
st.set_page_config(page_title="TPPR AI — 3D Enhanced Lab", layout="wide")
st.title("TPPR AI Safety Assistant — 3D Enhanced Lab Twin")

# Rooms (top-down layout)
ROOMS = [
    {"id": "mixing", "name": "Mixing Area",    "x": 0, "y": 0},
    {"id": "pack",   "name": "Packaging Line", "x": 4, "y": 0},
    {"id": "boiler", "name": "Boiler Room",    "x": 0, "y": 4},
    {"id": "waste",  "name": "Waste Treatment","x": 4, "y": 4},
]

DEFAULT_ROOM_COLOR = "#b3d9ff"  # light blue
ROOM_COLORS = {r["id"]: DEFAULT_ROOM_COLOR for r in ROOMS}

# Initialize session state
if "df" not in st.session_state:
    now = pd.Timestamp.now().floor("min")
    rows = []
    for minute in range(120):  # 2 hours baseline
        ts = now + pd.Timedelta(minutes=minute)
        for r in ROOMS:
            base = {"mixing":25,"pack":10,"boiler":5,"waste":8}[r["id"]]
            val = base + np.random.normal(0, base*0.03)
            rows.append({"timestamp": ts, "room": r["id"], "ppm": round(float(val),2)})
    st.session_state.df = pd.DataFrame(rows)
    st.session_state.selected_room = None
    st.session_state.room_colors = ROOM_COLORS.copy()
    st.session_state.sim_history = []

df = st.session_state.df

# Layout: top metrics row then main columns
st.markdown("### Live Metrics")
metric_cols = st.columns(len(ROOMS))
latest = df.groupby("room").tail(1).set_index("room")['ppm'].to_dict()
for i, r in enumerate(ROOMS):
    val = latest.get(r["id"], 0)
    prev = df[df["room"]==r["id"]].tail(2)
    delta = ""
    if len(prev) >= 2:
        delta = f"{round(val - prev['ppm'].iloc[0],2)} ppm"
    metric_cols[i].metric(label=r["name"], value=f"{val} ppm", delta=delta)

st.markdown("---")
left, center, right = st.columns([1.2, 2.4, 1.4])

with left:
    st.header("Controls")
    if st.button("Simulate Live Gas Event"):
        # random room and severity
        room = random.choice(ROOMS)
        severity = random.choices(["warning","critical"], weights=[0.6,0.4])[0]
        dur = 6 if severity=="warning" else 12
        peak = 80 if severity=="warning" else 160
        last_ts = st.session_state.df['timestamp'].max()
        start = last_ts + pd.Timedelta(minutes=1)
        new_rows = []
        for i in range(dur):
            ts = start + pd.Timedelta(minutes=i)
            for rr in ROOMS:
                base = {"mixing":25,"pack":10,"boiler":5,"waste":8}[rr["id"]]
                val = base + np.random.normal(0, base*0.03)
                if rr["id"] == room["id"]:
                    val += (i/dur)*peak + np.random.normal(0,5)
                new_rows.append({"timestamp": ts, "room": rr["id"], "ppm": round(float(val),2)})
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)
        # set glow color and record in history
        glow = "#ffb84d" if severity=="warning" else "#ff4c4c"
        st.session_state.room_colors[room["id"]] = glow
        st.session_state.selected_room = room["id"]
        st.session_state.sim_history.append({"time": pd.Timestamp.now(), "room": room["id"], "severity": severity})
        # rerun so plot updates and we show zoom
        st.experimental_rerun()

    if st.button("Reset Simulation Data"):
        # clear and recreate baseline
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("**Simulation history** (most recent first)")
    hist = list(reversed(st.session_state.sim_history[-10:]))
    for h in hist:
        st.write(f"- {h['time'].strftime('%H:%M:%S')} — {h['room']} — {h['severity']}")

    st.markdown("---")
    st.write("Tips: Click a room on the 3D floorplan to zoom in and inspect detector predictions. Click again to zoom out.")

# Build 3D floorplan as simple boxes + invisible scatter markers for clicks
mesh_traces = []
marker_x = []
marker_y = []
marker_z = []
marker_text = []
marker_room_ids = []

for r in ROOMS:
    x0, y0 = r["x"], r["y"]
    size = 2.0
    # vertices for a flat box with slight height
    vx = [x0, x0+size, x0+size, x0, x0, x0+size, x0+size, x0]
    vy = [y0, y0, y0+size, y0+size, y0, y0, y0+size, y0+size]
    vz = [0,0,0,0,0.6,0.6,0.6,0.6]
    faces = [[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    i,j,k = zip(*faces)
    mesh_traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k, color=st.session_state.room_colors[r["id"]], opacity=0.9, name=r["name"], hovertext=r["name"], hoverinfo="text"))
    # place a small invisible scatter marker at the center to capture clicks
    cx = x0 + size/2; cy = y0 + size/2; cz = 0.3
    marker_x.append(cx); marker_y.append(cy); marker_z.append(cz)
    marker_text.append(r["name"])
    marker_room_ids.append(r["id"])

marker_trace = go.Scatter3d(x=marker_x, y=marker_y, z=marker_z, mode='markers+text',
                            marker=dict(size=6, color='rgba(0,0,0,0)'), text=marker_text, textposition="top center",
                            hoverinfo='text')
fig = go.Figure(data=mesh_traces + [marker_trace])
# camera zoom settings
def camera_for_room(room_id):
    r = next(rr for rr in ROOMS if rr["id"]==room_id)
    return dict(eye=dict(x=r["x"]+3.5, y=r["y"]+3.5, z=2.5), center=dict(x=r["x"]+1, y=r["y"]+1, z=0.25))

camera = None
if st.session_state.selected_room:
    camera = camera_for_room(st.session_state.selected_room)
else:
    camera = dict(eye=dict(x=6, y=6, z=6))

fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                  height=560, scene_camera=camera, margin=dict(l=0,r=0,t=0,b=0))

with center:
    st.subheader("3D Floorplan — Click a room to inspect")
    # capture clicks from the invisible marker scatter (pointNumber corresponds to index in marker arrays)
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
    st.plotly_chart(fig, use_container_width=True)

# Handle clicks
if clicked and isinstance(clicked, list) and len(clicked)>0:
    pt = clicked[0]
    idx = pt.get('pointNumber', None)
    # marker pointNumber refers to index in the combined data traces; our marker is last trace
    if idx is not None:
        # When clicking Mesh3d elements, Plotly returns different indices; handle by mapping to marker trace
        # We will use customdata via text match based approach when available; else use nearest by coords
        # Here plotly_events gives 'curveNumber' and 'pointNumber'
        curve = pt.get('curveNumber', None)
        pnum = pt.get('pointNumber', None)
        # Our marker trace is the last trace (index len(mesh_traces))
        marker_curve_idx = len(mesh_traces)
        if curve == marker_curve_idx:
            room_id = marker_room_ids[pnum]
            # toggle selection: if same room clicked twice -> deselect
            if st.session_state.selected_room == room_id:
                st.session_state.selected_room = None
            else:
                st.session_state.selected_room = room_id
            st.experimental_rerun()

# Right panel: room info & mini-forecast chart
with right:
    sel = st.session_state.selected_room
    if sel is None:
        st.header("Inspector")
        st.write("Click a room to inspect its detector readings and short-term forecast.")
    else:
        room = next(r for r in ROOMS if r["id"]==sel)
        st.header(f"Room: {room['name']}")
        room_df = st.session_state.df[st.session_state.df['room']==sel].sort_values('timestamp')
        latest = room_df['ppm'].iloc[-1]
        prev = room_df['ppm'].iloc[-2] if len(room_df)>=2 else latest
        st.metric(label="Current ppm", value=f"{latest} ppm", delta=f"{round(latest-prev,2)} ppm")
        # mini forecast chart using last 20 samples
        recent = room_df['ppm'].dropna().iloc[-20:]
        if len(recent) >= 3:
            x = np.arange(len(recent))
            y = recent.values.astype(float)
            coef = np.polyfit(x,y,1)
            m,b = coef[0],coef[1]
            future_x = np.arange(len(recent), len(recent)+5)
            future_y = m*future_x + b
            import plotly.express as px
            hist_times = room_df['timestamp'].iloc[-len(recent):]
            df_plot = pd.DataFrame({"time": list(hist_times) + [hist_times.iloc[-1] + pd.Timedelta(minutes=i) for i in range(1,6)],
                                    "ppm": list(recent.values) + list(future_y),
                                    "type": ["measured"]*len(recent) + ["predicted"]*5})
            fig_mini = px.line(df_plot, x="time", y="ppm", color="type", markers=True, title="Short-term forecast")
            fig_mini.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.write("Not enough data for forecast yet.")

        st.markdown("Recent readings:")
        st.dataframe(room_df.tail(15)[['timestamp','ppm']].reset_index(drop=True))

        if st.button("Acknowledge this room"):
            name = st.text_input("Operator name for log", key=f"ack_{sel}")
            st.session_state.sim_history.append({"time": pd.Timestamp.now(), "room": sel, "severity": "ack", "operator": name})
            st.success("Acknowledged")

# After rendering, if any room has glow color, fade back to default after a short pause
if any(color in ["#ffb84d","#ff4c4c"] for color in st.session_state.room_colors.values()):
    time.sleep(1.2)
    st.session_state.room_colors = ROOM_COLORS.copy()
    st.experimental_rerun()
