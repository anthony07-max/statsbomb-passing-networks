"""Streamlit app: pick a tournament, match and team to see its passing network and metrics.
Run the pipeline first (./run_pipeline.sh), then: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_plot import draw_network  # noqa: E402

P = ROOT / "data" / "processed"
st.set_page_config(page_title="StatsBomb passing networks", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(P / "matches.csv"), pd.read_csv(P / "team_metrics.csv"), pd.read_csv(P / "nodes.csv"),
            pd.read_csv(P / "edges.csv"), pd.read_csv(P / "player_metrics.csv"), pd.read_csv(P / "key_connectors.csv"))


matches, tm, nodes, edges, players, kc = load()
st.title("Passing networks: World Cup 2022 and Euro 2024")
st.caption("Data: StatsBomb Open Data (free for non-commercial use, credit StatsBomb). "
           "Networks run from kick-off to each team's first substitution or a red card.")

tour = st.sidebar.selectbox("Tournament", ["world_cup_2022", "euro_2024"], format_func=lambda t: t.replace("_", " ").title())
ms = matches[matches.tournament == tour]
label = st.sidebar.selectbox("Match", ms.label.tolist(), index=len(ms) - 1)
mrow = ms[ms.label == label].iloc[0]
team = st.sidebar.radio("Team", [mrow.home, mrow.away])
min_passes = st.sidebar.slider("Minimum passes to draw a link", 1, 10, 3)

r = tm[(tm.match_id == mrow.match_id) & (tm.team == team)].iloc[0]
c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_network(ax, nodes[(nodes.match_id == mrow.match_id) & (nodes.team == team)],
                 edges[(edges.match_id == mrow.match_id) & (edges.team == team)],
                 f"{team} vs {r.opponent} (to {r.window_end_min:.0f}')", min_passes=min_passes)
    st.pyplot(fig)
with c2:
    st.subheader(f"{team} {int(r.goals_for)}-{int(r.goals_against)} {r.opponent}")
    a, b, c = st.columns(3)
    a.metric("Completed passes / 90", f"{r.passes_per90:.0f}", f"{r.passes_per90 - r.opp_passes_per90:+.0f} vs opp")
    b.metric("Density", f"{r.density:.2f}", f"{r.density - r.opp_density:+.2f}")
    c.metric("xG for / against", f"{r.xg_for:.2f} / {r.xg_against:.2f}")
    a.metric("Centralisation", f"{r.centralisation:.2f}")
    b.metric("Clustering", f"{r.clustering:.2f}")
    c.metric("Network height (m)", f"{r.network_height:.1f}")
    if r.window_end_min < 30:
        st.warning("Short window (<30 min) - this network is excluded from the statistical analysis.")
    pl = players[(players.match_id == mrow.match_id) & (players.team == team)]
    st.dataframe(pl[["short_name", "position", "passes_made", "passes_received", "betweenness", "eigenvector"]]
                 .sort_values("betweenness", ascending=False).round(3), hide_index=True, width="stretch")

st.subheader("Tournament key connectors (mean betweenness, min 3 networks)")
st.dataframe(kc[kc.tournament == tour].head(20)[["short_name", "team", "networks", "key_connector_games", "mean_betweenness"]]
             .round(3), hide_index=True, width="stretch")
