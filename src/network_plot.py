"""Shared pitch passing-network drawing (used by analysis.py and the Streamlit app)."""
import matplotlib.pyplot as plt
from mplsoccer import Pitch

MIN_PASSES = 3  # draw a link only when a pair completed at least this many passes


def draw_network(ax, nodes, edges, title, color="#d62828", min_passes=MIN_PASSES):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1f3d2b", line_color="#c7d5cc", linewidth=1)
    pitch.draw(ax=ax)
    pos = nodes.set_index("player")[["x", "y"]]
    e = edges[(edges.passes >= min_passes) & edges.passer.isin(pos.index) & edges.recipient.isin(pos.index)]
    # merge both directions so each pair is drawn once with its total volume
    pair = e.assign(a=e[["passer", "recipient"]].min(axis=1), b=e[["passer", "recipient"]].max(axis=1))
    pair = pair.groupby(["a", "b"], as_index=False)["passes"].sum()
    mx = max(pair.passes.max() if len(pair) else 1, 1)
    for r in pair.itertuples():
        pitch.lines(pos.loc[r.a, "x"], pos.loc[r.a, "y"], pos.loc[r.b, "x"], pos.loc[r.b, "y"],
                    lw=1 + 9 * r.passes / mx, color="white", alpha=0.25 + 0.6 * r.passes / mx, ax=ax, zorder=1)
    inv = nodes.passes_made + nodes.passes_received
    pitch.scatter(nodes.x, nodes.y, s=150 + 1400 * inv / max(inv.max(), 1), color=color,
                  edgecolors="white", linewidth=1.5, ax=ax, zorder=2)
    for r in nodes.itertuples():
        pitch.annotate(r.short_name if len(r.short_name) <= 14 else r.short_name.split()[-1], (r.x, r.y - 4.2), ax=ax, ha="center", va="center",
                       fontsize=8, color="white", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.text(1, 82.5, "Attacking direction ->   Data: StatsBomb Open Data", fontsize=7, color="#555")
    return ax


def match_figure(nodes, edges, match_id, teams, labels, colors=("#d62828", "#1d6fb8")):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, team, lab, c in zip(axes, teams, labels, colors):
        n = nodes[(nodes.match_id == match_id) & (nodes.team == team)]
        e = edges[(edges.match_id == match_id) & (edges.team == team)]
        draw_network(ax, n, e, lab, color=c)
    fig.tight_layout()
    return fig
