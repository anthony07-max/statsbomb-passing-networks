"""Network metrics, link to results/xG, charts and key findings.

Graph per team-match: nodes = starting XI, directed edges weighted by completed passes within the
network window (see clean.py). Statistical comparisons use networks with a window of at least 30 minutes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_plot import match_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P, CH = ROOT / "data" / "processed", ROOT / "charts"
MIN_WINDOW = 30
CREDIT = "Data: StatsBomb Open Data (World Cup 2022, Euro 2024)"
METRICS = {
    "passes_per90": "Completed passes per 90",
    "density": "Density (pairs linked by 3+ passes)",
    "clustering": "Weighted clustering",
    "centralisation": "Degree centralisation",
    "top_player_share": "Top player's share of involvements",
    "network_height": "Network height (mean x, m)",
    "pass_share": "Share of passes (possession proxy)",
}


def team_network(nodes, edges):
    g = nx.DiGraph()
    g.add_nodes_from(nodes.player)
    for r in edges.itertuples():
        g.add_edge(r.passer, r.recipient, weight=r.passes, distance=1 / r.passes)
    n = g.number_of_nodes()
    strong = sum(1 for _, _, w in g.edges(data="weight") if w >= 3)
    deg = dict(g.degree(weight="weight"))
    dmax = max(deg.values()) if deg else 0
    total = sum(deg.values())
    und = nx.Graph()
    und.add_nodes_from(g.nodes)
    for a, b, w in g.edges(data="weight"):
        und.add_edge(a, b, weight=und.get_edge_data(a, b, {"weight": 0})["weight"] + w)
    mxw = max([w for *_, w in und.edges(data="weight")] or [1])
    for a, b in und.edges:
        und[a][b]["w_norm"] = und[a][b]["weight"] / mxw
    btw = nx.betweenness_centrality(g, weight="distance", normalized=True)
    try:
        eig = nx.eigenvector_centrality_numpy(und, weight="weight")
    except Exception:
        eig = {k: np.nan for k in g.nodes}
    metrics = {
        "density": strong / (n * (n - 1)) if n > 1 else np.nan,
        "clustering": nx.average_clustering(und, weight="w_norm"),
        "centralisation": sum(dmax - d for d in deg.values()) / ((n - 1) * dmax) if dmax else np.nan,
        "top_player_share": dmax / total if total else np.nan,
    }
    players = pd.DataFrame({"player": list(g.nodes), "weighted_degree": [deg[k] for k in g.nodes],
                            "betweenness": [btw[k] for k in g.nodes], "eigenvector": [eig[k] for k in g.nodes]})
    return metrics, players


def savefig(fig, name):
    fig.savefig(CH / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    CH.mkdir(exist_ok=True)
    matches = pd.read_csv(P / "matches.csv")
    tm = pd.read_csv(P / "team_matches.csv")
    nodes = pd.read_csv(P / "nodes.csv")
    edges = pd.read_csv(P / "edges.csv")

    rows, prow = [], []
    for r in tm.itertuples():
        n = nodes[(nodes.match_id == r.match_id) & (nodes.team == r.team)]
        e = edges[(edges.match_id == r.match_id) & (edges.team == r.team)]
        met, pl = team_network(n, e)
        inv = n.passes_made + n.passes_received
        met["network_height"] = np.average(n.x.fillna(n.x.mean()), weights=inv.clip(lower=1)) * 105 / 120
        rows.append({"match_id": r.match_id, "team": r.team, **met})
        pl = pl.merge(n[["player", "short_name", "position", "passes_made", "passes_received"]], on="player")
        pl["match_id"], pl["team"], pl["tournament"] = r.match_id, r.team, r.tournament
        pl["window_end_min"] = r.window_end_min
        prow.append(pl)
    tm = tm.merge(pd.DataFrame(rows), on=["match_id", "team"])
    tm["passes_per90"] = tm.passes_completed / tm.window_end_min * 90
    tm["xg_diff"] = tm.xg_for - tm.xg_against
    tm["goal_diff"] = tm.goals_for - tm.goals_against
    # opponent's metrics in the same match, so differences between the two sides can be used
    opp = tm[["match_id", "team"] + list(METRICS)].rename(columns={"team": "opponent", **{k: f"opp_{k}" for k in METRICS}})
    tm = tm.merge(opp, on=["match_id", "opponent"])
    players = pd.concat(prow, ignore_index=True)
    tm.round(4).to_csv(P / "team_metrics.csv", index=False)
    players.round(4).to_csv(P / "player_metrics.csv", index=False)

    s = tm[(tm.window_end_min >= MIN_WINDOW) & (tm.opponent.map(tm.set_index("team").index.value_counts()) > 0)]
    s = s[s.match_id.isin(tm[tm.window_end_min >= MIN_WINDOW].groupby("match_id").size().loc[lambda x: x == 2].index)]
    kf = {"matches": int(len(matches)), "team_networks": int(len(tm)), "networks_used": int(len(s)),
          "median_window_min": float(tm.window_end_min.median())}

    # 1. correlations with outcomes
    corr = []
    for k, lab in METRICS.items():
        for target in ["xg_diff", "goal_diff", "points"]:
            rho, p = stats.spearmanr(s[k], s[target])
            corr.append({"metric": k, "label": lab, "target": target, "rho": rho, "p": p})
    corr = pd.DataFrame(corr)
    corr.round(4).to_csv(P / "metric_outcome_correlations.csv", index=False)
    kf["spearman_xg_diff"] = {r.metric: [round(r.rho, 3), float(f"{r.p:.2g}")]
                              for r in corr[corr.target == "xg_diff"].itertuples()}
    kf["spearman_points"] = {r.metric: [round(r.rho, 3), float(f"{r.p:.2g}")]
                             for r in corr[corr.target == "points"].itertuples()}

    # 2. regression on differences vs opponent (standardised), one row per team-match
    d = pd.DataFrame({k: s[k] - s[f"opp_{k}"] for k in METRICS})
    z = (d - d.mean()) / d.std()
    z["xg_diff"] = s.xg_diff.values
    z["match_id"] = s.match_id.values
    one = z.drop_duplicates("match_id")  # each match appears twice with mirrored sign; keep one side
    ols = smf.ols("xg_diff ~ passes_per90 + density + centralisation + clustering + network_height",
                  data=one).fit(cov_type="HC3")
    ols2 = smf.ols("xg_diff ~ pass_share + density + centralisation", data=one).fit(cov_type="HC3")
    with open(P / "regression_summary.txt", "w") as f:
        f.write(str(ols.summary()) + "\n\n" + str(ols2.summary()))
    kf["ols_r2"] = round(ols.rsquared, 3)
    kf["ols_coefs"] = {k: [round(v, 3), float(f"{ols.pvalues[k]:.2g}")] for k, v in ols.params.items() if k != "Intercept"}
    kf["ols2_coefs"] = {k: [round(v, 3), float(f"{ols2.pvalues[k]:.2g}")] for k, v in ols2.params.items() if k != "Intercept"}
    kf["ols2_r2"] = round(ols2.rsquared, 3)
    # Did the side with the denser network win more often?
    dec = s[s.result != "D"]
    kf["denser_side_won_pct"] = round(100 * ((dec.density > dec.opp_density) == (dec.result == "W")).mean(), 1)
    kf["more_passes_side_won_pct"] = round(100 * ((dec.passes_per90 > dec.opp_passes_per90) == (dec.result == "W")).mean(), 1)
    kf["more_passes_side_higher_xg_pct"] = round(100 * ((s.passes_per90 > s.opp_passes_per90) == (s.xg_diff > 0)).mean(), 1)
    kf["decisive_networks"] = int(len(dec))
    kf["by_result_means"] = s.groupby("result")[list(METRICS)].mean().round(3).to_dict(orient="index")

    # 3. team averages per tournament
    team = s.groupby(["tournament", "team"]).agg(matches=("match_id", "count"), passes_per90=("passes_per90", "mean"),
            density=("density", "mean"), centralisation=("centralisation", "mean"),
            clustering=("clustering", "mean"), xg_diff=("xg_diff", "mean"), points=("points", "mean")).reset_index()
    team.round(3).to_csv(P / "team_tournament_summary.csv", index=False)
    top = team.sort_values("passes_per90", ascending=False).head(5)
    kf["top_passing_teams"] = [[r.team, r.tournament, round(r.passes_per90, 1), round(r.density, 3)] for r in top.itertuples()]
    rho_team, p_team = stats.spearmanr(team.passes_per90, team.xg_diff)
    kf["team_level_passes_vs_xgdiff"] = [round(rho_team, 3), float(f"{p_team:.2g}")]

    # 4. key connectors
    pu = players[players.match_id.isin(s.match_id) & (players.window_end_min >= MIN_WINDOW)].copy()
    pu["key_connector"] = pu.groupby(["match_id", "team"]).betweenness.transform("max") == pu.betweenness
    kc = pu.groupby(["tournament", "player", "short_name", "team"]).agg(
        networks=("match_id", "count"), mean_betweenness=("betweenness", "mean"),
        key_connector_games=("key_connector", "sum"), mean_degree=("weighted_degree", "mean")).reset_index()
    kc = kc[kc.networks >= 3].sort_values("mean_betweenness", ascending=False)
    kc.round(4).to_csv(P / "key_connectors.csv", index=False)
    kf["top_connectors"] = [[r.short_name, r.team, r.tournament, round(r.mean_betweenness, 3), int(r.networks)]
                            for r in kc.head(5).itertuples()]
    kf["key_connector_positions"] = (pu[pu.key_connector].position.value_counts(normalize=True).head(5) * 100).round(1).to_dict()

    # 5. pass combinations
    ew = edges.merge(tm[["match_id", "team", "tournament"]], on=["match_id", "team"])
    sn = nodes.drop_duplicates("player").set_index("player").short_name
    ew["passer"], ew["recipient"] = ew.passer.map(sn), ew.recipient.map(sn)
    combo = ew.groupby(["tournament", "team", "passer", "recipient"]).agg(passes=("passes", "sum"), matches=("match_id", "nunique")).reset_index()
    combo["per_match"] = combo.passes / combo.matches
    combo = combo[combo.matches >= 3].sort_values("per_match", ascending=False)
    combo.head(200).round(2).to_csv(P / "top_combinations.csv", index=False)
    kf["top_combinations"] = [[r.passer, r.recipient, r.team, r.tournament, round(r.per_match, 1)] for r in combo.head(5).itertuples()]

    with open(P / "key_findings.json", "w") as f:
        json.dump(kf, f, indent=2, default=float)

    # ---------- charts ----------
    for tour, fn in [("world_cup_2022", "01_final_world_cup_2022.png"), ("euro_2024", "02_final_euro_2024.png")]:
        fm = matches[(matches.tournament == tour) & (matches.stage == "Final")].iloc[0]
        labs = []
        for t in (fm.home, fm.away):
            r = tm[(tm.match_id == fm.match_id) & (tm.team == t)].iloc[0]
            labs.append(f"{t}: {int(r.passes_completed)} completed passes to {r.window_end_min:.0f}'\n"
                        f"density {r.density:.2f} | centralisation {r.centralisation:.2f}")
        fig = match_figure(nodes, edges, fm.match_id, (fm.home, fm.away), labs)
        fig.suptitle(f"{tour.replace('_', ' ').title()} final: {fm.label}  (until first sub/red card)", y=1.02, fontsize=13)
        savefig(fig, fn)

    fig, ax = plt.subplots(figsize=(10, 7))
    for tour, c in [("world_cup_2022", "#8a1538"), ("euro_2024", "#1d4e89")]:
        t = team[team.tournament == tour]
        ax.scatter(t.passes_per90, t.xg_diff, s=30 + 25 * t.matches, color=c, alpha=0.7, label=tour.replace("_", " ").title())
        for r in t.itertuples():
            if abs(r.xg_diff) > 0.8 or r.passes_per90 > 500 or r.passes_per90 < 230:
                ax.annotate(r.team, (r.passes_per90, r.xg_diff), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Mean completed passes per 90 in network window")
    ax.set_ylabel("Mean xG difference per match")
    ax.set_title(f"Team passing volume vs xG difference (Spearman rho = {rho_team:.2f})")
    ax.legend(); ax.text(0.01, -0.09, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "03_team_passing_vs_xg.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k in zip(axes, ["passes_per90", "density", "centralisation"]):
        ax.scatter(s[k], s.xg_diff, s=14, alpha=0.5, color="#2a6f97")
        b = np.polyfit(s[k], s.xg_diff, 1); xs = np.linspace(s[k].min(), s[k].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#d62828")
        rho = kf["spearman_xg_diff"][k][0]
        ax.set_title(f"{METRICS[k]}\nSpearman rho = {rho:.2f}"); ax.set_xlabel(METRICS[k]); ax.set_ylabel("xG difference (match)")
    fig.text(0.01, -0.02, CREDIT + f"; {len(s)} team-match networks with window >= {MIN_WINDOW} min", fontsize=7, color="#555")
    fig.tight_layout(); savefig(fig, "04_metrics_vs_xg_diff.png")

    pv = corr.pivot(index="label", columns="target", values="rho")[["xg_diff", "goal_diff", "points"]]
    pv = pv.loc[pv.xg_diff.sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    yy = np.arange(len(pv))
    for i, (c, col) in enumerate(zip(pv.columns, ["#2a6f97", "#e9c46a", "#d62828"])):
        ax.barh(yy + (i - 1) * 0.27, pv[c], height=0.27, label=c.replace("_", " "), color=col)
    ax.set_yticks(yy, pv.index); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman correlation with match outcome"); ax.legend()
    ax.set_title("Which network metrics go with better results?")
    ax.text(0.01, -0.12, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "05_metric_outcome_correlations.png")

    k15 = kc.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([f"{r.short_name} ({r.team}, {'WC22' if r.tournament.startswith('world') else 'E24'})" for r in k15.itertuples()],
            k15.mean_betweenness, color="#2a9d8f")
    for i, r in enumerate(k15.itertuples()):
        ax.text(r.mean_betweenness, i, f"  {int(r.key_connector_games)}/{int(r.networks)}", va="center", fontsize=8)
    ax.set_xlabel("Mean betweenness centrality (label: matches as team's top connector / networks)")
    ax.set_title("Key connectors: players most often on the shortest passing routes (min 3 networks)")
    ax.text(0.01, -0.1, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "06_key_connectors.png")

    c15 = combo.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([f"{r.passer} -> {r.recipient} ({r.team}, {'WC22' if r.tournament.startswith('world') else 'E24'})" for r in c15.itertuples()], c15.per_match, color="#e76f51")
    ax.set_xlabel("Completed passes per match within network window (min 3 matches)")
    ax.set_title("Most frequent passing combinations")
    ax.text(0.01, -0.1, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "07_top_combinations.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, k in zip(axes, ["passes_per90", "density", "centralisation", "network_height"]):
        data = [s.loc[s.result == r, k] for r in "WDL"]
        ax.boxplot(data, tick_labels=["Won", "Drew", "Lost"], patch_artist=True,
                   boxprops=dict(facecolor="#a8dadc"), medianprops=dict(color="#d62828"))
        ax.set_title(METRICS[k], fontsize=10)
    fig.suptitle("Network metrics by match result", y=1.02)
    fig.text(0.01, -0.03, CREDIT, fontsize=7, color="#555")
    fig.tight_layout(); savefig(fig, "08_metrics_by_result.png")
    print(json.dumps(kf, indent=1, default=float))


if __name__ == "__main__":
    main()
