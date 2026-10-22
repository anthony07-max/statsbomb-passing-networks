"""Build tidy passing-network tables from the raw StatsBomb events.

For every team in every match the network covers the period from kick-off to the team's first
substitution or red card (either side's red card, since it changes the shape of both teams), so that
the same eleven players are on the pitch for the whole window. Completed passes only; throw-ins are left out.

Outputs (data/processed/):
  matches.csv       one row per match (teams, score, stage)
  team_matches.csv  one row per team per match (window, passes, goals, xG for/against, result)
  nodes.csv         one row per player per team-match: average pass location and pass counts
  edges.csv         passer -> recipient counts within the window
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
TOURNAMENTS = ["world_cup_2022", "euro_2024"]


def clock(df):
    """Match clock in minutes (StatsBomb minutes already run on across periods)."""
    return df["minute"] + df["second"] / 60


def short_name(full, nickname=None):
    if isinstance(nickname, str) and nickname:
        return nickname
    parts = str(full).split()
    return full if len(parts) <= 2 else f"{parts[0]} {parts[-1]}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pd.read_parquet(RAW / f"{t}_events.parquet") for t in TOURNAMENTS], ignore_index=True)
    m = pd.concat([pd.read_csv(RAW / f"{t}_matches.csv") for t in TOURNAMENTS], ignore_index=True)
    matches = pd.DataFrame({
        "match_id": m["match_id"], "tournament": m["tournament"], "date": m["match_date"],
        "stage": m["competition_stage.name"], "home": m["home_team.home_team_name"],
        "away": m["away_team.away_team_name"], "home_score": m["home_score"], "away_score": m["away_score"],
    }).sort_values(["tournament", "date"])
    matches["label"] = (matches["home"] + " " + matches["home_score"].astype(str) + "-" +
                        matches["away_score"].astype(str) + " " + matches["away"] + " (" + matches["stage"] + ")")
    matches.to_csv(OUT / "matches.csv", index=False)

    nick = pd.read_csv(RAW / "player_names.csv").drop_duplicates("player").set_index("player")["nickname"].to_dict()
    ev = ev[ev["period"] <= 4].copy()          # drop penalty shoot-outs
    ev["t"] = clock(ev)
    full_time = ev.groupby("match_id")["t"].max()

    team_rows, node_rows, edge_rows = [], [], []
    for mt in matches.itertuples():
        e = ev[ev.match_id == mt.match_id]
        reds = e.loc[e.type == "Red Card", "t"]
        red_t = reds.min() if len(reds) else np.inf
        for team, opp, gf, ga in [(mt.home, mt.away, mt.home_score, mt.away_score),
                                  (mt.away, mt.home, mt.away_score, mt.home_score)]:
            subs = e.loc[(e.type == "Substitution") & (e.team == team), "t"]
            end = min(subs.min() if len(subs) else np.inf, red_t, full_time[mt.match_id])
            xi = e[(e.type == "Starting XI") & (e.team == team)]
            pos = dict(zip(xi.player, xi.position))
            p = e[(e.type == "Pass") & (e.team == team) & (e.t < end) & (e.subtype != "Throw-in")]
            comp = p[p.outcome.isna() & p.recipient.isin(pos)]
            shots = e[e.type == "Shot"]
            # shots from penalties in normal/extra time count; shoot-outs were already removed
            xgf = shots.loc[shots.team == team, "xg"].sum()
            xga = shots.loc[shots.team == opp, "xg"].sum()
            opp_p = e[(e.type == "Pass") & (e.team == opp) & (e.t < end) & (e.subtype != "Throw-in")]
            team_rows.append({
                "match_id": mt.match_id, "tournament": mt.tournament, "stage": mt.stage, "team": team,
                "opponent": opp, "goals_for": gf, "goals_against": ga,
                "result": "W" if gf > ga else ("D" if gf == ga else "L"),
                "points": 3 if gf > ga else (1 if gf == ga else 0),
                "xg_for": round(xgf, 3), "xg_against": round(xga, 3),
                "window_end_min": round(float(end), 1), "passes_attempted": len(p),
                "passes_completed": len(comp),
                "pass_share": round(len(p) / max(len(p) + len(opp_p), 1), 3),
                "formation": "-".join(str(xi.subtype.iloc[0])) if len(xi) else None,
            })
            for player in pos:
                made = comp[comp.player == player]
                rec = comp[comp.recipient == player]
                # average position: where the player made and received completed passes
                xs = pd.concat([made.x, rec.end_x]); ys = pd.concat([made.y, rec.end_y])
                node_rows.append({"match_id": mt.match_id, "team": team, "player": player,
                                  "short_name": short_name(player, nick.get(player)), "position": pos[player],
                                  "x": xs.mean(), "y": ys.mean(),
                                  "passes_made": len(made), "passes_received": len(rec)})
            for (a, b), n in comp.groupby(["player", "recipient"]).size().items():
                edge_rows.append({"match_id": mt.match_id, "team": team, "passer": a, "recipient": b, "passes": n})

    tm = pd.DataFrame(team_rows)
    tm.to_csv(OUT / "team_matches.csv", index=False)
    pd.DataFrame(node_rows).round(2).to_csv(OUT / "nodes.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "edges.csv", index=False)
    print(f"{len(matches)} matches, {len(tm)} team networks, median window {tm.window_end_min.median():.1f} min")


if __name__ == "__main__":
    main()
