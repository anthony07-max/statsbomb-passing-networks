"""Download StatsBomb Open Data for the FIFA World Cup 2022 and UEFA Euro 2024.

Source: StatsBomb Open Data, https://github.com/statsbomb/open-data
Licence: StatsBomb Public Data User Agreement (non-commercial use, StatsBomb must be credited),
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf

Each full event file is a few MB, so it is reduced on the fly to the events this project needs
(passes, shots, substitutions, red cards, starting XIs) and saved as one parquet per tournament in
data/raw/. Player nicknames come from the lineup files.
Re-running skips tournaments that are already downloaded.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITIONS = [(43, 106, "world_cup_2022"), (55, 282, "euro_2024")]
SESSION = requests.Session()


def get(url):
    err = None
    for _ in range(5):
        try:
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccup: retry
            err = e
    raise err


def name(d):
    return (d or {}).get("name")


def reduce_match(match_id):
    rows = []
    for e in get(f"{BASE}/events/{match_id}.json"):
        t = e["type"]["name"]
        base = {"match_id": match_id, "index": e["index"], "period": e["period"],
                "minute": e["minute"], "second": e["second"], "type": t,
                "team": name(e.get("team")), "player": name(e.get("player")),
                "player_id": (e.get("player") or {}).get("id"), "position": name(e.get("position"))}
        loc = e.get("location") or [None, None]
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                rows.append({**base, "player": p["player"]["name"], "player_id": p["player"]["id"],
                             "position": p["position"]["name"], "subtype": str(e["tactics"]["formation"])})
        elif t == "Pass":
            p = e["pass"]
            end = p.get("end_location") or [None, None]
            rows.append({**base, "x": loc[0], "y": loc[1], "end_x": end[0], "end_y": end[1],
                         "recipient": name(p.get("recipient")),
                         "recipient_id": (p.get("recipient") or {}).get("id"),
                         "outcome": name(p.get("outcome")), "subtype": name(p.get("type")),
                         "play_pattern": name(e.get("play_pattern"))})
        elif t == "Shot":
            s = e["shot"]
            rows.append({**base, "x": loc[0], "y": loc[1], "xg": s.get("statsbomb_xg"),
                         "outcome": name(s.get("outcome")), "subtype": name(s.get("type"))})
        elif t == "Substitution":
            rows.append({**base, "recipient": name(e["substitution"].get("replacement")),
                         "recipient_id": e["substitution"]["replacement"]["id"]})
        elif t in ("Bad Behaviour", "Foul Committed"):
            card = name((e.get(t.lower().replace(" ", "_")) or {}).get("card"))
            if card in ("Red Card", "Second Yellow"):
                rows.append({**base, "type": "Red Card", "subtype": card})
    return rows


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, sid, label in COMPETITIONS:
        out = RAW / f"{label}_events.parquet"
        if out.exists():
            print("exists:", out.name)
            continue
        matches = get(f"{BASE}/matches/{cid}/{sid}.json")
        m = pd.json_normalize(matches)
        m["tournament"] = label
        m.to_csv(RAW / f"{label}_matches.csv", index=False)
        with ThreadPoolExecutor(12) as ex:
            res = list(ex.map(reduce_match, m["match_id"]))
        df = pd.DataFrame([r for rows in res for r in rows])
        df["tournament"] = label
        df.to_parquet(out, index=False)
        print(f"{label}: {len(m)} matches, {len(df):,} rows", flush=True)
    # player nicknames (e.g. "Rodri") from the lineup files, used for chart labels
    out = RAW / "player_names.csv"
    if not out.exists():
        ids = pd.concat([pd.read_csv(RAW / f"{l}_matches.csv")["match_id"] for *_, l in COMPETITIONS])
        with ThreadPoolExecutor(12) as ex:
            lineups = list(ex.map(lambda i: get(f"{BASE}/lineups/{i}.json"), ids))
        names = {(p["player_id"], p["player_name"], p.get("player_nickname"))
                 for lu in lineups for t in lu for p in t["lineup"]}
        pd.DataFrame(sorted(names, key=str), columns=["player_id", "player", "nickname"]).drop_duplicates(
            "player_id").to_csv(out, index=False)
        print("player names:", len(names))


if __name__ == "__main__":
    sys.exit(main())
