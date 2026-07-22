"""
Geospatial Crime Pattern Intelligence - Command Centre Query Engine
----------------------------------------------------------------------
Loads district-level complaint/incident records (geospatial_data.json)
and provides the query functions a command-centre dashboard would call:
  - get_top_hotspots(n)                  -> ranked patrol priority list
  - get_hotspot_detail(cluster_id)       -> drill-down for one hotspot
  - get_joint_patrol_recommendations()   -> cross-district coordination pairs
  - simulate_near_real_time_update(...)  -> simulates "near real-time" refresh

Unlike the original version, this build does not assume pre-computed
hotspot_clusters.csv / geo_events_clustered.csv / cross_district_coordination.csv
files exist. Instead it derives hotspot clusters, priority scores, and
joint-patrol pairs directly from the raw JSON records, using DBSCAN over
lat/lon to group nearby districts into a single hotspot.

Usage:
    python geo_query_engine.py
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


import os
DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "geospatial_data.json"))

# Priority level -> numeric weight, used when scoring a hotspot's urgency
PRIORITY_WEIGHTS = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# DBSCAN neighborhood radius in kilometers: districts within this distance
# of each other get folded into the same hotspot cluster
CLUSTER_RADIUS_KM = 150
EARTH_RADIUS_KM = 6371.0088


@lru_cache(maxsize=1)
def load_data():
    """Load the raw JSON records and derive the hotspot / event / joint-ops
    tables the rest of the module expects.

    Cached (lru_cache) because DBSCAN clustering is non-trivial work and
    the source data only refreshes when TM3's generator reruns — call
    `load_data.cache_clear()` after regenerating geospatial_data.json."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    events = pd.DataFrame(records)
    events["event_id"] = events.index
    events["priority_weight"] = events["priority_level"].map(PRIORITY_WEIGHTS)

    events = _assign_clusters(events)
    hotspots = _build_hotspot_summary(events)
    joint_ops = _build_joint_patrol_pairs(hotspots)

    return hotspots, events, joint_ops


def _assign_clusters(events: pd.DataFrame) -> pd.DataFrame:
    """Group nearby districts into spatial clusters via DBSCAN (haversine)."""
    coords = np.radians(events[["lat", "lon"]].to_numpy())
    eps = CLUSTER_RADIUS_KM / EARTH_RADIUS_KM  # convert km radius to radians
    db = DBSCAN(eps=eps, min_samples=1, metric="haversine")
    events["cluster_id"] = db.fit_predict(coords)
    return events


def _build_hotspot_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each cluster into one hotspot row with a patrol_priority_score."""
    rows = []
    max_year = events["year"].max()

    for cid, grp in events.groupby("cluster_id"):
        event_count = int(grp["complaint_count"].sum())
        avg_priority_weight = grp["priority_weight"].mean()
        # recency bonus: incidents from the latest year in the dataset count more
        recency_ratio = (grp["year"] == max_year).mean()

        # normalize each component to roughly 0-100 before combining
        volume_score = min(100, event_count / 5)          # 500+ complaints -> saturates
        urgency_score = (avg_priority_weight / 4) * 100    # Critical avg -> 100
        recency_score = recency_ratio * 100

        patrol_priority_score = round(
            0.5 * volume_score + 0.35 * urgency_score + 0.15 * recency_score, 1
        )

        rows.append({
            "cluster_id": f"HOTSPOT{cid}",
            "districts_spanned": ", ".join(sorted(grp["district"].unique())),
            "states_spanned": ", ".join(sorted(grp["state"].unique())),
            "dominant_crime_category": grp["incident_category"].mode().iat[0],
            "event_count": event_count,
            "num_districts": grp["district"].nunique(),
            "centroid_lat": round(grp["lat"].mean(), 4),
            "centroid_lon": round(grp["lon"].mean(), 4),
            "patrol_priority_score": patrol_priority_score,
        })

    return pd.DataFrame(rows)


def _build_joint_patrol_pairs(hotspots: pd.DataFrame) -> pd.DataFrame:
    """Pair up hotspots that sit in different states but are geographically
    close enough (within JOINT_RADIUS_KM) to justify coordinated patrols."""
    JOINT_RADIUS_KM = 400
    pairs = []

    for i in range(len(hotspots)):
        for j in range(i + 1, len(hotspots)):
            a, b = hotspots.iloc[i], hotspots.iloc[j]
            if a["states_spanned"] == b["states_spanned"]:
                continue  # same-state clusters aren't "cross-district" coordination
            dist = _haversine_km(a["centroid_lat"], a["centroid_lon"],
                                  b["centroid_lat"], b["centroid_lon"])
            if dist <= JOINT_RADIUS_KM:
                pairs.append({
                    "cluster_a": a["cluster_id"],
                    "cluster_b": b["cluster_id"],
                    "states_involved": f"{a['states_spanned']} | {b['states_spanned']}",
                    "distance_km": round(dist, 1),
                    "combined_priority_score": round(
                        a["patrol_priority_score"] + b["patrol_priority_score"], 1
                    ),
                })

    return pd.DataFrame(pairs)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def get_top_hotspots(hotspots: pd.DataFrame, n=10):
    """Ranked list for the command-centre map's priority panel."""
    return hotspots.sort_values("patrol_priority_score", ascending=False).head(n)


def get_hotspot_detail(hotspots: pd.DataFrame, events: pd.DataFrame, cluster_id: str):
    """Drill-down view: all events inside one hotspot, for click-to-expand on the map."""
    cid_num = int(cluster_id.replace("HOTSPOT", ""))
    detail_events = events[events["cluster_id"] == cid_num]
    summary = hotspots[hotspots["cluster_id"] == cluster_id]
    return summary, detail_events


def get_joint_patrol_recommendations(joint_ops: pd.DataFrame, top_n=5):
    """Cross-district/state coordination candidates - satisfies the PS's
    'inter-district intelligence sharing' requirement."""
    if joint_ops.empty:
        return joint_ops
    return joint_ops.sort_values("combined_priority_score", ascending=False).head(top_n)


def simulate_near_real_time_update(hotspots: pd.DataFrame, events: pd.DataFrame, new_event: dict):
    """
    Demonstrates the 'near real-time' requirement: a new incoming complaint/
    seizure event recalculates that district's priority score immediately,
    without needing to rerun full DBSCAN clustering on the whole dataset.
    In production this would be a lightweight incremental update, not a
    full batch re-cluster (which is too slow for real-time use).
    """
    matching = hotspots[hotspots["districts_spanned"].str.contains(
        new_event["jurisdiction_district"], na=False
    )]
    if matching.empty:
        print(f"[INFO] No existing hotspot for {new_event['jurisdiction_district']} - "
              f"would create a new emerging cluster")
        return None

    idx = matching.index[0]
    old_score = hotspots.loc[idx, "patrol_priority_score"]
    # simple incremental bump proportional to new event severity - placeholder for a real streaming update
    bump = new_event.get("severity_score", 10) * 0.05
    hotspots.loc[idx, "patrol_priority_score"] = round(min(100, old_score + bump), 1)
    print(f"[UPDATE] {new_event['jurisdiction_district']} priority score: "
          f"{old_score} -> {hotspots.loc[idx, 'patrol_priority_score']}")
    return hotspots.loc[idx]


if __name__ == "__main__":
    hotspots, events, joint_ops = load_data()

    print("=== TOP 5 PATROL-PRIORITY HOTSPOTS ===")
    print(get_top_hotspots(hotspots, 5)[
        ["cluster_id", "districts_spanned", "event_count", "patrol_priority_score"]
    ].to_string())

    print("\n=== TOP 5 JOINT PATROL COORDINATION RECOMMENDATIONS ===")
    print(get_joint_patrol_recommendations(joint_ops, 5).to_string())

    print("\n=== SIMULATED NEAR-REAL-TIME UPDATE (new incoming complaint) ===")
    sample_district = hotspots.iloc[0]["districts_spanned"].split(", ")[0]
    new_event = {"jurisdiction_district": sample_district, "severity_score": 85}
    simulate_near_real_time_update(hotspots, events, new_event)