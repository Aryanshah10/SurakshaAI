import os
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
 
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "graph")
 
MULE_DEVICE_THRESHOLD    = 5
HIGH_FREQ_CALL_THRESHOLD = 10
 
_G              = None
_accounts       = None
_linkages       = None
_devices        = None
_calls          = None
_victim_reports = None
 
 
def _load_graph():
    global _G, _accounts, _linkages, _devices, _calls, _victim_reports
    if _G is not None:
        return _G, _accounts, _linkages, _devices, _calls, _victim_reports
 
    accounts       = pd.read_csv(f"{DATA_DIR}/accounts_nodes.csv")
    edges          = pd.read_csv(f"{DATA_DIR}/account_network_edges.csv")
    victim_reports = pd.read_csv(f"{DATA_DIR}/victim_reports.csv")
    linkages       = pd.read_csv(f"{DATA_DIR}/account_linkages.csv")
    devices        = pd.read_csv(f"{DATA_DIR}/device_fingerprints.csv")
    calls          = pd.read_csv(f"{DATA_DIR}/call_records.csv")
 
    # Merge account_status from linkages into accounts
    accounts = accounts.merge(
        linkages[["account_id", "account_status"]],
        on="account_id", how="left", suffixes=("_base", "_link")
    )
    if "account_status_link" in accounts.columns:
        accounts["account_status"] = accounts["account_status_link"].fillna(
            accounts.get("account_status_base", "active")
        )
        accounts.drop(columns=["account_status_base", "account_status_link"],
                      inplace=True, errors="ignore")
 
    # is_fraudster: original flag OR reported/frozen status
    accounts["is_fraudster"] = (
        accounts["is_fraudster"].astype(bool) |
        accounts["account_status"].isin(["reported", "frozen"])
    )
 
    G = nx.Graph()
 
    # Account nodes
    for _, row in accounts.iterrows():
        G.add_node(
            row["account_id"],
            node_type             = "account",
            is_fraudster          = bool(row["is_fraudster"]),
            risk_score            = float(row["risk_score"]),
            fraud_rate            = float(row.get("fraud_rate", 0.0)),
            fraud_count           = int(row.get("fraud_count", 0)),
            account_status        = str(row.get("account_status", "active")),
            bank_name             = str(row.get("bank_name", "")),
            bank_ifsc             = str(row.get("bank_ifsc", "")),
            jurisdiction_state    = str(row.get("jurisdiction_state", "")),
            jurisdiction_district = str(row.get("jurisdiction_district", "")),
            account_age_days      = int(row.get("account_age_days", 0)),
            total_transactions    = int(row.get("total_transactions", 0)),
            total_amount          = float(row.get("total_amount", 0.0)),
            has_2fa               = bool(row.get("has_2fa", False)),
            kyc_address_hash      = str(row.get("kyc_address_hash", "")),
            linked_device_id      = str(row.get("device_id", "")),
            linked_phone          = str(row.get("linked_phone", "")),
        )
 
    # Device nodes
    for _, row in devices.iterrows():
        is_sus = (
            bool(row["is_emulator_or_rooted"]) or
            int(row["accounts_per_device"]) > MULE_DEVICE_THRESHOLD
        )
        G.add_node(
            row["device_id"],
            node_type             = "device",
            imei_hash             = str(row["imei_hash"]),
            sim_slot_count        = int(row["sim_slot_count"]),
            accounts_per_device   = int(row["accounts_per_device"]),
            ip_geolocation        = str(row["ip_geolocation"]),
            is_emulator_or_rooted = bool(row["is_emulator_or_rooted"]),
            is_fraudster          = is_sus,
            risk_score            = round(min(
                int(row["accounts_per_device"]) / 74 +
                (0.3 if row["is_emulator_or_rooted"] else 0.0), 1.0), 2),
        )
 
    # Phone nodes (pre-build from accounts)
    for _, row in accounts.iterrows():
        phone = str(row.get("linked_phone", ""))
        if phone and not G.has_node(phone):
            G.add_node(phone, node_type="phone", is_fraudster=False, risk_score=0.0)
 
    # Account -> Device + Account -> Phone edges
    for _, row in accounts.iterrows():
        acc   = row["account_id"]
        dev   = str(row.get("device_id", ""))
        phone = str(row.get("linked_phone", ""))
        if dev and G.has_node(dev):
            G.add_edge(acc, dev, edge_type="used_device",
                       flagged=bool(G.nodes[dev].get("is_fraudster")), amount=0.0)
        if phone and G.has_node(phone):
            G.add_edge(acc, phone, edge_type="linked_to", flagged=False, amount=0.0)
 
    # Account <-> Account ring_link edges
    for _, row in edges.iterrows():
        G.add_edge(
            row["account_a"], row["account_b"],
            edge_type        = "ring_link",
            shared_type      = str(row["shared_type"]),
            connection_count = int(row["connection_count"]),
            ring_id          = str(row["ring_id"]),
            both_fraud       = bool(row["both_fraud"]),
            flagged          = bool(row["both_fraud"]),
            amount           = 0.0,
        )
 
    # Account <-> Account shared KYC address edges
    addr_groups = accounts.groupby("kyc_address_hash")["account_id"].apply(list)
    for addr, accs in addr_groups.items():
        if len(accs) > 1:
            for i in range(len(accs)):
                for j in range(i + 1, len(accs)):
                    if not G.has_edge(accs[i], accs[j]):
                        G.add_edge(accs[i], accs[j],
                                   edge_type="shared_kyc", flagged=True, amount=0.0)
 
    # Phone -> Phone call edges
    for _, row in calls.iterrows():
        caller = str(row["caller_id"])
        callee = str(row["callee_id"])
        is_sus = (
            bool(row["is_spoofed_number"]) or
            int(row["call_frequency_24h"]) > HIGH_FREQ_CALL_THRESHOLD
        )
        for ph in [caller, callee]:
            if not G.has_node(ph):
                G.add_node(ph, node_type="phone", is_fraudster=False, risk_score=0.0)
        G.add_edge(
            caller, callee,
            edge_type          = "called",
            call_duration_sec  = int(row["call_duration_sec"]),
            is_spoofed         = bool(row["is_spoofed_number"]),
            call_frequency_24h = int(row["call_frequency_24h"]),
            flagged            = is_sus,
            amount             = 0.0,
            timestamp          = str(row["call_timestamp"]),
        )
        if is_sus:
            G.nodes[caller]["is_fraudster"] = True
            G.nodes[caller]["risk_score"]   = max(
                G.nodes[caller].get("risk_score", 0.0), 0.8)
 
    _G              = G
    _accounts       = accounts
    _linkages       = linkages
    _devices        = devices
    _calls          = calls
    _victim_reports = victim_reports
    return G, accounts, linkages, devices, calls, victim_reports
 
 
def get_subgraph(node_id: str, depth: int = 2) -> dict:
    """Returns nodes + edges JSON for react-force-graph around any node."""
    G, _, _, _, _, _ = _load_graph()
    if node_id not in G:
        return {"nodes": [], "edges": [], "fraud_ring_detected": False,
                "total_suspicious_amount": 0.0}
 
    sub = nx.ego_graph(G, node_id, radius=depth)
 
    nodes = [
        {
            "id":                n,
            "label":             d.get("node_type", "unknown"),
            "fraud_score":       float(d.get("risk_score", 0.0)),
            "is_fraudster":      bool(d.get("is_fraudster", False)),
            "transaction_count": int(d.get("total_transactions", 0)),
            "account_status":    d.get("account_status", ""),
            "bank":              d.get("bank_name", ""),
            "jurisdiction":      d.get("jurisdiction_state", "") or d.get("ip_geolocation", ""),
            "fraud_rate":        float(d.get("fraud_rate", 0.0)),
        }
        for n, d in sub.nodes(data=True)
    ]
 
    edges = [
        {
            "source":      u,
            "target":      v,
            "edge_type":   d.get("edge_type", "unknown"),
            "amount":      float(d.get("amount", 0.0)),
            "timestamp":   d.get("timestamp", ""),
            "flagged":     bool(d.get("flagged", False)),
            "shared_type": d.get("shared_type", ""),
            "ring_id":     d.get("ring_id", ""),
        }
        for u, v, d in sub.edges(data=True)
    ]
 
    fraud_nodes = [n for n, d in sub.nodes(data=True) if d.get("is_fraudster")]
    flagged_edges = [d for _, _, d in sub.edges(data=True) if d.get("flagged")]
 
    return {
        "nodes":                   nodes,
        "edges":                   edges,
        "fraud_ring_detected":     len(fraud_nodes) >= 2,
        "total_suspicious_amount": float(len(flagged_edges)),
    }
 
 
def build_intelligence_packages() -> list:
    """Community detection + victim enrichment -> court-admissible packages."""
    G, accounts, linkages, devices, calls, victim_reports = _load_graph()
 
    account_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "account"]
    sub           = G.subgraph(account_nodes)
    communities   = list(greedy_modularity_communities(sub))
 
    packages = []
    for i, comm in enumerate(communities, 1):
        comm = list(comm)
        if len(comm) < 2:
            continue
 
        sub_acc    = accounts[accounts["account_id"].isin(comm)]
        fraud_accs = sub_acc[sub_acc["is_fraudster"] == True]
        fraud_rate = len(fraud_accs) / len(sub_acc) if len(sub_acc) > 0 else 0.0
        if fraud_rate == 0:
            continue
 
        device_ids  = sub_acc["device_id"].dropna().unique().tolist()
        linked_devs = devices[devices["device_id"].isin(device_ids)]
        rooted_devs = linked_devs[linked_devs["is_emulator_or_rooted"] == 1]["device_id"].tolist()
        mule_devs   = linked_devs[
            linked_devs["accounts_per_device"] > MULE_DEVICE_THRESHOLD
        ]["device_id"].tolist()
 
        linked_phones = sub_acc["linked_phone"].astype(str).unique().tolist()
        spoofed_calls = calls[
            calls["caller_id"].astype(str).isin(linked_phones) &
            (calls["is_spoofed_number"] == 1)
        ]
 
        victim_rows     = victim_reports[victim_reports["reported_account_id"].isin(comm)]
        victim_count    = len(victim_rows)
        total_lost      = float(victim_rows["amount_lost"].sum()) if not victim_rows.empty else 0.0
        complaint_types = victim_rows["complaint_type"].unique().tolist() if not victim_rows.empty else []
        ncrb_codes      = victim_rows["ncrb_category_code"].unique().tolist() if not victim_rows.empty else []
 
        ring_ids = list(set(
            d.get("ring_id", "")
            for _, _, d in G.edges(nbunch=comm, data=True)
            if d.get("ring_id")
        ))
 
        packages.append({
            "cluster_id":             f"CLUSTER{str(i).zfill(4)}",
            "ring_ids":               ring_ids,
            "linked_accounts":        comm,
            "total_accounts":         len(comm),
            "fraud_accounts":         fraud_accs["account_id"].tolist(),
            "linked_devices":         device_ids,
            "rooted_devices":         rooted_devs,
            "mule_devices":           mule_devs,
            "linked_phones":          linked_phones,
            "spoofed_call_count":     len(spoofed_calls),
            "victim_count":           victim_count,
            "total_amount_defrauded": round(total_lost, 2),
            "complaint_types":        complaint_types,
            "ncrb_codes":             ncrb_codes,
            "jurisdictions_spanned":  sub_acc["jurisdiction_state"].dropna().unique().tolist(),
            "districts_spanned":      sub_acc["jurisdiction_district"].dropna().unique().tolist(),
            "banks_involved":         sub_acc["bank_name"].dropna().unique().tolist(),
            "confidence_score":       round(fraud_rate, 2),
            "evidence_chain":         comm[:20],
            "legal_authority":        "IT Act 2000 S.69 + PMLA 2002 S.17 + IPC 420",
        })
 
    return packages
 
 
def generate_block_request(cluster_id: str, packages: list) -> dict:
    """Structured block request for a fraud cluster.
    In production: submitted to RBI + I4C + telecom APIs.
    For demo: simulates full submission pipeline."""
    package = next((p for p in packages if p["cluster_id"] == cluster_id), None)
    if not package:
        return {"error": f"Cluster {cluster_id} not found"}
 
    return {
        "block_request_id":      f"BLK-{cluster_id}-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
        "cluster_id":            cluster_id,
        "ring_ids":              package["ring_ids"],
        "action":                "BLOCK",
        "status":                "SUBMITTED",
        "accounts_to_freeze":    package["fraud_accounts"],
        "devices_to_blacklist":  package["rooted_devices"] + package["mule_devices"],
        "phones_to_block":       package["linked_phones"],
        "victims_affected":      package["victim_count"],
        "total_loss_inr":        package["total_amount_defrauded"],
        "ncrb_codes":            package["ncrb_codes"],
        "complaint_types":       package["complaint_types"],
        "spoofed_calls_flagged": package["spoofed_call_count"],
        "jurisdictions":         package["jurisdictions_spanned"],
        "districts":             package["districts_spanned"],
        "banks_notified":        package["banks_involved"],
        "confidence_score":      package["confidence_score"],
        "submitted_to": [
            "RBI_FRAUD_MONITORING_CELL",
            "I4C_CFCFRMS",
            "TELECOM_SIM_BLOCK_PORTAL",
            "IMEI_BLACKLIST_REGISTRY",
            "STATE_CYBER_CELL",
        ],
        "timestamp":       pd.Timestamp.now().isoformat(),
        "evidence_chain":  package["evidence_chain"],
        "legal_authority": package["legal_authority"],
    }