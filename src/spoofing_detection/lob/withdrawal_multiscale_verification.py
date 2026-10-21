"""Independent ledger and priority verification for multiscale withdrawal.

No production score, aggregator or winner function is used here. Eligibility,
physical ownership and quantities are reconstructed from prepare inputs.
"""
from __future__ import annotations

import math
from collections import defaultdict

import polars as pl

KEYS = ["instrument", "partition_id", "actor_key", "execution_cluster_id"]


def _indexed(frame, fields, label):
    rows = {}
    for r in frame.iter_rows(named=True):
        key = tuple(r[k] for k in fields)
        if any(v is None for v in key) or key in rows:
            raise ValueError(f"null/duplicate {label} identity")
        rows[key] = r
    return rows


def _same(actual, expected, field):
    if expected is None:
        equal = actual is None
    elif type(expected) is bool:
        equal = type(actual) is bool and actual == expected
    elif isinstance(expected, (float, int)):
        equal = (not isinstance(actual, bool) and isinstance(actual, (float, int))
                 and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"independent verification mismatch: {field}")


def _logratio(x, q):
    if x == 0:
        return 0.0
    ratio = x / q
    return math.log1p(ratio) if math.isfinite(ratio) else math.log(x) - math.log(q) + math.log1p(q / x)


def verify_accounting(executions, candidates, cancellations, population, audit, horizons, policies):
    """Prove membership, quantities and winner flags by a direct ledger loop."""
    e = _indexed(executions, KEYS, "execution")
    c = _indexed(candidates, KEYS + ["deceptive_order_id"], "candidate")
    p = _indexed(population, KEYS + ["horizon_ns", "age_policy"], "population")
    a = _indexed(audit, KEYS + ["candidate_order_id", "cancel_sort_index", "horizon_ns", "age_policy"], "audit")
    if audit.schema["assigned_flag"] != pl.Boolean or audit["assigned_flag"].null_count():
        raise ValueError("winner flags must be non-null Boolean")
    if not horizons or not policies:
        raise ValueError("empty verification design")
    expected_keys = {ek + (h, policy["id"]) for ek in e for h in horizons for policy in policies}
    if set(p) != expected_keys:
        raise ValueError("population does not contain the exact execution/cell product")
    by_execution = defaultdict(list)
    for ck, row in c.items():
        ek = ck[:-1]
        if ek not in e:
            raise ValueError("orphan candidate")
        age = e[ek]["cluster_start_replay_time_ns"] - row["deceptive_order_first_seen_replay_ns"]
        if type(age) is not int or age < 0:
            raise ValueError("invalid native candidate age")
        quantity = row["deceptive_order_visible_qty_pre"]
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("invalid candidate quantity")
        by_execution[ek].append((row, age))
    physical = _indexed(cancellations, ["partition_id", "sort_index", "ORDERID"], "cancellation")
    by_lifecycle = defaultdict(list)
    for row in physical.values():
        quantity = row["visible_qty_pre_cancel"]
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("invalid cancellation quantity")
        identity = (row["partition_id"], row["actor_key"], row["side"], str(row["ORDERID"]), row["cancel_order_first_seen_sort_index"])
        by_lifecycle[identity].append(row)
    checked_audit = set()
    assigned_total = 0
    for h in horizons:
        for policy in policies:
            cohort = {}
            weights = {}
            competing = defaultdict(list)
            for ek, execution in e.items():
                cohort[ek] = []
                for order, age in by_execution[ek]:
                    if policy["max_age_ns"] is not None and age > policy["max_age_ns"]:
                        continue
                    half = policy["half_life_ns"]
                    weight = math.exp(-math.log(2) * age / half) if half is not None else 1.0
                    oid = str(order["deceptive_order_id"])
                    weights[(ek, oid)] = weight
                    cohort[ek].append(order)
                    lookup = (ek[1], ek[2], execution["deceptive_side"], oid, order["deceptive_order_first_seen_sort_index"])
                    for cancel in by_lifecycle[lookup]:
                        delay = cancel["cancellation_removal_ns"] - execution["cluster_end_fill_anchor_ns"]
                        if 0 <= delay < h and cancel["sort_index"] > execution["cluster_last_sort_index"]:
                            physical_id = (ek[1], cancel["sort_index"], oid)
                            competing[physical_id].append((ek, order, cancel))
            expected_links = []
            for physical_id, possibilities in competing.items():
                winner = min(possibilities, key=lambda t: (-e[t[0]]["cluster_last_sort_index"],
                              e[t[0]]["cluster_first_sort_index"], t[0][3]))
                for ek, order, cancel in possibilities:
                    aid = ek + (str(order["deceptive_order_id"]), cancel["sort_index"], h, policy["id"])
                    if aid not in a:
                        raise ValueError("missing eligible competitor in cancellation audit")
                    won = ek == winner[0]
                    native_integers = {
                        "cluster_first_sort_index": e[ek]["cluster_first_sort_index"],
                        "cluster_last_sort_index": e[ek]["cluster_last_sort_index"],
                        "age_ns": e[ek]["cluster_start_replay_time_ns"] - order["deceptive_order_first_seen_replay_ns"],
                        "deceptive_order_first_seen_sort_index": order["deceptive_order_first_seen_sort_index"],
                        "cancel_order_first_seen_sort_index": cancel["cancel_order_first_seen_sort_index"],
                    }
                    for field, native in native_integers.items():
                        if type(a[aid][field]) is not int or a[aid][field] != native:
                            raise ValueError(f"independent native audit mismatch: {field}")
                    _same(a[aid]["candidate_visible_qty_pre"], order["deceptive_order_visible_qty_pre"], "candidate_visible_qty_pre")
                    _same(a[aid]["cancel_visible_qty"], cancel["visible_qty_pre_cancel"], "cancel_visible_qty")
                    _same(a[aid]["assignment_rule"], "latest_prior_cluster_sort_index", "assignment_rule")
                    _same(a[aid]["assigned_flag"], won, "assigned_flag")
                    _same(a[aid]["competing_cluster_count"], len(possibilities), "competing_cluster_count")
                    expected_links.append((aid, won, order, cancel))
                    checked_audit.add(aid)
            raw = defaultdict(list)
            aged = defaultdict(list)
            used = defaultdict(float)
            for aid, won, order, cancel in sorted(expected_links, key=lambda t: (t[0][5], t[0][3], t[0][1], t[0][2])):
                ek = aid[:4]
                oid = aid[4]
                remaining = max(order["deceptive_order_visible_qty_pre"] - used[(ek, oid)], 0)
                removed = min(cancel["visible_qty_pre_cancel"], remaining) if won else 0.0
                used[(ek, oid)] += removed
                weighted = removed * weights[(ek, oid)]
                _same(a[aid]["attributed_cancel_visible_qty"], removed, "attributed_cancel_visible_qty")
                _same(a[aid]["attributed_cancel_qty_age_weighted"], weighted, "attributed_cancel_qty_age_weighted")
                _same(a[aid]["age_weight"], weights[(ek, oid)], "age_weight")
                raw[ek].append(removed)
                aged[ek].append(weighted)
                assigned_total += int(won)
            for ek, execution in e.items():
                row = p[ek + (h, policy["id"])]
                q = execution["execution_quantity"]
                d = execution["opposite_market_visible_qty_pre_topN"]
                pr = math.fsum(o["deceptive_order_visible_qty_pre"] for o in cohort[ek])
                pg = math.fsum(o["deceptive_order_visible_qty_pre"] * weights[(ek, str(o["deceptive_order_id"]))] for o in cohort[ek])
                ur, ug = math.fsum(raw[ek]), math.fsum(aged[ek])
                if ug > pg and math.isclose(ug, pg, rel_tol=1e-12, abs_tol=0):
                    ug = pg
                if q <= 0 or ur > pr or ug > pg or (d is not None and pr > d):
                    raise ValueError("independent quantity accounting invariant failed")
                share = ug / pg if pg > 0 else None
                product = _logratio(pg, q) * _logratio(ug, q)
                expected = {
                    "execution_quantity": q, "opposite_market_visible_qty_pre_topN": d,
                    "candidate_qty_raw": pr, "withdrawal_qty_raw": ur,
                    "candidate_qty_age_weighted": pg, "withdrawal_qty_age_weighted": ug,
                    "removal_share_age_weighted": share,
                    "candidate_market_share_age_weighted": pg / d if d is not None and d > 0 else None,
                    "WMSCI_self": product * share if share is not None else 0.0,
                    "WMSCI_market": product * (ug / d) if d is not None and d > 0 else None,
                    "is_candidate": pr > 0 and ur > 0,
                    "market_score_supported": d is not None and d > 0,
                    "age_weight_underflow_count": sum(o["deceptive_order_visible_qty_pre"] > 0 and weights[(ek, str(o["deceptive_order_id"]))] == 0 for o in cohort[ek]),
                }
                for field, value in expected.items():
                    _same(row[field], value, field)
    if checked_audit != set(a):
        raise ValueError("audit contains extra or ineligible links")
    return {"status": "verified", "execution_cells_checked": len(p),
            "eligible_links_checked": len(a), "assigned_cell_cancellations": assigned_total}


def verify_priorities(population, fraction):
    """Reconstruct budgets via Python sorting, independent of flag implementation."""
    columns = ["priority_top_w_self", "priority_top_w_market", "priority_top_j_given_self", "priority_top_j_given_market"]
    for name in columns + ["is_candidate", "complete_grid"]:
        if population.schema[name] != pl.Boolean or population[name].null_count():
            raise ValueError("priority and candidate flags must be non-null Boolean")
    data = _indexed(population, KEYS + ["horizon_ns", "age_policy"], "population")
    groups = defaultdict(list)
    for key, row in data.items():
        groups[(row["instrument"], row["horizon_ns"], row["age_policy"])].append(key)
    expected = {name: set() for name in columns}
    for keys in groups.values():
        for suffix in ("self", "market"):
            score = "WMSCI_" + suffix
            eligible = [k for k in keys if data[k]["is_candidate"] and data[k][score] is not None and math.isfinite(data[k][score]) and data[k][score] > 0]
            ordering = lambda k, field: (-data[k][field], data[k]["execution_cluster_id"], data[k]["partition_id"], data[k]["actor_key"])
            top = sorted(eligible, key=lambda k: ordering(k, score))[:math.ceil(len(eligible) * fraction)]
            expected["priority_top_w_" + suffix].update(top)
            js = [k for k in top if data[k]["complete_grid"] and data[k]["J"] is not None and math.isfinite(data[k]["J"])]
            chosen = sorted(js, key=lambda k: ordering(k, "J"))[:math.ceil(len(js) * fraction)]
            expected["priority_top_j_given_" + suffix].update(chosen)
    for key, row in data.items():
        for name in columns:
            _same(row[name], key in expected[name], name)
    return {"status": "verified", "priority_rows_checked": len(data)}
