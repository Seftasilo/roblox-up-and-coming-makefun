#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster — make.fun
"Predict Roblox's Up-and-Coming top 10" (roblox-up-and-coming-forecast-01).

Method — empirical rank-transition kernel (see README.md for the full write-up
and the backtest numbers):
  * The chart's rank is a fast-moving, mean-reverting diffusion. Measured on
    the kit's history (34 archive reads, 2026-10-30..2026-11-07): rank velocity
    has ~zero persistence (corr 0.055 over 24 h), top-10 survival is 83% at
    12 h, 70% at 24 h, 51% at 48 h and 38% at 72 h, and the #1 almost never
    survives 72 h (0/23). Trend-following on rank or on player counts (corr
    with rank change -0.018) does not help; honest broad probabilities do.
  * For every lag in {6,12,24,36,48,60,72} hours we estimate P(rank at lag h |
    rank now) from every pair of snapshots in the history, smoothed with a
    separable box blur (a Gaussian approximation) over both the starting rank
    and the landing rank, plus an exit mass for games that leave the chart.
  * The forecast is the uniform average of that kernel over the window hours
    0..72 (the score is the mean over the window's reads, so the time-averaged
    distribution is the honest forecast). A game is named at a place only
    while its probability beats the unnamed share it would otherwise get, so
    the long tail is left to the unnamed share instead of being named at a
    worse-than-nothing probability.
  * top10 is the greedy per-place argmax in place order.
  * No network, no wall clock, no randomness, no files besides /input:
    two runs on the same history write byte-identical output.

Backtest (local scorer, an exact port of the rules; it reproduces the
published house-baseline practice score -43.512039 to 6 decimals):
  * practice window (history to 2026-11-07T23:00Z, scored on the 10 archive
    reads to 2026-11-10): this forecaster -33.567, baseline -43.512
    (typical probability per place 0.0349 vs 0.0129), 6.4 top-10 hits.
  * walk-forward (5 cuts inside the history, next 72 h each): mean -35.64
    vs baseline -43.39. Without the SPREAD flattening the same harness
    scores -33.369 practice / -36.71 walk-forward mean.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LAGS = [6, 12, 24, 36, 48, 60, 72]
# Smoothing radii (box blur passes) chosen by backtest sweep; see README.md.
R_ROWS = 2   # smoothing over the starting rank
R_COLS = 2   # smoothing over the landing rank
LAG_TOL = 0.4  # relative lag tolerance for pair weights
# SPREAD flattens the time-averaged probabilities before naming: every named
# p is scaled by this factor and the released mass stays in the unnamed share.
# The kernel is estimated from only ~34 reads, so its time-averaged
# distribution is overconfident for a 72 h-ahead log score (a confident miss
# costs ln(0.0001) = -9.21 per place). Walk-forward backtest (5 cuts, next
# 72 h each): SPREAD 0.9 improves the mean from -36.71 to -35.64, at a
# practice-window cost of -33.37 -> -33.57, inside the practice read noise
# (SE ~1.1 over 10 reads). The 0.85-0.92 plateau all score alike, so 0.9 is
# a flat optimum, not a knife-edge fit.
SPREAD = 0.9


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-10-30T07:36:30.000Z
    return (
        int(s[0:4]), int(s[5:7]), int(s[8:10]),
        int(s[11:13]), int(s[14:16]), int(s[17:19]),
    )


def to_hours(t):
    # Days since 2026-01-01 00:00 UTC, in hours (naive but exact for ordering
    # and differences within one year; all inputs are 2026 UTC timestamps).
    y, mo, d, h, mi, se = t
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = sum(days_in_month[: mo - 1]) + (d - 1)
    return days * 24.0 + h + mi / 60.0 + se / 3600.0


def box_blur_1d(a, r):
    n = len(a)
    out = [0.0] * n
    pref = [0.0] * (n + 1)
    for i, v in enumerate(a):
        pref[i + 1] = pref[i] + v
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        out[i] = (pref[hi] - pref[lo]) / (hi - lo)
    return out


def smooth_matrix(mat, r_rows, r_cols):
    mat = [row[:] for row in mat]
    for i in range(len(mat)):
        mat[i] = box_blur_1d(mat[i], r_cols)
    for j in range(len(mat[0])):
        col = box_blur_1d([mat[i][j] for i in range(len(mat))], r_rows)
        for i in range(len(mat)):
            mat[i][j] = col[i]
    return mat


def build_pairs(rows):
    """(dt_hours, rank_now, rank_later_or_None) for every snapshot pair."""
    pairs = []
    times = [to_hours(parse_time(r["at"])) for r in rows]
    for i, a in enumerate(rows):
        ra = {g["universe_id"]: g["rank"] for g in a["up_and_coming"]}
        for j in range(i + 1, len(rows)):
            dt = times[j] - times[i]
            if dt > 80:
                break
            rb = {g["universe_id"]: g["rank"] for g in rows[j]["up_and_coming"]}
            for uid, r0 in ra.items():
                pairs.append((dt, r0, rb.get(uid)))
    return pairs


class Kernel:
    """P(future rank | current rank) per lag, smoothed; plus exit mass."""

    def __init__(self, pairs):
        self.mats = {}
        for lag in LAGS:
            n = 102
            count = [[0.0] * n for _ in range(n)]
            ex = [0.0] * n
            sig = lag * LAG_TOL + 1.0
            for dt, a, b in pairs:
                w = math.exp(-0.5 * ((dt - lag) / sig) ** 2)
                if w < 0.02:
                    continue
                if a >= n:
                    continue
                if b is None:
                    ex[a] += w
                elif b < n:
                    count[a][b] += w
            count = smooth_matrix(count, R_ROWS, R_COLS)
            ex = box_blur_1d(ex, R_ROWS)
            rows_out = {}
            for a in range(1, n):
                s = sum(count[a][1:]) + ex[a]
                if s <= 0:
                    rows_out[a] = None
                else:
                    rows_out[a] = (
                        [0.0] + [count[a][b] / s for b in range(1, n)],
                        ex[a] / s,
                    )
            self.mats[lag] = rows_out

    def dist(self, lag, r0):
        return self.mats[lag].get(r0)


def forecast(rows, places=10):
    kernel = Kernel(build_pairs(rows))
    last = sorted(rows[-1]["up_and_coming"], key=lambda g: g["rank"])
    chart = [g["universe_id"] for g in last]
    r0map = {g["universe_id"]: g["rank"] for g in last}
    probs = {}
    hours = list(range(0, 73))
    per_hour = 1.0 / len(hours)
    for t in hours:
        if t == 0:
            for uid in chart:
                r = r0map[uid]
                if r <= places:
                    probs[(uid, r)] = probs.get((uid, r), 0.0) + per_hour
            continue
        lo = max([lag for lag in LAGS if lag <= t], default=None)
        hi = min([lag for lag in LAGS if lag >= t], default=None)
        for uid in chart:
            r0 = r0map[uid]
            if lo is None:
                kd = kernel.dist(hi, r0)
            elif hi is None or lo == hi:
                kd = kernel.dist(lo, r0)
            else:
                a = kernel.dist(lo, r0)
                b = kernel.dist(hi, r0)
                if a is None or b is None:
                    continue
                f = (t - lo) / (hi - lo)
                kd = (
                    [(1.0 - f) * x + f * y for x, y in zip(a[0], b[0])],
                    (1.0 - f) * a[1] + f * b[1],
                )
            if kd is None:
                continue
            dist = kd[0]
            for k in range(1, places + 1):
                p = dist[k]
                if p > 0:
                    probs[(uid, k)] = probs.get((uid, k), 0.0) + p * per_hour
    # Flatten: scale every time-averaged probability by SPREAD (see above).
    # The released mass is not re-assigned to any game; it stays in the
    # unnamed share, which raises the floor for games the kernel did not
    # name at that place.
    if SPREAD != 1.0:
        probs = {k: v * SPREAD for k, v in probs.items()}
    # Greedy naming: name a game at a place only while its probability beats
    # the unnamed share it would otherwise receive.
    n_chart = len(chart)
    rows_out = []
    by_place = {}
    for (uid, k), p in probs.items():
        by_place.setdefault(k, []).append((p, uid))
    top_by_place = {}
    for k in range(1, places + 1):
        cand = sorted(by_place.get(k, []), reverse=True)
        named = []
        total = 0.0
        for p, uid in cand:
            if len(named) >= 100:
                break
            share = (1.0 - total) / (n_chart - len(named)) if n_chart > len(named) else 0.0
            if p <= share:
                break
            named.append((p, uid))
            total += p
        top_by_place[k] = named
        # Guard against rounding pushing a place's sum over 1.
        if total > 1.0:
            scale = 0.999999 / total
            named = [(p * scale, uid) for p, uid in named]
            top_by_place[k] = named
            total = 0.999999
        for p, uid in named:
            rows_out.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    rows_out.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    # top10: greedy argmax per place, in place order.
    used = set()
    top10 = []
    for k in range(1, places + 1):
        for _p, uid in top_by_place[k]:
            if uid not in used:
                used.add(uid)
                top10.append(uid)
                break
    if len(top10) < places:
        rest = sorted(
            ((p, uid) for (uid, k), p in probs.items() if uid not in used),
            reverse=True,
        )
        for _p, uid in rest:
            if len(top10) >= places:
                break
            used.add(uid)
            top10.append(uid)
    return {"probabilities": rows_out, "top10": top10[:places]}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    rows = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["at"])
    result = forecast(rows, places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, separators=(",", ":"), sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
