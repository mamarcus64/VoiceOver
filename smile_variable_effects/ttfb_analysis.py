"""
Time-to-First-Blink (TTFB) analysis around smile onset and offset.

For each smile:
  - TTFB_onset:  time from smile start_ts to the next AU45 blink event
  - TTFB_offset: time from smile end_ts   to the next AU45 blink event

Censored at CENSOR_S seconds if no blink is found.
Survival curves (Kaplan-Meier) and log-rank tests stratified by narrative valence.
Within-subject normalization: per-subject median TTFB used for paired tests.

Figures:
  ttfb_survival.png    — KM curves for onset + offset, by valence
  ttfb_summary.png     — median TTFB comparison + paper-quality version
  ttfb_paper.pdf/png   — figure for the paper
"""

import json, time, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import logrank, ttest_rel, CensoredData
from pathlib import Path

BASE     = Path("/Users/marcus/Desktop/usc/VoiceOver")
OF_DIR   = Path("/Users/marcus/Desktop/usc/openface_results")
SW_PKL   = BASE / "smile_variable_effects/smile_window_features.pkl"
OUT_DIR  = BASE / "smile_variable_effects"
PAPER_FIG_DIR = Path("/Users/marcus/Desktop/usc/-ICMI-2026-Smiling-Faces/figures")

BLINK_THRESH = 0.5      # AU45_r rising-edge threshold
CENSOR_S     = 10.0     # max observation window (s); 95th pctile inter-blink ~7.5s
MIN_SMILES   = 50       # minimum smiles per valence group to include

VAL_COLORS = {"negative": "#C0392B", "neutral": "#7F8C8D", "positive": "#27AE60"}
VALENCES   = ["negative", "neutral", "positive"]

# ─────────────────────────────────────────────────────────────────────────────
def kaplan_meier(times, events):
    """Return (t_steps, S) Kaplan-Meier survival curve. events: 1=blink, 0=censored."""
    times  = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order  = np.argsort(times)
    t, e   = times[order], events[order]

    unique_t = np.unique(t[e == 1])
    S = 1.0
    curve_t = [0.0]
    curve_S = [1.0]
    n_total = len(t)

    for ti in unique_t:
        n_risk   = int((t >= ti).sum())
        n_events = int(((t == ti) & (e == 1)).sum())
        if n_risk > 0:
            S *= 1.0 - n_events / n_risk
        curve_t.append(ti)
        curve_S.append(S)

    return np.array(curve_t), np.array(curve_S)

def median_survival(t_steps, S):
    """Median survival time (first t where S ≤ 0.5)."""
    idx = np.searchsorted(-S, -0.5)
    if idx < len(t_steps):
        return float(t_steps[idx])
    return np.nan

def sig_str(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load smile windows
# ─────────────────────────────────────────────────────────────────────────────
print("Loading smile windows...", flush=True)
sw = pd.read_pickle(SW_PKL)
sw = sw[sw["narrative_valence"].isin(VALENCES)].copy()
print(f"  {len(sw):,} smiles, {sw['subject'].nunique()} subjects")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute TTFB for every smile
# ─────────────────────────────────────────────────────────────────────────────
print("Computing TTFB per smile window...", flush=True)
t0 = time.time()

by_video = collections.defaultdict(list)
for idx, row in sw.iterrows():
    by_video[row["video_id"]].append((idx, row["start_ts"], row["end_ts"]))

ttfb_onset_arr  = np.full(len(sw), np.nan)
ttfb_offset_arr = np.full(len(sw), np.nan)
event_onset_arr  = np.zeros(len(sw), dtype=int)   # 1 = blink found
event_offset_arr = np.zeros(len(sw), dtype=int)

sw_idx_map = {idx: pos for pos, idx in enumerate(sw.index)}

for vi, (vid, entries) in enumerate(sorted(by_video.items())):
    of_path = OF_DIR / vid / "result.csv"
    if not of_path.exists():
        continue

    of = pd.read_csv(of_path)
    of.columns = [c.strip() for c in of.columns]
    ts    = of["timestamp"].values.astype(np.float64)
    au45  = of["AU45_r"].values.astype(np.float64)
    valid = ~((of["gaze_0_x"].values == 0.0) & (of["gaze_0_y"].values == 0.0))

    # Blink onset timestamps
    is_onset = np.zeros(len(ts), dtype=bool)
    is_onset[1:] = (au45[1:] >= BLINK_THRESH) & (au45[:-1] < BLINK_THRESH) & valid[1:]
    blink_ts = ts[is_onset]

    for (idx, s_start, s_end) in entries:
        pos = sw_idx_map[idx]

        # TTFB after onset
        after = blink_ts[blink_ts > s_start]
        if len(after) > 0:
            dt = float(after[0] - s_start)
            if dt <= CENSOR_S:
                ttfb_onset_arr[pos]  = dt
                event_onset_arr[pos] = 1
            else:
                ttfb_onset_arr[pos]  = CENSOR_S
        else:
            ttfb_onset_arr[pos] = CENSOR_S

        # TTFB after offset
        after = blink_ts[blink_ts > s_end]
        if len(after) > 0:
            dt = float(after[0] - s_end)
            if dt <= CENSOR_S:
                ttfb_offset_arr[pos]  = dt
                event_offset_arr[pos] = 1
            else:
                ttfb_offset_arr[pos]  = CENSOR_S
        else:
            ttfb_offset_arr[pos] = CENSOR_S

    if (vi + 1) % 500 == 0:
        print(f"  {vi+1}/{len(by_video)} ({time.time()-t0:.0f}s)", flush=True)

sw["ttfb_onset"]    = ttfb_onset_arr
sw["ttfb_offset"]   = ttfb_offset_arr
sw["event_onset"]   = event_onset_arr
sw["event_offset"]  = event_offset_arr

print(f"Done ({time.time()-t0:.0f}s). "
      f"Blink found: onset={event_onset_arr.mean()*100:.1f}%, "
      f"offset={event_offset_arr.mean()*100:.1f}%", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Summary stats per valence
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("TTFB SUMMARY BY VALENCE")
print("="*68)

km_data = {}  # (valence, which) → (t_steps, S, median)
for which, t_col, e_col in [("onset", "ttfb_onset", "event_onset"),
                              ("offset", "ttfb_offset", "event_offset")]:
    print(f"\n── TTFB after smile {which} ─────────────────────────────────────")
    print(f"  {'Valence':10s}  {'N':>6s}  {'Observed%':>10s}  {'Median(s)':>9s}  "
          f"{'Mean(s)':>8s}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*8}")
    for val in VALENCES:
        mask = sw["narrative_valence"] == val
        sub  = sw[mask]
        t_   = sub[t_col].values
        e_   = sub[e_col].values
        t_steps, S = kaplan_meier(t_, e_)
        med = median_survival(t_steps, S)
        km_data[(val, which)] = (t_steps, S, med)
        obs = e_.mean() * 100
        print(f"  {val:10s}  {len(t_):6,}  {obs:>9.1f}%  {med:>9.2f}s  "
              f"{t_[e_==1].mean():>8.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Log-rank tests between valence pairs
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Log-rank tests ──────────────────────────────────────────────────")
pairs = [("negative","positive"), ("negative","neutral"), ("neutral","positive")]
for which, t_col, e_col in [("onset","ttfb_onset","event_onset"),
                              ("offset","ttfb_offset","event_offset")]:
    print(f"\n  {which}:")
    for v1, v2 in pairs:
        g1 = sw[sw["narrative_valence"]==v1]
        g2 = sw[sw["narrative_valence"]==v2]
        cd1 = CensoredData(uncensored=g1.loc[g1[e_col]==1, t_col].values,
                           right     =g1.loc[g1[e_col]==0, t_col].values)
        cd2 = CensoredData(uncensored=g2.loc[g2[e_col]==1, t_col].values,
                           right     =g2.loc[g2[e_col]==0, t_col].values)
        res = logrank(cd1, cd2)
        p = res.pvalue
        print(f"    {v1:10s} vs {v2:10s}: p={p:.3e}  {sig_str(p)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Within-subject paired test (more conservative)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Within-subject paired t-test (median TTFB per subject) ──────────")
for which, t_col in [("onset","ttfb_onset"), ("offset","ttfb_offset")]:
    print(f"\n  {which}:")
    # Compare negative vs positive
    for v1, v2 in [("negative","positive"), ("neutral","positive")]:
        m1 = sw[sw["narrative_valence"]==v1].groupby("subject")[t_col].median()
        m2 = sw[sw["narrative_valence"]==v2].groupby("subject")[t_col].median()
        both = pd.DataFrame({"v1":m1, "v2":m2}).dropna()
        if len(both) < 10:
            continue
        delta = both["v1"] - both["v2"]
        t, p  = ttest_rel(both["v1"], both["v2"])
        d     = delta.mean() / delta.std()
        print(f"    {v1} − {v2}: Δ={delta.mean():+.3f}s, d={d:+.3f}, "
              f"p={p:.3e} {sig_str(p)}, n={len(both)} subjects")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE: Survival curves + median comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding figures...", flush=True)

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Time to First Blink (TTFB) — Kaplan-Meier survival curves\n"
             "Survival = probability of NOT yet having blinked",
             fontsize=11, fontweight="bold")

titles = ["After smile ONSET", "After smile OFFSET"]
which_list = ["onset", "offset"]
t_cols = ["ttfb_onset", "ttfb_offset"]
e_cols = ["event_onset", "event_offset"]

for ax, which, t_col, e_col, title in zip(axes, which_list, t_cols, e_cols, titles):
    for val in VALENCES:
        mask = sw["narrative_valence"] == val
        t_   = sw.loc[mask, t_col].values
        e_   = sw.loc[mask, e_col].values
        t_steps, S = km_data[(val, which)][:2]
        med = km_data[(val, which)][2]
        n   = mask.sum()
        ax.step(t_steps, S, where="post", color=VAL_COLORS[val], lw=2.0,
                label=f"{val.capitalize()} (n={n:,}, med={med:.2f}s)")
        # Mark median
        if not np.isnan(med):
            ax.axvline(med, color=VAL_COLORS[val], lw=0.8, ls=":", alpha=0.6)

    ax.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5, label="S=0.5 (median)")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("P(not yet blinked)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, CENSOR_S)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
fig.savefig(OUT_DIR / "ttfb_survival.png", bbox_inches="tight")
plt.close()
print("  Saved ttfb_survival.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PAPER FIGURE: clean 2-panel KM curves
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.dpi": 300,
})

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

panel_labels = ["(a) After smile onset", "(b) After smile offset"]
for ax, which, t_col, e_col, label in zip(
        axes, which_list, t_cols, e_cols, panel_labels):

    for val in VALENCES:
        t_steps, S, med = km_data[(val, which)]
        n = (sw["narrative_valence"] == val).sum()
        ax.step(t_steps, S, where="post", color=VAL_COLORS[val], lw=1.8,
                label=f"{val.capitalize()} (n={n:,})")
        if not np.isnan(med):
            ax.plot(med, 0.5, "o", color=VAL_COLORS[val], ms=4, zorder=5)

    ax.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Time since smile event (s)", fontsize=8)
    ax.set_ylabel("P(not yet blinked)", fontsize=8)
    ax.set_title(label, fontsize=8, pad=4)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.2, lw=0.4)
    ax.tick_params(labelsize=7)

    # Annotate log-rank p-values
    g_neg = sw[sw["narrative_valence"]=="negative"]
    g_pos = sw[sw["narrative_valence"]=="positive"]
    def cd(g, tc, ec):
        return CensoredData(uncensored=g.loc[g[ec]==1, tc].values,
                            right     =g.loc[g[ec]==0, tc].values)
    res = logrank(cd(g_neg, t_col, e_col), cd(g_pos, t_col, e_col))
    ax.text(0.97, 0.55, f"neg vs pos: {sig_str(res.pvalue)} (p={res.pvalue:.3e})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#333")

    g_neu = sw[sw["narrative_valence"]=="neutral"]
    res2 = logrank(cd(g_neu, t_col, e_col), cd(g_pos, t_col, e_col))
    ax.text(0.97, 0.47, f"neu vs pos: {sig_str(res2.pvalue)} (p={res2.pvalue:.3e})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#333")

plt.tight_layout(pad=0.8, w_pad=1.5)
for ext in [".pdf", ".png"]:
    out = PAPER_FIG_DIR / f"ttfb_survival{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
plt.close()

print("\nDone.")
