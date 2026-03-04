#!/usr/bin/env python3
"""
Smiling-segments parameter sweep.

Sweeps the three filtering parameters (intensityThreshold, mergeDistance,
minDuration) across the full corpus, then projects annotation time & cost
over a grid of (contextBefore, contextAfter, overhead_factor, hourly_rate).

Outputs:
  - smiling_sweep_results.csv   (one row per param combo, aggregate stats)
  - smiling_sweep_report.md     (structured doc with tables + figures)
  - figures/smiling_sweep_*.png (plots)
"""

import json
import os
import sys
import itertools
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SEG_DIR = ROOT / "data" / "smiling_segments"
MANIFEST = ROOT / "data" / "manifest.json"
OUT_DIR = ROOT / "analysis"
FIG_DIR = OUT_DIR / "figures"

# ── parameter grids ─────────────────────────────────────────────────────────
INTENSITY_GRID = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
MERGE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
MIN_DUR_GRID = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]

CONTEXT_BEFORE_GRID = [1.0, 2.0, 3.0, 5.0]
CONTEXT_AFTER_GRID = [1.0, 2.0, 3.0, 5.0]

OVERHEAD_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0]
HOURLY_RATES = [15.0, 20.0, 25.0, 30.0]


# ── core filter (mirrors frontend filterAndMerge) ───────────────────────────
def filter_and_merge(segments, intensity_thresh, merge_dist, min_dur):
    """Return list of (start, end) tuples after filtering + merging."""
    filtered = [(s["start_ts"], s["end_ts"], s["peak_r"], s["mean_r"])
                for s in segments if s["mean_r"] >= intensity_thresh]
    filtered.sort(key=lambda x: x[0])

    merged = []
    for start, end, peak, mean in filtered:
        if merged and start - merged[-1][1] <= merge_dist:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), max(prev[2], peak),
                          (prev[3] + mean) / 2)
        else:
            merged.append((start, end, peak, mean))

    return [(s, e) for s, e, _, _ in merged if e - s >= min_dur]


# ── load data ────────────────────────────────────────────────────────────────
def load_all():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    vid_to_subject = {v["id"]: v["int_code"] for v in manifest}

    videos = {}
    for fname in os.listdir(SEG_DIR):
        if not fname.endswith(".json"):
            continue
        vid_id = fname[:-5]
        with open(SEG_DIR / fname) as f:
            data = json.load(f)
        videos[vid_id] = {
            "segments": data["segments"],
            "total_duration_sec": data["total_duration_sec"],
            "subject": vid_to_subject.get(vid_id),
        }
    return videos


# ── sweep ────────────────────────────────────────────────────────────────────
def run_sweep(videos):
    combos = list(itertools.product(INTENSITY_GRID, MERGE_GRID, MIN_DUR_GRID))
    n_combos = len(combos)
    n_videos = len(videos)
    print(f"Sweeping {n_combos} param combos × {n_videos} videos "
          f"= {n_combos * n_videos:,} filter ops …")

    vid_ids = sorted(videos.keys())
    # pre-extract segment lists for speed
    seg_lists = [videos[v]["segments"] for v in vid_ids]
    subjects = [videos[v]["subject"] for v in vid_ids]
    durations = np.array([videos[v]["total_duration_sec"] for v in vid_ids])

    rows = []
    t0 = time.time()
    for ci, (intensity, merge_d, min_d) in enumerate(combos):
        if ci % 50 == 0:
            elapsed = time.time() - t0
            pct = ci / n_combos * 100
            print(f"  [{pct:5.1f}%] combo {ci}/{n_combos}  ({elapsed:.1f}s)")

        per_video_events = []
        per_video_core_dur = []
        per_subject_events = defaultdict(int)
        per_subject_core_dur = defaultdict(float)
        per_subject_n_vids = defaultdict(int)

        for vi, segs in enumerate(seg_lists):
            moments = filter_and_merge(segs, intensity, merge_d, min_d)
            n_events = len(moments)
            core_dur = sum(e - s for s, e in moments)

            per_video_events.append(n_events)
            per_video_core_dur.append(core_dur)

            subj = subjects[vi]
            if subj is not None:
                per_subject_events[subj] += n_events
                per_subject_core_dur[subj] += core_dur
                per_subject_n_vids[subj] += 1

        pv_events = np.array(per_video_events)
        pv_core = np.array(per_video_core_dur)

        total_events = int(pv_events.sum())
        total_core_sec = float(pv_core.sum())
        n_subjects = len(per_subject_events)
        subj_events = np.array(list(per_subject_events.values())) if per_subject_events else np.array([0])
        subj_core = np.array(list(per_subject_core_dur.values())) if per_subject_core_dur else np.array([0.0])

        rows.append({
            "intensity": intensity,
            "merge_dist": merge_d,
            "min_dur": min_d,
            "total_events": total_events,
            "total_core_sec": total_core_sec,
            "n_videos_with_events": int((pv_events > 0).sum()),
            "events_per_video_mean": float(pv_events.mean()),
            "events_per_video_median": float(np.median(pv_events)),
            "events_per_video_p25": float(np.percentile(pv_events, 25)),
            "events_per_video_p75": float(np.percentile(pv_events, 75)),
            "events_per_video_max": int(pv_events.max()),
            "core_sec_per_video_mean": float(pv_core.mean()),
            "core_sec_per_video_median": float(np.median(pv_core)),
            "n_subjects": n_subjects,
            "events_per_subject_mean": float(subj_events.mean()) if n_subjects else 0,
            "events_per_subject_median": float(np.median(subj_events)) if n_subjects else 0,
            "core_sec_per_subject_mean": float(subj_core.mean()) if n_subjects else 0,
        })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return pd.DataFrame(rows)


# ── projection helpers ───────────────────────────────────────────────────────
def project_annotation(df):
    """Add columns for various context / overhead / rate projections."""
    proj_rows = []
    for _, r in df.iterrows():
        for ctx_b in CONTEXT_BEFORE_GRID:
            for ctx_a in CONTEXT_AFTER_GRID:
                clip_sec = r["total_core_sec"] + r["total_events"] * (ctx_b + ctx_a)
                for factor in OVERHEAD_FACTORS:
                    annot_sec = clip_sec * factor
                    for rate in HOURLY_RATES:
                        cost = annot_sec / 3600 * rate
                        proj_rows.append({
                            "intensity": r["intensity"],
                            "merge_dist": r["merge_dist"],
                            "min_dur": r["min_dur"],
                            "ctx_before": ctx_b,
                            "ctx_after": ctx_a,
                            "overhead_factor": factor,
                            "hourly_rate": rate,
                            "total_events": r["total_events"],
                            "total_core_sec": r["total_core_sec"],
                            "clip_sec": clip_sec,
                            "annotation_sec": annot_sec,
                            "annotation_hours": annot_sec / 3600,
                            "annotation_cost_usd": cost,
                        })
    return pd.DataFrame(proj_rows)


# ── figures ──────────────────────────────────────────────────────────────────
def make_figures(df_core, df_proj):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Events heatmap: intensity × merge (aggregate over min_dur) ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["total_events", "events_per_video_mean", "n_videos_with_events"],
        ["Total events", "Events / video (mean)", "Videos with ≥1 event"],
    ):
        pivot = df_core.groupby(["intensity", "merge_dist"])[metric].mean().unstack()
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel("merge distance (s)")
        ax.set_ylabel("intensity threshold")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG_DIR / "smiling_sweep_heatmaps.png", dpi=150)
    plt.close(fig)

    # ---------- 2. Core clip time vs intensity (lines for each min_dur) ─────
    fig, ax = plt.subplots(figsize=(8, 5))
    for md in sorted(df_core["min_dur"].unique()):
        sub = df_core[(df_core["min_dur"] == md) & (df_core["merge_dist"] == 0.5)]
        ax.plot(sub["intensity"], sub["total_core_sec"] / 3600,
                marker="o", label=f"minDur={md:.1f}s")
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("Total core clip time (hours)")
    ax.set_title("Core clip time vs intensity (merge=0.5s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smiling_sweep_clip_time.png", dpi=150)
    plt.close(fig)

    # ---------- 3. Event count vs intensity (lines for merge distance) ──────
    fig, ax = plt.subplots(figsize=(8, 5))
    for mg in sorted(df_core["merge_dist"].unique()):
        sub = df_core[(df_core["merge_dist"] == mg) & (df_core["min_dur"] == 0.5)]
        ax.plot(sub["intensity"], sub["total_events"],
                marker="s", label=f"merge={mg:.2g}s")
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("Total events (corpus-wide)")
    ax.set_title("Event count vs intensity (minDur=0.5s)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smiling_sweep_event_count.png", dpi=150)
    plt.close(fig)

    # ---------- 4. Annotation cost surface ──────────────────────────────────
    # fix core params to defaults, vary context + overhead
    defaults = df_proj[
        (df_proj["intensity"] == 1.8) &
        (df_proj["merge_dist"] == 0.5) &
        (df_proj["min_dur"] == 0.5) &
        (df_proj["hourly_rate"] == 20.0)
    ]
    if not defaults.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # 4a: cost vs overhead, lines for total context
        ax = axes[0]
        for ctx_b in CONTEXT_BEFORE_GRID:
            sub = defaults[(defaults["ctx_before"] == ctx_b) & (defaults["ctx_after"] == 2.0)]
            ax.plot(sub["overhead_factor"], sub["annotation_cost_usd"],
                    marker="o", label=f"ctx_before={ctx_b:.0f}s")
        ax.set_xlabel("Overhead factor")
        ax.set_ylabel("Annotation cost (USD)")
        ax.set_title("Cost vs overhead (default filters, ctx_after=2s, $20/hr)")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

        # 4b: cost vs hourly rate, lines for overhead
        ax = axes[1]
        defaults2 = df_proj[
            (df_proj["intensity"] == 1.8) &
            (df_proj["merge_dist"] == 0.5) &
            (df_proj["min_dur"] == 0.5) &
            (df_proj["ctx_before"] == 3.0) &
            (df_proj["ctx_after"] == 2.0)
        ]
        for factor in OVERHEAD_FACTORS:
            sub = defaults2[defaults2["overhead_factor"] == factor]
            ax.plot(sub["hourly_rate"], sub["annotation_cost_usd"],
                    marker="s", label=f"overhead={factor:.1f}×")
        ax.set_xlabel("Hourly rate (USD)")
        ax.set_ylabel("Annotation cost (USD)")
        ax.set_title("Cost vs rate (default filters, ctx 3+2s)")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

        fig.savefig(FIG_DIR / "smiling_sweep_cost.png", dpi=150)
        plt.close(fig)

    # ---------- 5. Per-subject distribution (default params) ────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    default_core = df_core[
        (df_core["intensity"] == 1.8) &
        (df_core["merge_dist"] == 0.5) &
        (df_core["min_dur"] == 0.5)
    ]
    if not default_core.empty:
        ax.bar(["Events/video\n(mean)", "Events/video\n(median)",
                "Events/subject\n(mean)", "Events/subject\n(median)"],
               [default_core.iloc[0]["events_per_video_mean"],
                default_core.iloc[0]["events_per_video_median"],
                default_core.iloc[0]["events_per_subject_mean"],
                default_core.iloc[0]["events_per_subject_median"]],
               color=["#f59e0b", "#fbbf24", "#3b82f6", "#60a5fa"])
        ax.set_ylabel("Count")
        ax.set_title("Event distribution (default params: intensity=1.8, merge=0.5s, minDur=0.5s)")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smiling_sweep_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {FIG_DIR}")


# ── report ───────────────────────────────────────────────────────────────────
def write_report(df_core, df_proj, n_videos, n_subjects):
    lines = []
    lines.append("# Smiling Segments Parameter Sweep\n")
    lines.append(f"**Corpus:** {n_videos} videos, {n_subjects} subjects  ")
    lines.append(f"**Core param combos:** {len(df_core)}  ")
    lines.append(f"**Full projection rows:** {len(df_proj):,}\n")

    lines.append("## Parameter Grids\n")
    lines.append(f"| Parameter | Values |")
    lines.append(f"|-----------|--------|")
    lines.append(f"| intensityThreshold | {INTENSITY_GRID} |")
    lines.append(f"| mergeDistance (s) | {MERGE_GRID} |")
    lines.append(f"| minDuration (s) | {MIN_DUR_GRID} |")
    lines.append(f"| contextBefore (s) | {CONTEXT_BEFORE_GRID} |")
    lines.append(f"| contextAfter (s) | {CONTEXT_AFTER_GRID} |")
    lines.append(f"| overhead factor | {OVERHEAD_FACTORS} |")
    lines.append(f"| hourly rate (USD) | {HOURLY_RATES} |")
    lines.append("")

    # ── Key summary table (select combos) ────────────────────────────────
    lines.append("## Core Sweep Summary (selected combos)\n")
    lines.append("| intensity | merge | minDur | events | events/vid | "
                 "events/subj | core hrs | vids w/ events |")
    lines.append("|-----------|-------|--------|--------|------------|"
                 "-------------|----------|----------------|")
    showcase = df_core[df_core["merge_dist"].isin([0.0, 0.5, 2.0]) &
                       df_core["min_dur"].isin([0.0, 0.5, 1.0])].copy()
    showcase = showcase.sort_values(["intensity", "merge_dist", "min_dur"])
    for _, r in showcase.iterrows():
        lines.append(
            f"| {r['intensity']:.1f} | {r['merge_dist']:.1f} | {r['min_dur']:.1f} "
            f"| {int(r['total_events']):,} | {r['events_per_video_mean']:.1f} "
            f"| {r['events_per_subject_mean']:.0f} "
            f"| {r['total_core_sec']/3600:.1f} "
            f"| {int(r['n_videos_with_events'])} |"
        )
    lines.append("")

    # ── Default params spotlight ─────────────────────────────────────────
    default = df_core[
        (df_core["intensity"] == 1.8) &
        (df_core["merge_dist"] == 0.5) &
        (df_core["min_dur"] == 0.5)
    ]
    if not default.empty:
        r = default.iloc[0]
        lines.append("## Default Parameters Spotlight\n")
        lines.append(f"intensityThreshold=1.8, mergeDistance=0.5s, minDuration=0.5s\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total events | {int(r['total_events']):,} |")
        lines.append(f"| Videos with ≥1 event | {int(r['n_videos_with_events'])} / {n_videos} |")
        lines.append(f"| Events / video (mean) | {r['events_per_video_mean']:.1f} |")
        lines.append(f"| Events / video (median) | {r['events_per_video_median']:.1f} |")
        lines.append(f"| Events / video (p25–p75) | {r['events_per_video_p25']:.1f} – {r['events_per_video_p75']:.1f} |")
        lines.append(f"| Events / video (max) | {int(r['events_per_video_max'])} |")
        lines.append(f"| Events / subject (mean) | {r['events_per_subject_mean']:.0f} |")
        lines.append(f"| Events / subject (median) | {r['events_per_subject_median']:.0f} |")
        lines.append(f"| Core smile time (total) | {r['total_core_sec']/3600:.2f} hrs |")
        lines.append(f"| Core smile time / video (mean) | {r['core_sec_per_video_mean']:.1f}s |")
        lines.append(f"| Core smile time / video (median) | {r['core_sec_per_video_median']:.1f}s |")
        lines.append("")

    # ── Annotation cost table (default filter params, vary context/overhead/rate)
    lines.append("## Annotation Time & Cost Projections\n")
    lines.append("Default filter params (1.8 / 0.5 / 0.5). "
                 "Clip time = core + N_events × (ctx_before + ctx_after). "
                 "Annotation time = clip time × overhead factor.\n")

    cost_sub = df_proj[
        (df_proj["intensity"] == 1.8) &
        (df_proj["merge_dist"] == 0.5) &
        (df_proj["min_dur"] == 0.5)
    ].copy()

    if not cost_sub.empty:
        lines.append("### Clip time by context window\n")
        lines.append("| ctx_before | ctx_after | clip hours |")
        lines.append("|------------|-----------|------------|")
        clip_summary = cost_sub.groupby(["ctx_before", "ctx_after"])["clip_sec"].first()
        for (cb, ca), clip in clip_summary.items():
            lines.append(f"| {cb:.0f}s | {ca:.0f}s | {clip/3600:.1f} |")
        lines.append("")

        lines.append("### Annotation cost (ctx 3+2s)\n")
        lines.append("| overhead | $15/hr | $20/hr | $25/hr | $30/hr |")
        lines.append("|----------|--------|--------|--------|--------|")
        cost_ctx = cost_sub[
            (cost_sub["ctx_before"] == 3.0) & (cost_sub["ctx_after"] == 2.0)
        ]
        for factor in OVERHEAD_FACTORS:
            row_data = cost_ctx[cost_ctx["overhead_factor"] == factor]
            cells = []
            for rate in HOURLY_RATES:
                val = row_data[row_data["hourly_rate"] == rate]["annotation_cost_usd"]
                cells.append(f"${val.iloc[0]:,.0f}" if len(val) else "—")
            lines.append(f"| {factor:.1f}× | {' | '.join(cells)} |")
        lines.append("")

        lines.append("### Annotation cost (ctx 5+3s)\n")
        lines.append("| overhead | $15/hr | $20/hr | $25/hr | $30/hr |")
        lines.append("|----------|--------|--------|--------|--------|")
        cost_ctx2 = cost_sub[
            (cost_sub["ctx_before"] == 5.0) & (cost_sub["ctx_after"] == 3.0)
        ]
        for factor in OVERHEAD_FACTORS:
            row_data = cost_ctx2[cost_ctx2["overhead_factor"] == factor]
            cells = []
            for rate in HOURLY_RATES:
                val = row_data[row_data["hourly_rate"] == rate]["annotation_cost_usd"]
                cells.append(f"${val.iloc[0]:,.0f}" if len(val) else "—")
            lines.append(f"| {factor:.1f}× | {' | '.join(cells)} |")
        lines.append("")

    # ── Sensitivity analysis ─────────────────────────────────────────────
    lines.append("## Sensitivity Analysis\n")
    lines.append("Which parameter has the largest marginal effect on event count?\n")
    for param, label in [("intensity", "intensityThreshold"),
                          ("merge_dist", "mergeDistance"),
                          ("min_dur", "minDuration")]:
        grouped = df_core.groupby(param)["total_events"].mean()
        lo, hi = grouped.iloc[0], grouped.iloc[-1]
        pct = (hi - lo) / lo * 100 if lo else 0
        lines.append(f"- **{label}**: {int(lo):,} → {int(hi):,} events "
                      f"({pct:+.0f}% from lowest to highest setting)")
    lines.append("")

    # ── Figures ──────────────────────────────────────────────────────────
    lines.append("## Figures\n")
    for fname, caption in [
        ("smiling_sweep_heatmaps.png",
         "Heatmaps of events (total, per-video, videos with events) "
         "across intensity × merge distance, averaged over minDuration"),
        ("smiling_sweep_clip_time.png",
         "Core clip time vs intensity threshold (merge=0.5s, lines per minDuration)"),
        ("smiling_sweep_event_count.png",
         "Event count vs intensity threshold (minDur=0.5s, lines per merge distance)"),
        ("smiling_sweep_cost.png",
         "Annotation cost projections (default filter params)"),
        ("smiling_sweep_distribution.png",
         "Event distribution per video and per subject (default params)"),
    ]:
        lines.append(f"### {caption}\n")
        lines.append(f"![{caption}](figures/{fname})\n")

    report = "\n".join(lines)
    report_path = OUT_DIR / "smiling_sweep_report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading smiling segment data …")
    videos = load_all()
    n_videos = len(videos)
    subjects = {v["subject"] for v in videos.values() if v["subject"] is not None}
    n_subjects = len(subjects)
    print(f"  {n_videos} videos, {n_subjects} subjects")

    df_core = run_sweep(videos)
    df_core.to_csv(OUT_DIR / "smiling_sweep_core.csv", index=False)
    print(f"Core results → {OUT_DIR / 'smiling_sweep_core.csv'}")

    print("Projecting annotation time & cost …")
    df_proj = project_annotation(df_core)
    df_proj.to_csv(OUT_DIR / "smiling_sweep_projections.csv", index=False)
    print(f"Projections → {OUT_DIR / 'smiling_sweep_projections.csv'}  "
          f"({len(df_proj):,} rows)")

    print("Generating figures …")
    make_figures(df_core, df_proj)

    print("Writing report …")
    write_report(df_core, df_proj, n_videos, n_subjects)

    print("\nDone.")


if __name__ == "__main__":
    main()
