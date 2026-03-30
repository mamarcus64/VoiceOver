#!/usr/bin/env python3
"""Generate stratification visualisation suite from smile_features.csv.

Each plot addresses: "How does smiling behaviour change with variable X?"
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "brown": "#937860",
    "pink": "#DA8BC3",
    "gray": "#8C8C8C",
    "yellow": "#CCB974",
    "cyan": "#64B5CD",
}
C = list(PALETTE.values())


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ─── Individual plots ───────────────────────────────────────────

def plot_01_score_distribution(df):
    """Detection confidence: full model vs fallback."""
    fig, ax = plt.subplots()
    full = df.loc[df["model"] == "full", "score"]
    fall = df.loc[df["model"] == "fallback", "score"]
    bins = np.linspace(0.45, 1.0, 56)
    ax.hist(full, bins=bins, alpha=0.7, label=f"full ({len(full):,})", color=C[0])
    ax.hist(fall, bins=bins, alpha=0.7, label=f"fallback ({len(fall):,})", color=C[1])
    ax.set_xlabel("Detection score")
    ax.set_ylabel("Count")
    ax.set_title("Smile detection score distribution (full vs. fallback model)")
    ax.legend()
    ax.annotate(f"median full={full.median():.3f}\nmedian fallback={fall.median():.3f}",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, color="gray")
    save(fig, "01_score_distribution")


def plot_02_duration(df):
    """How long do detected smiles last?"""
    fig, ax = plt.subplots()
    dur = df["duration"].clip(upper=30)
    ax.hist(dur, bins=100, color=C[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Smile duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Smile duration distribution (clipped at 30 s)")
    med = df["duration"].median()
    ax.axvline(med, color=C[3], ls="--", label=f"median = {med:.2f}s")
    ax.legend()
    save(fig, "02_duration")


def plot_03_smiles_per_video(df):
    """How variable is the number of smiles across videos?"""
    vc = df.groupby("video_id").size()
    fig, ax = plt.subplots()
    ax.hist(vc, bins=80, color=C[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Smiles per video")
    ax.set_ylabel("Number of videos")
    ax.set_title("Distribution of smile counts per video")
    ax.annotate(f"n={len(vc):,} videos\nmedian={vc.median():.0f}  mean={vc.mean():.1f}",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9, color="gray")
    save(fig, "03_smiles_per_video")


def plot_04_smile_density(df):
    """Smile rate (per minute) across videos."""
    vid = df.drop_duplicates("video_id")[["video_id", "smile_density_per_min"]].dropna()
    fig, ax = plt.subplots()
    ax.hist(vid["smile_density_per_min"].clip(upper=10), bins=80, color=C[2], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Smiles per minute")
    ax.set_ylabel("Number of videos")
    ax.set_title("Smile density across videos (clipped at 10/min)")
    med = vid["smile_density_per_min"].median()
    ax.axvline(med, color=C[3], ls="--", label=f"median = {med:.2f}/min")
    ax.legend()
    save(fig, "04_smile_density")


def plot_05_position_in_tape(df):
    """Does smiling rate change through a tape?"""
    pos = df["position_in_tape"].dropna()
    fig, ax = plt.subplots()
    counts, edges = np.histogram(pos, bins=40, range=(0, 1))
    centres = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centres, counts, alpha=0.4, color=C[0])
    ax.plot(centres, counts, color=C[0], lw=2)
    ax.set_xlabel("Fractional position within tape (0=start, 1=end)")
    ax.set_ylabel("Smile count (proportional to rate)")
    ax.set_title("Smile density across tape position")
    ax.set_xlim(0, 1)
    save(fig, "05_position_in_tape")


def plot_06_interview_phase(df):
    """Does smile rate change between early / middle / late tapes?"""
    sub = df.dropna(subset=["video_duration"])
    vid_agg = sub.groupby("video_id").agg(
        tape_frac=("tape_frac", "first"),
        n_smiles=("video_id", "size"),
        dur_min=("video_duration", lambda x: x.iloc[0] / 60),
    )
    vid_agg["rate"] = vid_agg["n_smiles"] / vid_agg["dur_min"]

    def phase(f):
        if f < 0.34:
            return "early\n(pre-war)"
        elif f < 0.67:
            return "middle\n(during)"
        return "late\n(post-war)"

    vid_agg["phase"] = vid_agg["tape_frac"].apply(phase)
    order = ["early\n(pre-war)", "middle\n(during)", "late\n(post-war)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: box plot of per-video smile rates
    ax = axes[0]
    data = [vid_agg.loc[vid_agg["phase"] == p, "rate"].values for p in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, widths=0.5)
    for patch, c in zip(bp["boxes"], [C[0], C[1], C[2]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Smiles per minute (per video)")
    ax.set_title("Smile rate by interview phase")

    # Right: total smile count per phase
    ax = axes[1]
    counts = [vid_agg.loc[vid_agg["phase"] == p, "n_smiles"].sum() for p in order]
    bars = ax.bar(order, counts, color=[C[0], C[1], C[2]], alpha=0.7, edgecolor="white")
    for bar, ct in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{ct:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total smile count")
    ax.set_title("Total smiles by interview phase")

    fig.suptitle("Interview phase (tape position as chronological proxy)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "06_interview_phase")


def plot_07_speaking_context(df):
    """Who is speaking when the survivor smiles?"""
    ctx = df["speaking_context"].dropna()
    ctx = ctx[ctx != "unknown"]
    counts = ctx.value_counts()

    labels_map = {
        "interviewee_speaking": "Interviewee\nspeaking",
        "interviewer_speaking": "Interviewer\nspeaking",
        "pause_question": "Pause after\nquestion",
    }

    fig, ax = plt.subplots()
    x_labels = [labels_map.get(k, k) for k in counts.index]
    colors = [C[0], C[1], C[4]][:len(counts)]
    bars = ax.bar(x_labels, counts.values, color=colors, alpha=0.8, edgecolor="white")
    total = counts.sum()
    for bar, ct in zip(bars, counts.values):
        pct = 100 * ct / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{ct:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Smile count")
    ax.set_title("Speaking context when smiles occur")
    save(fig, "07_speaking_context")


def plot_08_time_to_interviewer(df):
    """How far are smiles from the last interviewer utterance?"""
    t = df["time_to_interviewer"].dropna()
    t = t[t < 120]  # clip at 2 min

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(t, bins=100, color=C[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Seconds since last interviewer speech ended")
    ax.set_ylabel("Smile count")
    ax.set_title("Time from last interviewer utterance")
    ax.axvline(10, color=C[3], ls="--", alpha=0.7, label="10s boundary")
    ax.legend()

    ax = axes[1]
    ax.hist(t.clip(upper=60), bins=60, color=C[2], edgecolor="white", linewidth=0.3, cumulative=True, density=True)
    ax.set_xlabel("Seconds since last interviewer speech")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF: how quickly after interviewer speech?")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)

    fig.suptitle("Distance to interviewer question (narrative vs. polite proxy)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "08_time_to_interviewer")


def plot_09_duchenne(df):
    """AU6 vs AU12: separating Duchenne from non-Duchenne smiles."""
    sub = df.dropna(subset=["au06_during", "au12_during"])
    if len(sub) < 100:
        print("  ⚠ Skipping Duchenne plot – insufficient AU data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = sub["au12_during"].clip(upper=4)
    y = sub["au06_during"].clip(upper=4)
    ax.hexbin(x, y, gridsize=50, cmap="Blues", mincnt=1)
    ax.set_xlabel("AU12 (lip corner puller)")
    ax.set_ylabel("AU06 (cheek raiser)")
    ax.set_title("AU12 vs AU06 intensity during smiles")
    ax.axhline(0.7, color=C[3], ls="--", alpha=0.5)
    ax.axvline(1.0, color=C[3], ls="--", alpha=0.5)
    ax.text(3.5, 3.5, "Duchenne\nzone", ha="center", fontsize=9, color=C[3])
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Count")

    ax = axes[1]
    if "duchenne" in sub.columns:
        d = sub["duchenne"].dropna()
        vals = [d.sum(), len(d) - d.sum()]
        labels = [f"Duchenne\n({vals[0]:,.0f})", f"Non-Duchenne\n({vals[1]:,.0f})"]
        ax.pie(vals, labels=labels, colors=[C[2], C[8]], autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.set_title("Duchenne classification (AU06>0.7 & AU12>1.0)")

    fig.suptitle("Facial Action Unit analysis: genuine vs. social smiles", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "09_duchenne_scatter")


def plot_10_audio_valence(df):
    """What emotional context do smiles occur in?"""
    sub = df["audio_valence"].dropna()
    if len(sub) < 100:
        print("  ⚠ Skipping audio valence – insufficient data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(sub, bins=80, color=C[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Audio valence")
    ax.set_ylabel("Smile count")
    ax.set_title("Audio valence at smile time")
    ax.axvline(sub.median(), color=C[3], ls="--", label=f"median = {sub.median():.3f}")
    ax.legend()

    # Valence × arousal scatter
    ax = axes[1]
    both = df.dropna(subset=["audio_valence", "audio_arousal"])
    if len(both) > 100:
        ax.hexbin(both["audio_valence"], both["audio_arousal"], gridsize=40, cmap="YlOrRd", mincnt=1)
        ax.set_xlabel("Audio valence")
        ax.set_ylabel("Audio arousal")
        ax.set_title("Valence × arousal at smile time")
        cb = plt.colorbar(ax.collections[0], ax=ax)
        cb.set_label("Count")

    fig.suptitle("Audio emotional context during smiles", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "10_audio_valence")


def plot_11_inter_smile_interval(df):
    """Do smiles cluster in bursts or occur in isolation?"""
    isi = df["inter_smile_interval"].dropna()
    isi = isi[isi > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(isi.clip(upper=120), bins=100, color=C[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Inter-smile interval (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Time between consecutive smiles (linear, clipped 120s)")
    ax.axvline(10, color=C[3], ls="--", alpha=0.7, label="10s (burst threshold)")
    ax.legend()

    ax = axes[1]
    log_isi = np.log10(isi.clip(lower=0.1))
    ax.hist(log_isi, bins=80, color=C[2], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log₁₀(inter-smile interval in seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-smile interval (log scale)")
    burst_frac = (isi < 10).mean()
    ax.annotate(f"{burst_frac:.1%} within 10s\n(burst smiling)", xy=(0.03, 0.95),
                xycoords="axes fraction", va="top", fontsize=9, color=C[3])

    fig.suptitle("Inter-smile interval: burst vs. isolated smiles", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "11_inter_smile_interval")


def plot_12_intensity_vs_duration(df):
    """Are intense smiles shorter or longer?"""
    sub = df.dropna(subset=["peak_r", "duration"])
    dur = sub["duration"].clip(upper=20)
    peak = sub["peak_r"].clip(upper=5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(dur, peak, c=sub["score"], cmap="viridis", s=1, alpha=0.3, rasterized=True)
    ax.set_xlabel("Duration (s, clipped 20s)")
    ax.set_ylabel("Peak AU12 ratio")
    ax.set_title("Intensity vs. duration (colored by score)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Detection score")

    ax = axes[1]
    dur_bins = pd.cut(dur, bins=[0, 1, 2, 3, 5, 10, 20], labels=["<1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-20s"])
    grouped = sub.assign(dur_bin=dur_bins).groupby("dur_bin")["peak_r"]
    means = grouped.mean()
    stds = grouped.std()
    x = range(len(means))
    ax.bar(x, means, yerr=stds, color=C[4], alpha=0.7, edgecolor="white", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=0)
    ax.set_xlabel("Duration bin")
    ax.set_ylabel("Mean peak AU12 ratio")
    ax.set_title("Peak intensity by duration bin (±1 SD)")

    fig.suptitle("Smile intensity vs. duration", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "12_intensity_vs_duration")


def plot_13_gender(df):
    """Does smile rate differ by gender?"""
    sub = df.dropna(subset=["gender", "video_duration"])
    if len(sub) < 100:
        print("  ⚠ Skipping gender plot – insufficient demographics")
        return

    vid_agg = sub.groupby(["video_id", "gender"]).agg(
        n_smiles=("video_id", "size"),
        dur_min=("video_duration", lambda x: x.iloc[0] / 60),
    ).reset_index()
    vid_agg["rate"] = vid_agg["n_smiles"] / vid_agg["dur_min"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    genders = sorted(vid_agg["gender"].unique())
    data = [vid_agg.loc[vid_agg["gender"] == g, "rate"].values for g in genders]
    bp = ax.boxplot(data, tick_labels=genders, patch_artist=True, showfliers=False, widths=0.5)
    for patch, c in zip(bp["boxes"], C):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Smiles per minute")
    ax.set_title("Smile rate by gender")

    ax = axes[1]
    for i, g in enumerate(genders):
        rates = vid_agg.loc[vid_agg["gender"] == g, "rate"]
        ax.hist(rates.clip(upper=8), bins=40, alpha=0.5, label=f"{g} (n={len(rates):,})", color=C[i])
    ax.set_xlabel("Smiles per minute")
    ax.set_ylabel("Number of videos")
    ax.set_title("Distribution of smile rates by gender")
    ax.legend()

    fig.suptitle("Gender and smile rate", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "13_gender_smile_rate")


def plot_14_birth_decade(df):
    """Does smile rate correlate with generation?"""
    sub = df.dropna(subset=["birth_year", "video_duration"])
    if len(sub) < 100:
        print("  ⚠ Skipping birth decade – insufficient demographics")
        return

    sub = sub.copy()
    sub["decade"] = (sub["birth_year"] // 10) * 10

    vid_agg = sub.groupby(["video_id", "decade"]).agg(
        n_smiles=("video_id", "size"),
        dur_min=("video_duration", lambda x: x.iloc[0] / 60),
    ).reset_index()
    vid_agg["rate"] = vid_agg["n_smiles"] / vid_agg["dur_min"]

    decade_stats = vid_agg.groupby("decade")["rate"].agg(["mean", "std", "count"]).reset_index()
    decade_stats = decade_stats[decade_stats["count"] >= 5]

    fig, ax = plt.subplots()
    ax.bar(decade_stats["decade"].astype(str), decade_stats["mean"],
           yerr=decade_stats["std"], color=C[4], alpha=0.7, edgecolor="white", capsize=4)
    for _, row in decade_stats.iterrows():
        ax.text(str(int(row["decade"])), row["mean"] + row["std"] + 0.05,
                f"n={int(row['count'])}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Birth decade")
    ax.set_ylabel("Mean smiles per minute (±1 SD)")
    ax.set_title("Smile rate by birth decade")
    save(fig, "14_birth_decade_smile_rate")


def plot_15_correlation_matrix(df):
    """Feature correlations to identify redundant dimensions."""
    cols = [
        "score", "duration", "peak_r", "mean_r",
        "position_in_tape", "tape_frac",
        "smile_density_per_min", "time_to_interviewer",
        "au06_during", "au12_during", "au09_during", "au04_during",
        "audio_valence", "audio_arousal", "audio_dominance",
        "gaze_valence", "gaze_arousal",
        "inter_smile_interval",
    ]
    present = [c for c in cols if c in df.columns]
    corr = df[present].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    short_names = [c.replace("_during", "").replace("audio_", "a_").replace("gaze_", "g_")
                   .replace("smile_density_per_min", "density").replace("position_in_tape", "pos_tape")
                   .replace("inter_smile_interval", "ISI").replace("time_to_interviewer", "t_to_int")
                   .replace("tape_frac", "interview_pos") for c in present]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(present)):
        for j in range(len(present)):
            val = corr.iloc[i, j]
            if abs(val) > 0.3:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    save(fig, "15_correlation_matrix")


def plot_16_duchenne_by_phase(df):
    """Does the fraction of Duchenne smiles change through the interview?"""
    sub = df.dropna(subset=["duchenne", "tape_frac"])
    if len(sub) < 100:
        print("  ⚠ Skipping Duchenne-by-phase – insufficient data")
        return

    bins = np.linspace(0, 1, 11)
    sub = sub.copy()
    sub["pos_bin"] = pd.cut(sub["tape_frac"], bins=bins)

    grouped = sub.groupby("pos_bin")["duchenne"].agg(["mean", "sum", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 20]
    centres = [(b.left + b.right) / 2 for b in grouped["pos_bin"]]

    fig, ax = plt.subplots()
    ax.plot(centres, grouped["mean"], "o-", color=C[2], lw=2, markersize=6)
    ax.fill_between(centres, grouped["mean"] - 0.02, grouped["mean"] + 0.02, alpha=0.15, color=C[2])
    ax.set_xlabel("Position in interview (0=start, 1=end)")
    ax.set_ylabel("Fraction of Duchenne smiles")
    ax.set_title("Duchenne smile fraction across interview timeline")
    ax.set_xlim(0, 1)
    save(fig, "16_duchenne_by_phase")


def plot_17_affect_by_context(df):
    """Audio valence broken down by speaking context."""
    sub = df.dropna(subset=["audio_valence", "speaking_context"])
    sub = sub[sub["speaking_context"].isin(["interviewee_speaking", "interviewer_speaking", "pause_question"])]
    if len(sub) < 100:
        print("  ⚠ Skipping affect-by-context – insufficient data")
        return

    labels_map = {
        "interviewee_speaking": "Interviewee\nspeaking",
        "interviewer_speaking": "Interviewer\nspeaking",
        "pause_question": "Pause",
    }
    order = ["interviewee_speaking", "interviewer_speaking", "pause_question"]

    fig, ax = plt.subplots()
    data = [sub.loc[sub["speaking_context"] == ctx, "audio_valence"].values for ctx in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(C[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels_map[k] for k in order])
    ax.set_ylabel("Audio valence")
    ax.set_title("Audio valence at smile time, by speaking context")
    for i, ctx in enumerate(order):
        n = (sub["speaking_context"] == ctx).sum()
        ax.text(i, ax.get_ylim()[0], f"n={n:,}", ha="center", va="top", fontsize=8, color="gray")
    save(fig, "17_affect_by_context")


def plot_18_smiles_per_subject(df):
    """How variable is total smiling across subjects?"""
    sc = df.groupby("subject_id").size()
    fig, ax = plt.subplots()
    ax.hist(sc.clip(upper=500), bins=80, color=C[5], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Total smiles per subject (clipped at 500)")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Smiles per subject (n={len(sc):,} subjects)")
    med = sc.median()
    ax.axvline(med, color=C[3], ls="--", label=f"median = {med:.0f}")
    ax.legend()
    save(fig, "18_smiles_per_subject")


def plot_19_model_vs_features(df):
    """Do full-model vs fallback smiles differ on other features?"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, label in zip(
        axes,
        ["duration", "peak_r", "au12_during"],
        ["Duration (s)", "Peak AU12 ratio", "AU12 during smile"],
    ):
        for model, color in [("full", C[0]), ("fallback", C[1])]:
            vals = df.loc[df["model"] == model, col].dropna()
            if col == "duration":
                vals = vals.clip(upper=15)
            elif col == "peak_r":
                vals = vals.clip(upper=5)
            elif col == "au12_during":
                vals = vals.clip(upper=4)
            ax.hist(vals, bins=60, alpha=0.5, label=f"{model} (n={len(vals):,})", color=color, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Full vs. fallback model: feature comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "19_model_vs_features")


def plot_20_tape_number_rate(df):
    """Raw tape number vs smile rate (not fractional)."""
    sub = df.dropna(subset=["video_duration"])
    vid_agg = sub.groupby("video_id").agg(
        tape_num=("tape_num", "first"),
        n_smiles=("video_id", "size"),
        dur_min=("video_duration", lambda x: x.iloc[0] / 60),
    ).reset_index()
    vid_agg["rate"] = vid_agg["n_smiles"] / vid_agg["dur_min"]

    tape_stats = vid_agg.groupby("tape_num")["rate"].agg(["mean", "std", "count"]).reset_index()
    tape_stats = tape_stats[tape_stats["count"] >= 10]
    tape_stats = tape_stats[tape_stats["tape_num"] <= 12]

    fig, ax = plt.subplots()
    ax.bar(tape_stats["tape_num"], tape_stats["mean"], yerr=tape_stats["std"],
           color=C[0], alpha=0.7, edgecolor="white", capsize=3)
    for _, row in tape_stats.iterrows():
        ax.text(row["tape_num"], row["mean"] + row["std"] + 0.05,
                f"n={int(row['count'])}", ha="center", fontsize=7, color="gray")
    ax.set_xlabel("Tape number")
    ax.set_ylabel("Mean smiles per minute (±1 SD)")
    ax.set_title("Smile rate by tape number (later tapes = later in interview)")
    save(fig, "20_tape_number_rate")


# ─── Main ───────────────────────────────────────────────────────

def main():
    setup_style()

    csv_path = Path(__file__).resolve().parent / "smile_features.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run build_features.py first.")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} smiles loaded\n")

    print("Generating plots:")
    plot_01_score_distribution(df)
    plot_02_duration(df)
    plot_03_smiles_per_video(df)
    plot_04_smile_density(df)
    plot_05_position_in_tape(df)
    plot_06_interview_phase(df)
    plot_07_speaking_context(df)
    plot_08_time_to_interviewer(df)
    plot_09_duchenne(df)
    plot_10_audio_valence(df)
    plot_11_inter_smile_interval(df)
    plot_12_intensity_vs_duration(df)
    plot_13_gender(df)
    plot_14_birth_decade(df)
    plot_15_correlation_matrix(df)
    plot_16_duchenne_by_phase(df)
    plot_17_affect_by_context(df)
    plot_18_smiles_per_subject(df)
    plot_19_model_vs_features(df)
    plot_20_tape_number_rate(df)

    print(f"\nDone! {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
