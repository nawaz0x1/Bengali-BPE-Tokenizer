import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
SUBTEXT = "#8b949e"
GREEN = "#3fb950"  # ours
BLUE = "#388bfd"  # GPT-4
RED = "#f85149"  # GPT-2
YELLOW = "#e3b341"  # subtitles / callouts
ACCENT = "#f5c6a5"  # annotation highlight

labels = ["Bengali-BPE\n(ours)", "GPT-4\n(cl100k)", "GPT-2"]
clrs = [GREEN, BLUE, RED]


tokens = [1_046, 4_712, 7_663]  # lower is better
ratio = [3.63, 0.81, 0.50]  # higher is better
tpw = [2.03, 8.71, 13.65]  # lower is better

fig, axes = plt.subplots(1, 3, figsize=(26, 8))
fig.patch.set_facecolor(BG)


def styled_barh(ax, vals, xlabel, title, note, max_x, fmt, suffix=""):
    ax.set_facecolor(PANEL)
    bars = ax.barh(labels, vals, color=clrs, height=0.52, edgecolor=BORDER, linewidth=0.6)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + max_x * 0.015,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(v) + suffix,
            va="center",
            ha="left",
            color=TEXT,
            fontsize=22,
            fontweight="bold",
        )
    ax.set_xlim(0, max_x)
    ax.set_xlabel(xlabel, color="#c9d1d9", fontsize=22, labelpad=10)
    ax.set_title(title, color=TEXT, fontsize=26, fontweight="bold", pad=44)

    ax.text(
        0.5,
        1.015,
        note,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=YELLOW,
        fontsize=22,
        fontweight="bold",
        style="italic",
    )

    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(axis="x", colors="#c9d1d9", labelsize=22)
    ax.tick_params(axis="y", colors=TEXT, labelsize=22)
    ax.xaxis.set_tick_params(color=BORDER)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)


styled_barh(
    axes[0], tokens, "tokens", "Full-Corpus Token Count", "lower is better", 9_200, "{:,.0f}"
)

styled_barh(
    axes[1], ratio, "chars / token", "Compression Ratio", "higher is better", 4.6, "{:.2f}", "×"
)

styled_barh(
    axes[2],
    tpw,
    "tokens / Bengali word",
    "Avg Tokens per Bengali Word",
    "lower is better",
    17.5,
    "{:.2f}",
)

axes[0].text(
    tokens[0] + 200,
    -0.26,
    "7.33x fewer than GPT-2",
    color=YELLOW,
    fontsize=22,
    va="center",
    fontweight="bold",
)

fig.suptitle(
    "Bengali BPE Tokenizer  ·  Benchmark vs GPT-2 & GPT-4  (CC-100, 8k vocab)",
    color=TEXT,
    fontsize=28,
    fontweight="bold",
    y=1.03,
)

plt.tight_layout(rect=[0, 0, 1, 1])
out = "docs/benchmark.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"Saved {out}")
