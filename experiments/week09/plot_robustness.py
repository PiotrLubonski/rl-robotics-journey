"""Tydzień 9: wykres odporności do README (z runs/week09/robustness.csv)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path("runs/week09/robustness.csv")
OUT = Path("docs/img/week09_robustness.png")

# kolejność = kolejność kolorów; kolor idzie za modelem, nie za miejscem
MODELS = {
    "reczny_regulator": ("ręczny regulator", "#2a78d6"),
    "sac_arm_cossin_vel1_50000_seed0": ("SAC", "#eb6834"),
    "sac_arm_dr_noise0_50000_seed0": ("SAC + losowa fizyka", "#1baf7a"),
    "sac_arm_dr_50000_seed0": ("SAC + losowa fizyka + szum", "#eda100"),
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"

with open(SRC, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
scenarios = list(dict.fromkeys(r["scenariusz"] for r in rows))
success = {}  # ostatni wpis wygrywa, gdy coś liczono dwa razy
for r in rows:
    success.setdefault(r["model"], {})[r["scenariusz"]] = float(r["success_pct"])
models = [m for m in MODELS if m in success]

x = np.arange(len(scenarios))
width = 0.8 / len(models)
fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, m in enumerate(models):
    label, color = MODELS[m]
    values = [success[m].get(s, np.nan) for s in scenarios]
    ax.bar(
        x + (i - (len(models) - 1) / 2) * width,
        values,
        width,
        label=label,
        color=color,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )

ax.set_xticks(x, scenarios, color=INK)
ax.set_ylim(0, 105)
ax.set_ylabel("sukces na końcu epizodu (%)", color=INK)
ax.set_title(
    "ArmReach: ten sam model w zmienionych światach (200 epizodów na słupek)",
    color=INK,
    loc="left",
)
ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.tick_params(colors=MUTED, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(models),
    frameon=False,
    labelcolor=INK,
)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(f"zapisano {OUT}")
