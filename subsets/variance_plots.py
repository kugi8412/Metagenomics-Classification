import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_DIR = Path("eval")
OUT_DIR = EVAL_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RE_CLASS = re.compile(r"AUC-ROC\s+for\s+class\s+(.+?):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_AVG = re.compile(r"Average\s+AUC-ROC.*?:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

files = sorted(
    EVAL_DIR.glob("eval_*.txt"),
    key=lambda p: int(re.search(r"eval_(\d+)\.txt$", p.name).group(1)))

rows = []
run_avgs = []

for path in files:
    m = re.search(r"eval_(\d+)\.txt$", path.name)
    run_idx = int(m.group(1)) if m else None

    text = path.read_text(encoding="utf-8", errors="ignore")
    class_matches = RE_CLASS.findall(text)
    if not class_matches:
        continue

    for city, auc_str in class_matches:
        rows.append({
            "run": run_idx,
            "file": path.name,
            "city": city.strip(),
            "auc": float(auc_str)
        })

    avg_match = RE_AVG.search(text)
    if avg_match:
        run_avgs.append({
            "run": run_idx,
            "file": path.name,
            "avg_auc": float(avg_match.group(1))
        })

df = pd.DataFrame(rows)
df_avg = pd.DataFrame(run_avgs)

city_order = (df.groupby("city")["auc"].mean().sort_values(ascending=False).index.tolist())

# boxplot + punkty
fig, ax = plt.subplots(figsize=(max(10, len(city_order) * 1.2), 6))
data = [df.loc[df["city"] == c, "auc"].values for c in city_order]
ax.boxplot(data, labels=city_order, showfliers=False)

rng = np.random.default_rng(0)
for i, c in enumerate(city_order, start=1):
    y = df.loc[df["city"] == c, "auc"].values
    x = rng.normal(i, 0.06, size=len(y))
    ax.scatter(x, y, s=18, alpha=0.7)

ax.set_title("AUC-ROC per miasto — rozrzut między próbami")
ax.set_ylabel("AUC-ROC")
ax.set_ylim(-0.05, 1.05)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "auc_per_city_variance_boxplot.png", dpi=200)
plt.close(fig)

# slupki wariancji per miasto
fig, ax = plt.subplots(figsize=(max(10, len(city_order) * 1.2), 5))
var_by_city = df.groupby("city")["auc"].var(ddof=1).reindex(city_order)
ax.bar(city_order, var_by_city.values)

ax.set_title("Wariancja AUC-ROC per miasto (między próbami)")
ax.set_ylabel("VAR(AUC-ROC)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "auc_per_city_variance_bar.png", dpi=200)
plt.close(fig)

# srednie AUC-ROC między próbami

if not df_avg.empty:
    df_avg_sorted = df_avg.sort_values("run")

    mean_auc = df_avg_sorted["avg_auc"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df_avg_sorted["run"],
        df_avg_sorted["avg_auc"],
        marker="o",
        label="Average AUC-ROC per run"
    )

    ax.axhline(
        mean_auc,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_auc:.4f}"
    )

    ax.set_title("Average AUC-ROC — zmienność między próbami")
    ax.set_xlabel("Run (eval_N)")
    ax.set_ylabel("Average AUC-ROC")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_auc_across_runs_line_mean.png", dpi=200)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_avg_sorted["avg_auc"].values, bins=min(12, max(5, len(df_avg_sorted)//2)))
    ax.set_title("Histogram Average AUC-ROC (między próbami)")
    ax.set_xlabel("Average AUC-ROC")
    ax.set_ylabel("Liczba prób")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_auc_across_runs_hist.png", dpi=200)
    plt.close(fig)
