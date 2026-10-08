import os
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
csv_path = 'model/runs/detect/train6/results.csv'  # Change to your run
save_path = 'model/runs/detect/train6/train7_metrics_plot.png'

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Metrics to Plot ===
metrics = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
]

# === Create Plot ===
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    if metric not in df.columns:
        continue
    ax = axes[idx]
    ax.plot(df[metric], 'o-', label='results')
    ax.plot(df[metric].rolling(window=5, min_periods=1).mean(), 'orange', linestyle='--', label='smooth')
    ax.set_title(metric)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
