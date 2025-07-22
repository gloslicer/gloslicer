import os
import json
import numpy as np
import matplotlib.pyplot as plt

result_path = './experiment/figs/figure/result.json'
output_path = './experiment/figs/figure/slicing_metrics_heatmaps.png'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

models_json = [
    "deepseek-v3",
    "deepseek-r1",
    "gemini-2.0-flash",
    "gemini-2.0-pro-exp",
    "gpt-4o"
]
model_names = [
    "DS V3", "DS R1",
    "G-2.0-F", "G-2.0-P", "GPT-4o"
]
prompts = ["zero-shot", "one-shot", "cot"]
prompt_short = ["Z-S", "O-S", "CoT"]

# ===== 读取 result.json =====
with open(result_path, 'r', encoding='utf-8') as f:
    summary = json.load(f)

def to_matrix(direction, metric):
    mat = np.zeros((len(models_json), len(prompts)))
    for i, m in enumerate(models_json):
        for j, p in enumerate(prompts):
            found = [r for r in summary if r['model'] == m and r['prompt'] == p]
            if found:
                mat[i, j] = found[0][direction][metric]
            else:
                mat[i, j] = 0.0
    return mat

forward_precision = to_matrix("forward", "precision")
forward_recall = to_matrix("forward", "recall")
forward_f1 = to_matrix("forward", "f1")
backward_precision = to_matrix("backward", "precision")
backward_recall = to_matrix("backward", "recall")
backward_f1 = to_matrix("backward", "f1")

heatmaps = [
    (forward_precision, "F-P"),
    (forward_recall, "F-R"),
    (forward_f1, "F-F1"),
    (backward_precision, "B-P"),
    (backward_recall, "B-R"),
    (backward_f1, "B-F1"),
]

fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.5), sharex=True, sharey=False)
vmin, vmax = 0, 1
cmap = "YlGnBu"

for idx, (data, title) in enumerate(heatmaps):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(prompts)))
    ax.set_xticklabels(prompt_short, fontsize=5)
    ax.set_yticks(np.arange(len(model_names)))
    if col == 0:
        ax.set_yticklabels(model_names, fontsize=5)
    else:
        ax.set_yticklabels([''] * len(model_names))
    for i in range(len(model_names)):
        for j in range(len(prompts)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="black" if data[i, j] < 0.5 else "white", fontsize=4, fontweight='bold')
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(axis='both', which='major', length=0)

fig.subplots_adjust(left=0.21, right=0.88, wspace=0.15, hspace=0.27)
cbar_ax = fig.add_axes([0.91, 0.19, 0.015, 0.60])
cbar = fig.colorbar(im, cax=cbar_ax, label='Score', fraction=0.05, pad=0.01)
cbar.set_label('Score', fontsize=6)
cbar.ax.tick_params(labelsize=5)

plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Figure generated: {output_path}")
