import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Setup
# ============================================================
MAX_STEPS = 200  # must match evaluation script

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir("plots")

def make_subplot_grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


# ============================================================
# Load CSV files
# ============================================================

def load_all(alg):
    """Load episode, update, and result files for a given algorithm."""
    ep = pd.read_csv(f"foraging_{alg}_episode_metrics.csv")
    upd = pd.read_csv(f"foraging_{alg}_update_metrics.csv")
    res = pd.read_csv(f"replay_{alg}_results.csv")
    return ep, upd, res


knn_ep, knn_upd, knn_res = load_all("knn")
win_ep, win_upd, win_res = load_all("window")


# ============================================================
# Helper functions
# ============================================================

def plot_metric_compare(knn_df, win_df, x_col, y_col,
                        ylabel, title, filename):
    """Generic line plot comparing KNN vs WINDOW."""
    plt.figure(figsize=(10, 5))
    plt.plot(knn_df[x_col], knn_df[y_col], label="KNN", alpha=0.9)
    plt.plot(win_df[x_col], win_df[y_col], label="Window", alpha=0.9)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()


# format replay environment labels correctly
def format_env_labels(df):
    """Convert grid/agents/items into a readable environment label."""
    labels = []
    for _, r in df.iterrows():
        g = int(r["grid"])
        a = int(r["agents"])
        it = int(r["items"])
        labels.append(f"{g}x{g}  | {a}A  | {it}I")
    return labels


# ============================================================
# UPDATE METRICS
# ============================================================

print("Plotting UPDATE metrics...")

plot_metric_compare(
    knn_upd, win_upd,
    x_col="update", y_col="approx_kl",
    ylabel="Approx KL", title="Approximate KL Divergence",
    filename="update_approx_kl"
)

plot_metric_compare(
    knn_upd, win_upd,
    x_col="update", y_col="policy_loss",
    ylabel="Policy Loss", title="Policy Loss Over Training",
    filename="update_policy_loss"
)

plot_metric_compare(
    knn_upd, win_upd,
    x_col="update", y_col="value_loss",
    ylabel="Value Loss", title="Value Loss Over Training",
    filename="update_value_loss"
)

plot_metric_compare(
    knn_upd, win_upd,
    x_col="update", y_col="entropy",
    ylabel="Entropy", title="Entropy Over Training",
    filename="update_entropy"
)



# ============================================================
# EPISODE METRICS
# ============================================================

print("Plotting EPISODE metrics...")

plot_metric_compare(
    knn_ep, win_ep,
    x_col="episode", y_col="episode_length",
    ylabel="Episode Length", title="Episode Length Over Time",
    filename="episode_length"
)

plot_metric_compare(
    knn_ep, win_ep,
    x_col="episode", y_col="env_steps",
    ylabel="Env Steps", title="Environment Steps Per Episode",
    filename="episode_env_steps"
)

plot_metric_compare(
    knn_ep, win_ep,
    x_col="episode", y_col="episode_return",
    ylabel="Episode Return", title="Episode Return Over Time",
    filename="episode_return"
)

plot_metric_compare(
    knn_ep, win_ep,
    x_col="episode", y_col="fraction_items_collected",
    ylabel="Fraction Items Collected",
    title="Fraction of Items Collected",
    filename="episode_fraction_items"
)

plot_metric_compare(
    knn_ep, win_ep,
    x_col="episode", y_col="mean_pairwise_dist",
    ylabel="Mean Pairwise Distance",
    title="Mean Pairwise Distance Over Time",
    filename="episode_mean_pairwise"
)



# ============================================================
# BUILD ENV LABELS
# ============================================================

def build_env_labels(df):
    """
    Turns (grid, agents, items) into readable strings:
    e.g., "10x10 | 8A | 8I"
    """
    labels = []
    for _, row in df.iterrows():
        g = int(row["grid"])
        a = int(row["agents"])
        it = int(row["items"])
        labels.append(f"{g}x{g} | {a} Agents | {it} Items")
    return labels

knn_res["env"] = build_env_labels(knn_res)
win_res["env"] = build_env_labels(win_res)

envs = knn_res["env"].tolist()


# Helper to make subplot titles readable
def clean_env_label(env):
    return env.replace(" | ", "\n")


# ============================================================
# EXTRACT METRICS
# ============================================================

# Completion
K_comp = knn_res["completion_mean"].tolist()
W_comp = win_res["completion_mean"].tolist()

# Timesteps
K_time = knn_res["timesteps_mean"].tolist()
W_time = win_res["timesteps_mean"].tolist()

# Mean Pairwise Distance
K_mpd = knn_res["mpd_mean"].tolist()
W_mpd = win_res["mpd_mean"].tolist()


# ============================================================
# FIGURE 1 — COMPLETION RATE
# ============================================================

rows, cols = make_subplot_grid(len(envs))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

for i, ax in enumerate(axes[:len(envs)]):
    ax.bar(["KNN", "Window"], [K_comp[i], W_comp[i]])
    ax.set_ylim(0.0, 1.0)   # <-- FIXED
    ax.set_title(f"Completion\n{clean_env_label(envs[i])}", fontsize=12)
    ax.set_ylabel("Completion Rate")
    ax.grid(axis="y", alpha=0.3)

# Disable unused subplots
for ax in axes[len(envs):]:
    ax.axis("off")

plt.suptitle("Completion Rate Across All Scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/summary_completion.png")
plt.close()

# ============================================================
# FIGURE 2 — MEAN TIMESTEPS
# ============================================================

rows, cols = make_subplot_grid(len(envs))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

for i, ax in enumerate(axes[:len(envs)]):
    ax.bar(["KNN", "Window"], [K_time[i], W_time[i]])
    ax.set_ylim(0, MAX_STEPS)   # <-- FIXED
    ax.set_title(f"Timesteps\n{clean_env_label(envs[i])}", fontsize=12)
    ax.set_ylabel("Mean Timesteps")
    ax.grid(axis="y", alpha=0.3)

for ax in axes[len(envs):]:
    ax.axis("off")

plt.suptitle("Timesteps Across All Scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/summary_timesteps.png")
plt.close()


# ============================================================
# FIGURE 3 — MEAN PAIRWISE DISTANCE
# ============================================================

rows, cols = make_subplot_grid(len(envs))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

for i, ax in enumerate(axes[:len(envs)]):
    ax.bar(["KNN", "Window"], [K_mpd[i], W_mpd[i]])
    ax.set_ylim(0, 15)   # <-- FIXED
    ax.set_title(f"Mean Pairwise Distance\n{clean_env_label(envs[i])}", fontsize=12)
    ax.set_ylabel("Mean Pairwise Distance")
    ax.grid(axis="y", alpha=0.3)
    
for ax in axes[len(envs):]:
    ax.axis("off")
    
plt.suptitle("Mean Pairwise Distance Across All Scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/summary_mpd.png")
plt.close()

print("\nAll plots saved under ./plots/\n")
