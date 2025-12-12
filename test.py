import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ============================================================
# CONFIGURATION
# ============================================================

MAX_STEPS = 200
N_EVAL = 100                # <-- number of evaluation episodes
CONF_Z = 1.96              # 95% CI
DPI = 300

ENV_SIZES = [5, 10, 20]

REGIMES = {
    "Scarcity\n":  (8, 4),
    "Abundance\n": (4, 8),
    "Equivalent\n":   (8, 8),
}

# ============================================================
# SETUP
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir("plots")

# ============================================================
# LOAD DATA
# ============================================================

knn_res = pd.read_csv("replay_knn_results.csv")
win_res = pd.read_csv("replay_window_results.csv")



# ============================================================
# CI COMPUTATION
# ============================================================

def ci_from_std(std, n=N_EVAL):
    """95% confidence interval from sample std."""
    return CONF_Z * (std / sqrt(n))

# ============================================================
# METRIC EXTRACTION
# ============================================================

def build_metric_tables(knn_df, win_df, mean_col, std_col):
    rows, cols = len(ENV_SIZES), len(REGIMES)

    K_mean = np.full((rows, cols), np.nan)
    K_ci   = np.full((rows, cols), np.nan)
    W_mean = np.full((rows, cols), np.nan)
    W_ci   = np.full((rows, cols), np.nan)

    for i, g in enumerate(ENV_SIZES):
        for j, (regime_name, (a, it)) in enumerate(REGIMES.items()):

            k_rows = knn_df[
                (knn_df["grid"] == g) &
                (knn_df["agents"] == a) &
                (knn_df["items"] == it)
            ]

            w_rows = win_df[
                (win_df["grid"] == g) &
                (win_df["agents"] == a) &
                (win_df["items"] == it)
            ]

            if len(k_rows) == 0:
                raise ValueError(
                    f"[KNN missing] grid={g}, agents={a}, items={it}"
                )
            if len(w_rows) == 0:
                raise ValueError(
                    f"[WINDOW missing] grid={g}, agents={a}, items={it}"
                )

            k = k_rows.iloc[0]
            w = w_rows.iloc[0]

            K_mean[i, j] = k[mean_col]
            W_mean[i, j] = w[mean_col]

            K_ci[i, j] = ci_from_std(k[std_col])
            W_ci[i, j] = ci_from_std(w[std_col])

    return K_mean, K_ci, W_mean, W_ci


# ============================================================
# PLOTTING CORE
# ============================================================

def plot_factor_grid(
    K_mean, K_ci,
    W_mean, W_ci,
    ylabel,
    title,
    ylim,
    filename
):
    rows, cols = len(ENV_SIZES), len(REGIMES)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.6 * cols, 3.7 * rows),
        sharey=True
    )

    bar_w = 0.30
    x = np.array([0.0])
    offset = bar_w / 2

    def annotate_bar(ax, xpos, val, err):
        pad = 0.02 * (ylim[1] - ylim[0])
        label = f"{val:.2f}" if ylim[1] <= 1.1 else f"{val:.1f}"
        ax.text(
            xpos, val + err + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]

            # KNN
            ax.bar(
                x - offset,
                K_mean[i, j],
                bar_w,
                yerr=K_ci[i, j],
                capsize=6,
                label="KNN" if (i == 0 and j == 0) else None
            )

            # Window
            ax.bar(
                x + offset,
                W_mean[i, j],
                bar_w,
                yerr=W_ci[i, j],
                capsize=6,
                label="Window" if (i == 0 and j == 0) else None
            )

            # Annotations
            annotate_bar(ax, x[0] - offset, K_mean[i, j], K_ci[i, j])
            annotate_bar(ax, x[0] + offset, W_mean[i, j], W_ci[i, j])

            ax.set_xticks(x)
            ax.set_xticklabels([""], fontsize=11)

            if j == 0:
                ax.set_ylabel(
                    f"{ENV_SIZES[i]}×{ENV_SIZES[i]}\n{ylabel}",
                    fontsize=12
                )

            if i == 0:
                ax.set_title(
                    list(REGIMES.keys())[j],
                    fontsize=13
                )

            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=18, y=0.98)

    fig.legend(
        loc="upper right",
        frameon=False,
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"plots/{filename}.png", dpi=DPI)
    plt.close()

# ============================================================
# FIGURE 1 — COMPLETION RATE
# ============================================================

Kc, Kc_ci, Wc, Wc_ci = build_metric_tables(
    knn_res, win_res,
    mean_col="completion_mean",
    std_col="completion_std"
)

plot_factor_grid(
    Kc, Kc_ci,
    Wc, Wc_ci,
    ylabel="Completion Rate",
    title="Task Completion Across Environment Size and Resource Regime",
    ylim=(0.0, 1.05),
    filename="paper_completion_ci"
)

# ============================================================
# FIGURE 2 — TIMESTEPS
# ============================================================

Kt, Kt_ci, Wt, Wt_ci = build_metric_tables(
    knn_res, win_res,
    mean_col="timesteps_mean",
    std_col="timesteps_std"
)

plot_factor_grid(
    Kt, Kt_ci,
    Wt, Wt_ci,
    ylabel="Mean Timesteps",
    title="Episode Length Across Environment Size and Resource Regime",
    ylim=(0, MAX_STEPS),
    filename="paper_timesteps_ci"
)

# ============================================================
# FIGURE 3 — MEAN PAIRWISE DISTANCE
# ============================================================

Km, Km_ci, Wm, Wm_ci = build_metric_tables(
    knn_res, win_res,
    mean_col="mpd_mean",
    std_col="mpd_std"
)

plot_factor_grid(
    Km, Km_ci,
    Wm, Wm_ci,
    ylabel="Mean Pairwise Distance",
    title="Agent Dispersion Across Environment Size and Resource Regime",
    ylim=(0, 15),
    filename="paper_mpd_ci"
)

print("\n✔ All paper-quality plots with 95% CI saved to ./plots/\n")

def plot_per_grid_summary(
    grid_size,
    knn_df,
    win_df
):
    regimes = list(REGIMES.keys())
    cols = len(regimes)

    def extract(metric_mean, metric_std):
        K_mean, K_ci, W_mean, W_ci = [], [], [], []

        for regime_name, (a, it) in REGIMES.items():
            k_rows = knn_df[
                (knn_df.grid == grid_size) &
                (knn_df.agents == a) &
                (knn_df.items == it)
            ]

            w_rows = win_df[
                (win_df.grid == grid_size) &
                (win_df.agents == a) &
                (win_df.items == it)
            ]

            if len(k_rows) == 0:
                raise ValueError(
                    f"[KNN missing] grid={grid_size}, agents={a}, items={it}"
                )
            if len(w_rows) == 0:
                raise ValueError(
                    f"[WINDOW missing] grid={grid_size}, agents={a}, items={it}"
                )

            k = k_rows.iloc[0]
            w = w_rows.iloc[0]

            K_mean.append(k[metric_mean])
            W_mean.append(w[metric_mean])

            K_ci.append(ci_from_std(k[metric_std]))
            W_ci.append(ci_from_std(w[metric_std]))

        return np.array(K_mean), np.array(K_ci), np.array(W_mean), np.array(W_ci)

    # ---- Extract metrics ----
    Kc, Kc_ci, Wc, Wc_ci = extract("completion_mean", "completion_std")
    Kt, Kt_ci, Wt, Wt_ci = extract("timesteps_mean", "timesteps_std")
    Km, Km_ci, Wm, Wm_ci = extract("mpd_mean", "mpd_std")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    bar_w = 0.30
    x = np.arange(cols)
    offset = bar_w / 2

    def annotate(ax, xpos, val, err, ylim):
        pad = 0.02 * (ylim[1] - ylim[0])
        label = f"{val:.2f}" if ylim[1] <= 1.1 else f"{val:.1f}"
        ax.text(xpos, val + err + pad, label, ha="center", fontsize=10)

    plots = [
        (axes[0], Kc, Kc_ci, Wc, Wc_ci, "Completion Rate", (0, 1.05)),
        (axes[1], Kt, Kt_ci, Wt, Wt_ci, "Mean Timesteps", (0, MAX_STEPS)),
        (axes[2], Km, Km_ci, Wm, Wm_ci, "Mean Pairwise Distance", (0, 15)),
    ]

    for ax, K_m, K_ci_m, W_m, W_ci_m, ylabel, ylim in plots:
        ax.bar(x - offset, K_m, bar_w, yerr=K_ci_m, capsize=5, label="KNN")
        ax.bar(x + offset, W_m, bar_w, yerr=W_ci_m, capsize=5, label="Window")

        for i in range(cols):
            annotate(ax, x[i] - offset, K_m[i], K_ci_m[i], ylim)
            annotate(ax, x[i] + offset, W_m[i], W_ci_m[i], ylim)

        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(regimes, fontsize=11)

    fig.suptitle(
        f"Performance by Resource Regime ({grid_size}×{grid_size})",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        loc="upper right",
        frameon=False,
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"plots/paper_grid_{grid_size}_summary.png", dpi=DPI)
    plt.close()


for g in ENV_SIZES:
    plot_per_grid_summary(
        grid_size=g,
        knn_df=knn_res,
        win_df=win_res
    )
    print(f"✔ Summary plot for {g}×{g} saved to ./plots/")