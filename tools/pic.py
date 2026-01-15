import numpy as np
import matplotlib.pyplot as plt

# --- data ---
baseline_sm1 = np.array([11912298,11910602,11910421,11921031,11920371,11913765,11913606,11912102], dtype=float)
opt_sm1      = np.array([5379651,5380035,5380086,5379969,5379410,5379701,5379628,5379520], dtype=float)
opt_sm0      = np.array([5379335,5379846,5380148,5379916,5380056,5379420,5379574,5379705], dtype=float)

# 2x8 matrices: rows=[SM0, SM1], cols=[warp0..7]
baseline = np.full((2,8), np.nan)
baseline[1,:] = baseline_sm1
opt = np.vstack([opt_sm0, opt_sm1])

def annotate(ax, mat, title):
    # use masked array so NaN shows as blank
    m = np.ma.masked_invalid(mat)
    im = ax.imshow(m, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"W{w}" for w in range(8)])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["SM0","SM1"])

    # per-row mean for deviation (ignore NaN)
    row_mean = np.nanmean(mat, axis=1)

    for r in range(2):
        for c in range(8):
            if np.isnan(mat[r,c]):
                ax.text(c, r, "inactive", ha="center", va="center", fontsize=8)
                continue
            v = mat[r,c]
            dev = (v/row_mean[r]-1.0)*100.0
            ax.text(c, r, f"{v/1e6:.3f}M\n{dev:+.3f}%", ha="center", va="center", fontsize=8)
    return im

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)

im0 = annotate(axs[0], baseline, "Baseline (1WG): only SM1 active")
im1 = annotate(axs[1], opt,      "Optimized (2WG): SM0+SM1 balanced")

# separate colorbars (scales differ a lot); looks cleaner
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Optional: add strong side notes in PPT, or print here
print("Optimized totals:", opt_sm0.sum(), opt_sm1.sum())

plt.savefig("sm_warp_balance_heatmap.png", dpi=300)
plt.show()
