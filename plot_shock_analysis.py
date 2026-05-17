"""
Plotting script for THEMIS shock analysis results.
Generates figures showing shock crossing characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the results
OUTPUT_CSV = "shock_window_scan_results_with_speed.csv"
df = pd.read_csv(OUTPUT_CSV)

# Create figure directory
import os
os.makedirs("figures", exist_ok=True)

# ==========================================================
# Figure 1: Score Distribution
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["score"], bins=30, alpha=0.7, color="blue", edgecolor="black")
ax.axvline(df["score"].max(), color="red", linestyle="--", linewidth=2, label=f"Max: {df['score'].max():.3f}")
ax.set_xlabel("Score (λ₂/λ₃ / (1 + angle))", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Shock Window Scores", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/01_score_distribution.png", dpi=150)
plt.close()
print("✓ Saved: 01_score_distribution.png")

# ==========================================================
# Figure 2: Lambda2/Lambda3 vs Angle
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df["angle_MVA_CP_deg"], df["lambda2_lambda3"], 
                     c=df["score"], cmap="viridis", s=50, alpha=0.6, edgecolors="black", linewidth=0.5)
ax.set_xlabel("Angle between MVA and CP normals (degrees)", fontsize=12)
ax.set_ylabel("λ₂/λ₃ Ratio", fontsize=12)
ax.set_title("MVA Quality vs Normal Agreement", fontsize=14, fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Score", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/02_lambda_vs_angle.png", dpi=150)
plt.close()
print("✓ Saved: 02_lambda_vs_angle.png")

# ==========================================================
# Figure 3: Shock Speed Distribution
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6))
valid_speeds = df[df["Vsh_kms"].notna()]["Vsh_kms"]
ax.hist(valid_speeds, bins=30, alpha=0.7, color="green", edgecolor="black")
if len(valid_speeds) > 0:
    ax.axvline(valid_speeds.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {valid_speeds.mean():.1f} km/s")
ax.set_xlabel("Shock Speed (km/s)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Computed Shock Speeds (Rankine-Hugoniot)", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/03_shock_speed_distribution.png", dpi=150)
plt.close()
print("✓ Saved: 03_shock_speed_distribution.png")

# ==========================================================
# Figure 4: Top 10 Results
# ==========================================================
df_top10 = df.nlargest(10, "score")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Score
ax = axes[0, 0]
ax.barh(range(len(df_top10)), df_top10["score"].values, color="steelblue")
ax.set_yticks(range(len(df_top10)))
ax.set_yticklabels([f"{i+1}" for i in range(len(df_top10))])
ax.set_xlabel("Score", fontsize=11)
ax.set_ylabel("Rank", fontsize=11)
ax.set_title("Top 10: Scores", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Subplot 2: Lambda2/Lambda3
ax = axes[0, 1]
ax.barh(range(len(df_top10)), df_top10["lambda2_lambda3"].values, color="darkgreen")
ax.set_yticks(range(len(df_top10)))
ax.set_yticklabels([f"{i+1}" for i in range(len(df_top10))])
ax.set_xlabel("λ₂/λ₃", fontsize=11)
ax.set_ylabel("Rank", fontsize=11)
ax.set_title("Top 10: Lambda2/Lambda3", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Subplot 3: Angle
ax = axes[1, 0]
ax.barh(range(len(df_top10)), df_top10["angle_MVA_CP_deg"].values, color="darkred")
ax.set_yticks(range(len(df_top10)))
ax.set_yticklabels([f"{i+1}" for i in range(len(df_top10))])
ax.set_xlabel("Angle (degrees)", fontsize=11)
ax.set_ylabel("Rank", fontsize=11)
ax.set_title("Top 10: MVA-CP Angle", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Subplot 4: Shock Speed
ax = axes[1, 1]
vsh_top10 = df_top10["Vsh_kms"].values
colors_vsh = ["steelblue" if not np.isnan(v) else "gray" for v in vsh_top10]
ax.barh(range(len(df_top10)), [v if not np.isnan(v) else 0 for v in vsh_top10], color=colors_vsh)
ax.set_yticks(range(len(df_top10)))
ax.set_yticklabels([f"{i+1}" for i in range(len(df_top10))])
ax.set_xlabel("Shock Speed (km/s)", fontsize=11)
ax.set_ylabel("Rank", fontsize=11)
ax.set_title("Top 10: Shock Speeds", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Top 10 Shock Window Candidates", fontsize=16, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig("figures/04_top10_results.png", dpi=150)
plt.close()
print("✓ Saved: 04_top10_results.png")

# ==========================================================
# Figure 5: Best Result - Normal Vectors
# ==========================================================
best = df.nlargest(1, "score").iloc[0]

fig = plt.figure(figsize=(12, 5))

# MVA Normal
ax1 = fig.add_subplot(121, projection="3d")
n_mva = np.array([best["MVA_nx"], best["MVA_ny"], best["MVA_nz"]])
ax1.quiver(0, 0, 0, n_mva[0], n_mva[1], n_mva[2], arrow_length_ratio=0.1, color="blue", linewidth=3, label="MVA Normal")
n_cp = np.array([best["CP_nx"], best["CP_ny"], best["CP_nz"]])
ax1.quiver(0, 0, 0, n_cp[0], n_cp[1], n_cp[2], arrow_length_ratio=0.1, color="red", linewidth=3, label="CP Normal")
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.set_xlabel("X", fontsize=11)
ax1.set_ylabel("Y", fontsize=11)
ax1.set_zlabel("Z", fontsize=11)
ax1.set_title("Shock Normal Directions (Best Result)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Shock velocity vector
ax2 = fig.add_subplot(122, projection="3d")
vsh_vec = np.array([best["Vsh_vec_x"], best["Vsh_vec_y"], best["Vsh_vec_z"]])
v_mag = np.linalg.norm(vsh_vec)
if v_mag > 0:
    vsh_normalized = vsh_vec / v_mag
else:
    vsh_normalized = np.array([0, 0, 0])
ax2.quiver(0, 0, 0, vsh_normalized[0], vsh_normalized[1], vsh_normalized[2], 
           arrow_length_ratio=0.1, color="green", linewidth=3, label=f"Shock Velocity\n({best['Vsh_kms']:.1f} km/s)")
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.set_xlabel("X", fontsize=11)
ax2.set_ylabel("Y", fontsize=11)
ax2.set_zlabel("Z", fontsize=11)
ax2.set_title("Shock Velocity Direction (Best Result)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.suptitle(f"Best Shock Crossing: Score = {best['score']:.4f}, Angle = {best['angle_MVA_CP_deg']:.2f}°", 
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/05_best_result_vectors.png", dpi=150)
plt.close()
print("✓ Saved: 05_best_result_vectors.png")

# ==========================================================
# Figure 6: Plasma Parameters (Best Result)
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Upstream vs Downstream Density
ax = axes[0, 0]
densities = [best["n1_upstream"], best["n2_downstream"]]
ax.bar(["Upstream", "Downstream"], densities, color=["blue", "red"], alpha=0.7, edgecolor="black")
ax.set_ylabel("Density (cm⁻³)", fontsize=11)
ax.set_title("Ion Density (Best Result)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(densities):
    ax.text(i, v + 0.1, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

# Velocity comparison
ax = axes[0, 1]
V1 = np.array([best["V1x"], best["V1y"], best["V1z"]])
V2 = np.array([best["V2x"], best["V2y"], best["V2z"]])
vel_mag1 = np.linalg.norm(V1)
vel_mag2 = np.linalg.norm(V2)
ax.bar(["Upstream", "Downstream"], [vel_mag1, vel_mag2], color=["blue", "red"], alpha=0.7, edgecolor="black")
ax.set_ylabel("Bulk Velocity Magnitude (km/s)", fontsize=11)
ax.set_title("Ion Bulk Velocity (Best Result)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([vel_mag1, vel_mag2]):
    ax.text(i, v + 10, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

# Magnetic field comparison
ax = axes[1, 0]
Bu = np.array([best["Bu_x"], best["Bu_y"], best["Bu_z"]])
Bd = np.array([best["Bd_x"], best["Bd_y"], best["Bd_z"]])
B_mag1 = np.linalg.norm(Bu)
B_mag2 = np.linalg.norm(Bd)
ax.bar(["Upstream", "Downstream"], [B_mag1, B_mag2], color=["blue", "red"], alpha=0.7, edgecolor="black")
ax.set_ylabel("Magnetic Field Magnitude (nT)", fontsize=11)
ax.set_title("Mean Magnetic Field (Best Result)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([B_mag1, B_mag2]):
    ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

# Eigenvalues
ax = axes[1, 1]
evals = [best["lambda1"], best["lambda2"], best["lambda3"]]
ax.semilogy(["λ₁", "λ₂", "λ₃"], evals, "o-", color="purple", linewidth=2, markersize=10)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("MVA Eigenvalues (Best Result)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, which="both")
for i, v in enumerate(evals):
    ax.text(i, v * 1.5, f"{v:.3e}", ha="center", fontsize=9)

plt.suptitle("Best Result: Plasma & Magnetic Field Parameters", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/06_best_plasma_parameters.png", dpi=150)
plt.close()
print("✓ Saved: 06_best_plasma_parameters.png")

# ==========================================================
# Figure 7: Window Parameter Sensitivity
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Score vs Upstream window
ax = axes[0]
for down_sec in sorted(df["downstream_sec"].unique()):
    data = df[df["downstream_sec"] == down_sec]
    data_sorted = data.sort_values("upstream_sec")
    ax.plot(data_sorted["upstream_sec"], data_sorted["score"].groupby(data_sorted["upstream_sec"]).mean(), 
            marker="o", label=f"Downstream: {down_sec}s", linewidth=2, markersize=6)
ax.set_xlabel("Upstream Window (seconds)", fontsize=12)
ax.set_ylabel("Mean Score", fontsize=12)
ax.set_title("Score Sensitivity to Window Parameters", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Score vs Downstream window
ax = axes[1]
for up_sec in sorted(df["upstream_sec"].unique()):
    data = df[df["upstream_sec"] == up_sec]
    data_sorted = data.sort_values("downstream_sec")
    ax.plot(data_sorted["downstream_sec"], data_sorted["score"].groupby(data_sorted["downstream_sec"]).mean(), 
            marker="s", label=f"Upstream: {up_sec}s", linewidth=2, markersize=6)
ax.set_xlabel("Downstream Window (seconds)", fontsize=12)
ax.set_ylabel("Mean Score", fontsize=12)
ax.set_title("Score Sensitivity to Window Parameters", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("Impact of Window Sizes on Analysis Quality", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/07_window_sensitivity.png", dpi=150)
plt.close()
print("✓ Saved: 07_window_sensitivity.png")

print("\n" + "="*50)
print("✓ All figures saved to ./figures/")
print("="*50)
