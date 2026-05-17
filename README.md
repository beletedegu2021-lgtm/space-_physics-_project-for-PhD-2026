# space-_physics-_project-for-PhD-2026
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pyspedas
from pytplot import get_data, tplot_names
from scipy.linalg import eigh

# ==========================================================
# USER SETTINGS
# ==========================================================
PROBE = "d"   # THEMIS probe:  b, c, d, e
TRANGE = ["2007-07-26/16:00", "2007-07-26/17:00"]

OUTPUT_CSV = "shock_window_scan_results_with_speed.csv"
OUTPUT_REPORT = "belete_THB_data11.txt"

# Candidate shock crossing time (approximate)
T0 = "2007-07-26 16:00:00"

# Scan settings
UPSTREAM_WINDOWS = [30, 60, 90, 120]      # seconds
DOWNSTREAM_WINDOWS = [30, 60, 90, 120]    # seconds
OFFSET_LIST = np.arange(-120, 121, 10)    # seconds

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def str_to_datetime(tstr):
    return datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")

def datetime_to_unix(dt):
    return dt.timestamp()

def normalize(vec):
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n

def angle_deg(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def select_interval(time_arr, data_arr, t_start, t_end):
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    if np.sum(mask) < 5:
        return None
    return data_arr[mask]

# ==========================================================
# MVA FUNCTION
# ==========================================================
def compute_mva(B):
    Bmean = np.mean(B, axis=0)
    dB = B - Bmean

    M = np.zeros((3, 3))
    for i in range(len(dB)):
        M += np.outer(dB[i], dB[i])
    M /= len(dB)

    evals, evecs = eigh(M)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    n_mva = evecs[:, -1]  # minimum variance direction
    return evals, evecs, normalize(n_mva)

# ==========================================================
# COPLANARITY NORMAL FUNCTION
# ==========================================================
def compute_cp_normal(Bu, Bd):
    cross_B = np.cross(Bu, Bd)
    deltaB = Bd - Bu
    n_cp = np.cross(cross_B, deltaB)
    return normalize(n_cp)

# ==========================================================
# SHOCK SPEED (Rankine-Hugoniot mass continuity)
# ==========================================================
def compute_shock_speed_rh(n1, V1, n2, V2, n_hat):
    """
    Returns shock speed along normal in spacecraft frame (km/s).
    Uses: Vsh = (n2*V2n - n1*V1n)/(n2-n1)
    """
    V1n = np.dot(V1, n_hat)
    V2n = np.dot(V2, n_hat)

    if abs(n2 - n1) < 1e-6:
        return np.nan

    Vsh = (n2 * V2n - n1 * V1n) / (n2 - n1)
    return Vsh

# ==========================================================
# MAIN SCRIPT
# ==========================================================
print("Downloading THEMIS FGM data...")
pyspedas.themis.fgm(trange=TRANGE, probe=PROBE, level="l2", time_clip=True)

print("Downloading THEMIS ESA ion moments...")
pyspedas.themis.esa(trange=TRANGE, probe=PROBE, level="l2", time_clip=True)

# ==========================================================
# LOAD FGM DATA
# ==========================================================
fgm_name = f"th{PROBE}_fgs_gsm"
fgm = get_data(fgm_name)

if fgm is None:
    print("Available variables:")
    print(tplot_names())
    raise RuntimeError(f"FGM data not found: {fgm_name}")

t_fgm = fgm.times
B_fgm = fgm.y
print("FGM loaded:", B_fgm.shape)

# ==========================================================
# LOAD ESA MOMENT DATA
# ==========================================================
dens_name = f"th{PROBE}_peif_density"
vel_name  = f"th{PROBE}_peif_velocity"

dens = get_data(dens_name)
vel  = get_data(vel_name)

if dens is None or vel is None:
    print("\nERROR: ESA density/velocity not found.")
    print("Available variables:")
    print(tplot_names())
    raise RuntimeError("Cannot compute shock speed without density and velocity moments.")

t_dens = dens.times
n_ion  = dens.y

t_vel = vel.times
V_ion = vel.y

print("ESA density loaded:", n_ion.shape)
print("ESA velocity loaded:", V_ion.shape)

# ==========================================================
# SCAN WINDOWS AROUND T0
# ==========================================================
t0_dt = str_to_datetime(T0)
results = []

print("\nScanning windows around t0 =", t0_dt)

for shift in OFFSET_LIST:
    t_cross = t0_dt + timedelta(seconds=int(shift))
    t_cross_unix = datetime_to_unix(t_cross)

    for up_sec in UPSTREAM_WINDOWS:
        for down_sec in DOWNSTREAM_WINDOWS:

            # intervals
            t1u = t_cross_unix - up_sec
            t2u = t_cross_unix - 5

            t1d = t_cross_unix + 5
            t2d = t_cross_unix + down_sec

            # Magnetic field windows
            Bu_data = select_interval(t_fgm, B_fgm, t1u, t2u)
            Bd_data = select_interval(t_fgm, B_fgm, t1d, t2d)

            # Plasma windows
            nu_data = select_interval(t_dens, n_ion, t1u, t2u)
            nd_data = select_interval(t_dens, n_ion, t1d, t2d)

            Vu_data = select_interval(t_vel, V_ion, t1u, t2u)
            Vd_data = select_interval(t_vel, V_ion, t1d, t2d)

            if Bu_data is None or Bd_data is None:
                continue
            if nu_data is None or nd_data is None:
                continue
            if Vu_data is None or Vd_data is None:
                continue

            Bu_avg = np.mean(Bu_data, axis=0)
            Bd_avg = np.mean(Bd_data, axis=0)

            n1 = float(np.mean(nu_data))
            n2 = float(np.mean(nd_data))

            V1 = np.mean(Vu_data, axis=0)
            V2 = np.mean(Vd_data, axis=0)

            # MVA across combined B interval
            B_combined = np.vstack([Bu_data, Bd_data])
            evals, evecs, n_mva = compute_mva(B_combined)

            lam1, lam2, lam3 = evals[0], evals[1], evals[2]
            ratio_23 = lam2 / lam3 if lam3 != 0 else np.nan

            # CP normal
            n_cp = compute_cp_normal(Bu_avg, Bd_avg)

            # Angle
            ang = angle_deg(n_mva, n_cp)

            # Shock speed using RH mass continuity
            Vsh = compute_shock_speed_rh(n1, V1, n2, V2, n_mva)

            # Shock speed vector
            Vsh_vec = Vsh * n_mva if not np.isnan(Vsh) else np.array([np.nan, np.nan, np.nan])

            results.append({
                "t_cross": t_cross,
                "shift_sec": shift,
                "upstream_sec": up_sec,
                "downstream_sec": down_sec,

                "lambda1": lam1,
                "lambda2": lam2,
                "lambda3": lam3,
                "lambda2_lambda3": ratio_23,

                "MVA_nx": n_mva[0],
                "MVA_ny": n_mva[1],
                "MVA_nz": n_mva[2],

                "CP_nx": n_cp[0],
                "CP_ny": n_cp[1],
                "CP_nz": n_cp[2],

                "angle_MVA_CP_deg": ang,

                "Bu_x": Bu_avg[0],
                "Bu_y": Bu_avg[1],
                "Bu_z": Bu_avg[2],

                "Bd_x": Bd_avg[0],
                "Bd_y": Bd_avg[1],
                "Bd_z": Bd_avg[2],

                "n1_upstream": n1,
                "n2_downstream": n2,

                "V1x": V1[0],
                "V1y": V1[1],
                "V1z": V1[2],

                "V2x": V2[0],
                "V2y": V2[1],
                "V2z": V2[2],

                "Vsh_kms": Vsh,
                "Vsh_vec_x": Vsh_vec[0],
                "Vsh_vec_y": Vsh_vec[1],
                "Vsh_vec_z": Vsh_vec[2]
            })

# ==========================================================
# SAVE RESULTS
# ==========================================================
df = pd.DataFrame(results)

if len(df) == 0:
    raise RuntimeError("No valid windows found. Expand TRANGE or scan settings.")

df["score"] = df["lambda2_lambda3"] / (1.0 + df["angle_MVA_CP_deg"])
df_sorted = df.sort_values(by="score", ascending=False)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved scan results to: {OUTPUT_CSV}")

# ==========================================================
# BEST RESULT
# ==========================================================
best = df_sorted.iloc[0]

# ==========================================================
# WRITE REPORT
# ==========================================================
with open(OUTPUT_REPORT, "w") as f:
    f.write("=== BEST SHOCK WINDOW RESULT (AUTO SCAN + RH SPEED) ===\n\n")
    f.write(f"Probe: THEMIS-{PROBE.upper()}\n")
    f.write(f"Overall TRANGE: {TRANGE}\n\n")

    f.write(f"Best crossing time: {best['t_cross']}\n")
    f.write(f"Shift from T0 (sec): {best['shift_sec']}\n")
    f.write(f"Upstream window (sec): {best['upstream_sec']}\n")
    f.write(f"Downstream window (sec): {best['downstream_sec']}\n\n")

    f.write("Eigenvalues:\n")
    f.write(f"lambda1 = {best['lambda1']:.6f}\n")
    f.write(f"lambda2 = {best['lambda2']:.6f}\n")
    f.write(f"lambda3 = {best['lambda3']:.6f}\n")
    f.write(f"lambda2/lambda3 = {best['lambda2_lambda3']:.6f}\n\n")

    f.write("Shock Normals:\n")
    f.write(f"MVA normal = [{best['MVA_nx']:.6f}, {best['MVA_ny']:.6f}, {best['MVA_nz']:.6f}]\n")
    f.write(f"CP  normal = [{best['CP_nx']:.6f}, {best['CP_ny']:.6f}, {best['CP_nz']:.6f}]\n")
    f.write(f"Angle(MVA, CP) = {best['angle_MVA_CP_deg']:.3f} deg\n\n")

    f.write("Plasma moments:\n")
    f.write(f"Upstream density n1 = {best['n1_upstream']:.4f} cm^-3\n")
    f.write(f"Downstream density n2 = {best['n2_downstream']:.4f} cm^-3\n")
    f.write(f"Upstream velocity V1 = [{best['V1x']:.3f}, {best['V1y']:.3f}, {best['V1z']:.3f}] km/s\n")
    f.write(f"Downstream velocity V2 = [{best['V2x']:.3f}, {best['V2y']:.3f}, {best['V2z']:.3f}] km/s\n\n")

    f.write("Shock speed (Rankine-Hugoniot mass continuity):\n")
    f.write(f"Vsh (along normal) = {best['Vsh_kms']:.3f} km/s\n")
    f.write(f"Shock velocity vector = [{best['Vsh_vec_x']:.3f}, {best['Vsh_vec_y']:.3f}, {best['Vsh_vec_z']:.3f}] km/s\n\n")

    f.write("Magnetic field means (GSM):\n")
    f.write(f"Bu = [{best['Bu_x']:.6f}, {best['Bu_y']:.6f}, {best['Bu_z']:.6f}]\n")
    f.write(f"Bd = [{best['Bd_x']:.6f}, {best['Bd_y']:.6f}, {best['Bd_z']:.6f}]\n\n")

    f.write(f"Score = {best['score']:.6f}\n")

print(f"Saved best-result report to: {OUTPUT_REPORT}")

# ==========================================================
# PRINT BEST RESULT
# ==========================================================
print("\n================ BEST RESULT ================")
print(best[[
    "t_cross", "shift_sec", "upstream_sec", "downstream_sec",
    "lambda2_lambda3", "angle_MVA_CP_deg",
    "Vsh_kms", "Vsh_vec_x", "Vsh_vec_y", "Vsh_vec_z"
]])
print("=============================================")
