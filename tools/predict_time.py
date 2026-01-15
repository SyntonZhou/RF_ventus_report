import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

# ================== 输出精度：拉满 ==================
np.set_printoptions(precision=30, suppress=False, linewidth=200)
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 30)

# decimal 展示用（后处理显示更长，不改变 float64 拟合本质）
getcontext().prec = 80

def D(x):
    # Decimal(str(x)) 比 Decimal(x) 更能保留 float 的十进制表示
    return Decimal(str(x))

# ====== 输入：你的原始数据（方阵：M=N=K=S）======
S = np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=np.float64)

t_old = np.array([
    0.000029605,
    0.000038695,
    0.000087105,
    0.000350115,
    0.002085525,
    0.015549915,
    0.112245315,
    0.866288685
], dtype=np.float64)

t_new = np.array([
    0.000042925,
    0.000080885,
    0.000250695,
    0.000418775,
    0.001168265,
    0.007189475,
    0.050004065,
    0.378897815
], dtype=np.float64)

# 方阵 GEMM 的 FLOPs
F = 2.0 * (S ** 3)

P_old = F / t_old
P_new = F / t_new

def fit_t_vs_s3(S, t, fit_min=32):
    """
    拟合模型：t(S) ≈ t0 + alpha * S^3
    返回：alpha, t0, P_peak(=2/alpha), rmse, used_points_count
    """
    mask = S >= fit_min
    x = (S[mask] ** 3)

    A = np.vstack([x, np.ones_like(x)]).T
    alpha, t0 = np.linalg.lstsq(A, t[mask], rcond=None)[0]

    P_peak = 2.0 / alpha  # FLOP/s
    t_hat = t0 + alpha * x
    rmse = np.sqrt(np.mean((t[mask] - t_hat) ** 2))

    return alpha, t0, P_peak, rmse, mask.sum()

def predict(alpha, t0, S_list):
    S_list = np.array(S_list, dtype=np.float64)
    F_list = 2.0 * (S_list ** 3)
    t_list = t0 + alpha * (S_list ** 3)
    P_list = F_list / t_list
    return F_list, t_list, P_list

def print_fit_and_check(tag, alpha, t0, rmse, fit_points, S_check_list):
    """
    打印：
    1) 拟合公式 t = t0 + alpha*S^3
    2) 性能公式 P(S) = 2S^3 / (t0 + alpha*S^3)
    3) t/(2S^3) 的值（按拟合时间）
    4) n->inf 的极限：P_peak=2/alpha, 以及 t/(2S^3)->alpha/2
    """
    print(f"\n================== [{tag}] ==================")
    print("Fit model: t(S) = t0 + alpha * S^3")
    print(f"fit_points = {fit_points}")
    print(f"t0    = {t0}")
    print(f"alpha = {alpha}")
    print(f"rmse  = {rmse}")

    print("\nDerived performance model:")
    print("P(S) = 2*S^3 / (t0 + alpha*S^3)")

    P_peak = 2.0 / alpha
    print("\nLimits as S->infty:")
    print(f"P_peak = lim P(S) = 2/alpha = {P_peak} FLOP/s  (= {P_peak/1e6} MFLOP/s)")
    print(f"lim t(S)/(2*S^3) = alpha/2 = {alpha/2.0} s/FLOP")

    # 用 Decimal 再展示一次（只是展示更长）
    print("\nHigh-precision display (Decimal, based on float64 values):")
    print(f"t0    = {D(t0)}")
    print(f"alpha = {D(alpha)}")
    print(f"P_peak(=2/alpha) = {D(2.0) / D(alpha)}")
    print(f"alpha/2          = {D(alpha) / D(2.0)}")

    # 在若干 S 点上检查：t_fit/(2S^3) 与 P_fit
    S_check = np.array(S_check_list, dtype=np.float64)
    t_fit = t0 + alpha * (S_check ** 3)
    ratio = t_fit / (2.0 * (S_check ** 3))     # 你要的：时间/(2n^3)
    P_fit = (2.0 * (S_check ** 3)) / t_fit     # 对应性能

    df_chk = pd.DataFrame({
        "S": S_check.astype(np.int64),
        "t_fit(s)": t_fit,
        "t_fit/(2*S^3) (s/FLOP)": ratio,
        "P_fit (FLOP/s)": P_fit,
        "P_fit (MFLOP/s)": P_fit / 1e6
    })

    print("\nCheck points (using fitted t):")
    print(df_chk.to_string(index=False))

# ====== 拟合（默认用 S>=32）======
alpha_old, t0_old, Ppk_old, rmse_old, n_old = fit_t_vs_s3(S, t_old, fit_min=32)
alpha_new, t0_new, Ppk_new, rmse_new, n_new = fit_t_vs_s3(S, t_new, fit_min=32)

print("=== Fit results (t ≈ t0 + alpha*S^3) ===")
print(f"[old] t0={t0_old} alpha={alpha_old} P_peak={Ppk_old} rmse={rmse_old} (MFLOP/s={Ppk_old/1e6})")
print(f"[new] t0={t0_new} alpha={alpha_new} P_peak={Ppk_new} rmse={rmse_new} (MFLOP/s={Ppk_new/1e6})")

# 512 点“贴近峰值”的百分比（实测）
idx_512 = np.where(S == 512)[0][0]
print("\n=== Peak proximity at S=512 (measured) ===")
print(f"[old] P_meas(512)={P_old[idx_512]} FLOP/s, ratio={P_old[idx_512]/Ppk_old}")
print(f"[new] P_meas(512)={P_new[idx_512]} FLOP/s, ratio={P_new[idx_512]/Ppk_new}")
print(f"speedup_peak = {Ppk_new/Ppk_old}")

# 额外：把拟合后的性能公式/ratio 打印出来给你验算
print_fit_and_check("old", alpha_old, t0_old, rmse_old, n_old, S_check_list=[32, 64, 128, 256, 512, 1024, 2048, 4096])
print_fit_and_check("new", alpha_new, t0_new, rmse_new, n_new, S_check_list=[32, 64, 128, 256, 512, 1024, 2048, 4096])

# ====== 外推预测 ======
S_pred = [1024, 2048, 4096]
F_old_pred, t_old_pred, P_old_pred = predict(alpha_old, t0_old, S_pred)
F_new_pred, t_new_pred, P_new_pred = predict(alpha_new, t0_new, S_pred)

df = pd.DataFrame({
    "S(M=N=K)": np.array(S_pred, dtype=np.int64),
    "FLOPs(2*S^3)": F_old_pred,  # float64，但整数可读
    "pred_time_old(s)": t_old_pred,
    "pred_perf_old(FLOP/s)": P_old_pred,
    "pred_perf_old(MFLOP/s)": P_old_pred/1e6,
    "pred_time_new(s)": t_new_pred,
    "pred_perf_new(FLOP/s)": P_new_pred,
    "pred_perf_new(MFLOP/s)": P_new_pred/1e6,
    "speedup_time(old/new)": t_old_pred / t_new_pred,
    # 你要的 ratio：t/(2S^3)，这里也给外推点
    "pred_old_t/(2*S^3)": t_old_pred / (2.0*(np.array(S_pred, dtype=np.float64)**3)),
    "pred_new_t/(2*S^3)": t_new_pred / (2.0*(np.array(S_pred, dtype=np.float64)**3)),
})

print("\n=== Extrapolation (no rounding) ===")
print(df.to_string(index=False))

df.to_csv("gemm_peak_fit_predict.csv", index=False)
print("\n[OK] saved -> gemm_peak_fit_predict.csv")
