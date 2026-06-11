import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
# %%
# ==================== 1. 參數與目標頻率設定 ====================
tau_g = 20         # 限制 20 ns
fs = 1.0           # 取樣率 1 GHz
t = np.arange(0, tau_g + 0.5/fs, 1/fs)
N = len(t)
sup_power = -80    # 目標壓制 -80 dB

# target_holes = np.array([75])
# target_holes = np.array([-180, -190, -170, -150, -120, -100, 50, 150, 200])
# target_holes = np.array([-180, 60, -120]) # IQM

## Crowded holes may cause issues
# target_holes = np.array([-180, -190, -170, -150, -120, -100, 50, 45, -35])
# target_holes = np.array([-90, 104, 97, -132, -116, -100, 122, 147, -163])

# 10Qv2 case
target_holes = np.array([60,30,230,450,70,-120, -150, 50, 230, -110])

# 理論 M 值推導 (M = 2 * tau_g * f_max)
MM = math.ceil(2 * tau_g * max(np.abs(target_holes))/1000)

if MM < 5:
    Ms = np.arange(1, MM+5, 1).tolist()
else:
    Ms = np.arange(MM-5, 2*MM, 1).tolist() 

# ==================== 2. 定義優化專用（無正規化）的波形重構 ====================
def reconstruct_raw(coeffs, M_val):
    a = coeffs[:M_val]
    b = coeffs[M_val:]
    I_ch = np.zeros(N)
    Q_ch = np.zeros(N)
    for k in range(1, M_val + 1):
        # 純 Sine 基底確保頭尾完美歸零
        I_ch += a[k-1] * np.sin(k * np.pi * t / tau_g)
        Q_ch += b[k-1] * np.sin(k * np.pi * t / tau_g)
    return I_ch, Q_ch

# ==================== 3. 執行 M 值掃描與強力優化 ====================
best_M = None
best_res = None
best_penalty = float('inf')

for M in Ms:
    def basis_objective_log(coeffs):
        I_ch, Q_ch = reconstruct_raw(coeffs, M)
        complex_signal = I_ch + 1j * Q_ch
        
        N_padded = N * 50
        fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
        freqs = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(fs * 1000)))
        
        # 頻譜內部正規化
        psd_norm = np.abs(fft_y)**2 / np.max(np.abs(fft_y)**2)
        
        penalty_holes = 0.0
        for hole_f in target_holes:
            idx = np.argmin(np.abs(freqs - hole_f))
            db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
            if db_val > sup_power:
                penalty_holes += (db_val - sup_power)**2 

        overflow_I = np.maximum(0, np.abs(I_ch) - 1.0)**2
        overflow_Q = np.maximum(0, np.abs(Q_ch) - 1.0)**2
        penalty_amplitude = 10 * np.log10(np.sum(overflow_I) + np.sum(overflow_Q) + 1e-12)
                
        return penalty_holes + penalty_amplitude

    # 初始化係數
    initial_coeffs = np.zeros(2 * M)
    initial_coeffs[0] = 1.0  # I Channel 主頻成分
    initial_coeffs[M] = 0.2  # Q Channel 初始微調

    res = minimize(basis_objective_log, initial_coeffs, method='L-BFGS-B', options={'maxiter': 2000})
    
    print(f"測試 M={M}, 最終懲罰值: {res.fun:.4f}")
    
    # 紀錄最佳結果
    if res.fun < best_penalty:
        best_penalty = res.fun
        best_res = res
        best_M = M

print(f"\n🚀 最佳化完成！選擇了最適合的基底數量 M = {best_M}")

# ==================== 4. 最終波形正規化與畫圖驗證 ====================
opt_I_raw, opt_Q_raw = reconstruct_raw(best_res.x, best_M)

# 只有在畫圖前，才進行一次性的安全正規化
max_amp = np.max(np.abs(opt_I_raw))
opt_I = opt_I_raw / max_amp
opt_Q = opt_Q_raw / max_amp

plt.figure(figsize=(12, 4))

# 時域圖
plt.subplot(1, 2, 1)
plt.plot(t, opt_I, 'b.-', label='I Channel')
plt.plot(t, opt_Q, 'r.-', label='Q Channel')
plt.title('Optimized Time Domain (Normalized)')
plt.ylabel('Amplitude [a.u.]')
plt.xlabel('Time (ns)')
plt.grid(True, alpha=0.3)
plt.legend()

# 頻域圖
plt.subplot(1, 2, 2)
N_p = N * 30
freqs = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(fs * 1000)))
fft_complex = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=N_p))
db_complex = 10 * np.log10(np.abs(fft_complex)**2 / np.max(np.abs(fft_complex)**2))

plt.plot(freqs, db_complex, 'purple', label='Optimized Spectrum')
for i, hole_f in enumerate(target_holes):
    plt.axvline(hole_f, color='red', linestyle='--', alpha=0.5, label='Target Holes' if i == 0 else "")
plt.axhline(sup_power, color='black', linestyle=':', label=f'{sup_power} dB Line')

plt.ylim(-100, 5)
plt.ylabel('Amplitude [dB]')
plt.xlim(-300, 300)
plt.legend()
plt.title('Optimized Frequency Domain')
plt.grid(True, alpha=0.3)

plt.suptitle(f"L-BFGS-B Optimization Result\nTarget Avoid Frequencies: {', '.join(map(str, target_holes))} MHz\nGate Time: {int(tau_g)} ns | Optimal Sine Basis: M={best_M}")
plt.tight_layout()
plt.show()

# %%