import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
# %%
# ==================== 1. 參數與 9 個洞的頻率設定 ====================
tau_g = 20         # 嚴格限制 20 ns
fs = 1.0           # 取樣率 1 GHz (每 1 ns 一個點)
t = np.arange(0, tau_g + 0.5/fs, 1/fs)
N = len(t)         # 時域點數：21 個點
sup_power = -45    # 目標壓制 -80 dB

# 9 個目標頻率點 (MHz)
# target_holes = np.array([75])
# target_holes = np.array([-180, -190, -170, -150, -120, -100, 50, 150, 200])
# target_holes = np.array([-180, 60, -120]) # IQM
# target_holes = np.array([-180, 60, -120, -30])
# target_holes = np.array([-180, 60, -120, -30, 110])
## Crowded holes may cause issues
# target_holes = np.array([-180, -190, -170, -150, -120, -100, 50, 45, -35])
target_holes = np.array([-90, 104, 97, -132, -116, -100, 122, 147, -153])

# 10Qv2 case
# target_holes = np.array([60,30,230,450,70,-120, -150, 50, 230, -110])

# ==================== 2. 設計虛擬實體低通濾波器 ====================
# 因為目標破洞最高到 200 MHz，我們將濾波器截止頻率設在 250 MHz
# 這能確保 200 MHz 以內的控制訊號能通過，同時抹除更高頻的刺蝟雜訊
cutoff_mhz = np.max(np.abs(target_holes))*1.25
nyquist_mhz = (fs * 1000) / 2  # 奈奎斯特頻率 = 500 MHz
Wn = cutoff_mhz / nyquist_mhz  # 正規化截止頻率

# 建立 4 階低通巴特沃斯濾波器係數
b_filter, a_filter = butter(4, Wn, btype='low')

# ==================== 3. 定義 GRAPE 波形重構與濾波函式 ====================
def reconstruct_and_filter(coeffs):
    # coeffs 是優化器試圖控制的「原始不平滑振幅」
    I_raw = coeffs[:N].copy()
    Q_raw = coeffs[N:].copy()
    
    # 使用 filtfilt 進行雙向濾波，確保「零相位延遲」（波形不會在時間軸上向後位移）
    # 這是逼迫演算法保持平滑的核心步驟
    I_smooth = filtfilt(b_filter, a_filter, I_raw)
    Q_smooth = filtfilt(b_filter, a_filter, Q_raw)
    
    # 強制頭尾歸零 (邊界條件)
    I_smooth[0] = I_smooth[-1] = 0.0
    Q_smooth[0] = Q_smooth[-1] = 0.0
    
    return I_raw, Q_raw, I_smooth, Q_smooth

# ==================== 4. 濾波型 GRAPE 目標函數 ====================
def grape_filtered_objective(coeffs):
    # 拿到燙平後的平滑波形
    _, _, I_smooth, Q_smooth = reconstruct_and_filter(coeffs)
    complex_signal = I_smooth + 1j * Q_smooth
    
    # 高倍率補零，精準對齊頻率格點
    N_padded = N * 50
    fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(fs * 1000)))
    
    # 頻譜內部正規化
    psd_norm = np.abs(fft_y)**2 / (np.max(np.abs(fft_y)**2) + 1e-12)
    
    # 計算 9 個洞的懲罰值
    penalty_holes = 0.0
    for hole_f in target_holes:
        idx = np.argmin(np.abs(freqs - hole_f))
        db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
        if db_val > sup_power:
            penalty_holes += (db_val - sup_power)**2 
            
    # 注意：這裡完全不需要加任何 smoothness penalty！
    return penalty_holes 

# ==================== 5. 執行 L-BFGS-B 強力優化 ====================
# 初始化：給予一個基本的高斯或正弦波形作為優化起點
initial_coeffs = np.zeros(2 * N)
initial_coeffs[:N] = np.sin(np.pi * t / tau_g)
initial_coeffs[N:] = 0.1 * np.sin(np.pi * t / tau_g)
box_bounds = [(-1.0, 1.0)] * (2 * N)

print("🚀 啟動濾波型 GRAPE 演算法...")
print(f"時域自由度: {2*N} 點 | 濾波截止頻率: {cutoff_mhz} MHz")

res = minimize(grape_filtered_objective, initial_coeffs, method='L-BFGS-B', bounds=box_bounds, options={'maxiter': 60000})

print("-" * 50)
print(f"優化完成！最終頻譜懲罰值: {res.fun:.4f}")
if res.fun == 0:
    print("✨ 滿分！9 個目標頻率點已全數成功壓制到 -80 dB 以下！")
print("-" * 50)

# ==================== 6. 最終波形提取與畫圖驗證 ====================
opt_I_raw, opt_Q_raw, opt_I_plot, opt_Q_plot = reconstruct_and_filter(res.x)


plt.figure(figsize=(14, 5))

# --- 左圖：時域波形對比 ---
plt.subplot(1, 2, 1)
# 畫出最終要送進 Qubit 的平滑波形 (實線)
plt.plot(t, opt_I_plot, 'b-', linewidth=2.5, label='Filtered I')
plt.plot(t, opt_Q_plot, 'r-', linewidth=2.5, label='Filtered Q')
# 隱約畫出優化器在背後操控的原始離散點 (虛線/點)，觀察濾波器如何馴服它
plt.plot(t, opt_I_raw, 'b.-', alpha=0.2, linestyle='--', label='GRAPE I')
plt.plot(t, opt_Q_raw, 'r.-', alpha=0.2, linestyle='--', label='GRAPE Q')

plt.title('Time Domain')
plt.ylabel('Amplitude [a.u.]')
plt.xlabel('Time (ns)')
plt.grid(True, alpha=0.3)
plt.legend()

# --- 右圖：頻域能量譜 ---
plt.subplot(1, 2, 2)
N_p = N * 50
freqs_p = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(fs * 1000)))
fft_complex = np.fft.fftshift(np.fft.fft(opt_I_plot + 1j * opt_Q_plot, n=N_p))
db_complex = 10 * np.log10(np.abs(fft_complex)**2 / np.max(np.abs(fft_complex)**2))

plt.plot(freqs_p, db_complex, 'purple', linewidth=2, label='Filtered Spectrum')
for i, hole_f in enumerate(target_holes):
    plt.axvline(hole_f, color='red', linestyle='--', alpha=0.6, label='Target Holes' if i == 0 else "")
plt.axhline(sup_power, color='black', linestyle=':', linewidth=1.5, label=f'{sup_power} dB Line')

plt.ylim(-100, 5)
plt.xlim(-300, 300)
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency (MHz)')
plt.title('Frequency Domain')
plt.grid(True, alpha=0.3)
plt.legend()

plt.suptitle(f"Hardware-In-The-Loop GRAPE Optimization\nTarget avoid frequencies: {', '.join(map(str, target_holes))} MHz \nGate Time: {tau_g} ns", y=0.98)
plt.tight_layout()
plt.show()
# %%



