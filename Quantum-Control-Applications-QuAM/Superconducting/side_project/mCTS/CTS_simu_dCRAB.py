import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# %%
# ==================== 1. 基本參數與目標設定 ====================
tau_g = 16      # 嚴格限制 16 ns
fs = 1.0           # 取樣率 1 GHz
t = np.arange(0, tau_g + 0.5/fs, 1/fs)
N = len(t)
sup_power = -60  # target crosstalk level
# 9 個需要挖洞的目標頻率 (MHz)
# target_holes = np.array([-210, -190, -170, -150, -120, -100, 50, 150, 200]) 
target_holes = np.array([60,30,230,450,70,-120, -150, 50, 230, -110])
# 天然邊界包絡線：確保隨機頻率不管怎麼加，兩端一定完美歸零
envelope = np.sin(np.pi * t / tau_g)

# 用來記錄 dCRAB 逐步累積的隨機頻率陣列
accumulated_freqs = []

# ==================== 2. 定義從頻率與係數重構波形的函數 ====================
def reconstruct_dcrab(coeffs, freqs_list):
    n_freqs = len(freqs_list)
    if n_freqs == 0:
        return np.zeros(N), np.zeros(N)
        
    # 前 n_freqs 個係數是 I 的振幅，後 n_freqs 個是 Q 的振幅
    a = coeffs[:n_freqs]
    b = coeffs[n_freqs:]
    
    I_core = np.zeros(N)
    Q_core = np.zeros(N)
    
    # 疊加隨機頻率成份 (使用 GHz 單位進行時間乘法)
    for i, f_mhz in enumerate(freqs_list):
        f_ghz = f_mhz / 1000.0
        I_core += a[i] * np.sin(2 * np.pi * f_ghz * t)
        Q_core += b[i] * np.cos(2 * np.pi * f_ghz * t)
        
    # 乘以包絡線強迫邊界歸零，並加上一個直流基本項確保主瓣存在
    I_ch = envelope * (1.0 + I_core)
    Q_ch = envelope * Q_core
    
    # 正規化 I Channel 最大值為 1.0
    max_I = np.max(np.abs(I_ch))
    if max_I > 0:
        I_ch /= max_I
        Q_ch /= max_I
        
    return I_ch, Q_ch

# ==================== 3. dCRAB 目標成本函數 ====================
def dcrab_objective(coeffs):
    I_ch, Q_ch = reconstruct_dcrab(coeffs, accumulated_freqs)
    complex_signal = I_ch + 1j * Q_ch
    
    # 高解析度 FFT
    N_padded = N * 50
    fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
    freqs_fft = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(fs * 1000)))
    
    psd_norm = np.abs(fft_y)**2 / np.max(np.abs(fft_y)**2)
    
    worst_db = -100.0
    penalty = 0.0
    
    # 檢查 9 個目標點
    for hole_f in target_holes:
        idx = np.argmin(np.abs(freqs_fft - hole_f))
        db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
        worst_db = max(worst_db, db_val)
        
        # 只要沒有低於 -60 dB，就給予平方懲罰
        if db_val > sup_power:
            penalty += (db_val - (sup_power))**2

    overflow_I = np.maximum(0, np.abs(I_ch) - 1.0)**2
    overflow_Q = np.maximum(0, np.abs(Q_ch) - 1.0)**2
    penalty_amplitude = 10 * np.log10(np.sum(overflow_I) + np.sum(overflow_Q) + 1e-12)
            
    return penalty + penalty_amplitude

# ==================== 4. dCRAB 核心主適應性迴圈 ====================
print("🚀 開始執行 dCRAB 適應性隨機基底優化...")
max_macro_iters = 10  # 最多進行 6 輪隨機頻率追加
success = False

np.random.seed(42) # 固定隨機種子確保可重複性

for macro_it in range(1, max_macro_iters + 1):
    # 每一輪隨機挑選 2 個新頻率（範圍在 30 ~ 250 MHz 之間，避開直流）
    new_freqs = np.random.uniform(30, 250, size=2)
    accumulated_freqs.extend(new_freqs)
    
    n_current = len(accumulated_freqs)
    print(f"\n[第 {macro_it} 輪] 目前已累積 {n_current} 個隨機基底頻率...")
    print(f"-> 新增隨機頻率: {new_freqs[0]:.1f} MHz, {new_freqs[1]:.1f} MHz")
    
    # 初始化目前所有頻率的係數 (2 * n_current 個變數)
    initial_coeffs = np.zeros(2 * n_current)
    
    # 執行優化
    res = minimize(dcrab_objective, initial_coeffs, method='L-BFGS-B', options={'maxiter': 1000})
    
    # 驗證這一輪優化完後，最爛的洞表現如何
    opt_I, opt_Q = reconstruct_dcrab(res.x, accumulated_freqs)
    fft_res = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=N*50))
    freqs_fft = np.fft.fftshift(np.fft.fftfreq(N*50, 1/(fs * 1000)))
    psd_res = np.abs(fft_res)**2 / np.max(np.abs(fft_res)**2)
    
    all_holes_db = [10 * np.log10(psd_res[np.argmin(np.abs(freqs_fft - h))] + 1e-12) for h in target_holes]
    worst_hole_current = max(all_holes_db)
    
    print(f"-> 最佳化完成！當前 9 個目標洞中，最差的殘留能量為: {worst_hole_current:.2f} dB")
    
    # 如果最差的洞都順利跌破 -60 dB，大功告成，提前提早結束！
    if worst_hole_current <= sup_power:
        print(f"🎉 成功達標！所有目標點皆低於 {sup_power} dB！")
        success = True
        break

# ==================== 5. 繪製最終 dCRAB 成果圖 ====================
plt.figure(figsize=(12, 4))

# 左圖：時間域（自帶包絡線，絕對平滑且頭尾歸零）
plt.subplot(1, 2, 1)
plt.plot(t, opt_I, 'b.-', linewidth=2, label='I Channel')
plt.plot(t, opt_Q, 'r.-', linewidth=2, label='Q Channel')
plt.title(f'Time Domain')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.legend()

# 右圖：頻率域（看隨機組合出來的非對稱挖洞神蹟）
plt.subplot(1, 2, 2)
N_p = N * 50
freqs_plot = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(fs * 1000)))
fft_final = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=N_p))
db_final = 10 * np.log10(np.abs(fft_final)**2 / np.max(np.abs(fft_final)**2))

plt.plot(freqs_plot, db_final, 'purple', linewidth=1.5, label='dCRAB Spectrum')
for i, hole_f in enumerate(target_holes):
    plt.axvline(hole_f, color='red', linestyle='--', alpha=0.5, label='9 Target Holes' if i == 0 else "")
plt.axhline(sup_power, color='black', linestyle=':', label=f'{sup_power} dB Line')

plt.ylim(-85, 5)
plt.xlim(-300, 300)
plt.title(f'Frequency Domain (All Holes < {sup_power}dB)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle(f"dCRAB Optimization Result\nTarget avoid frequencies: {', '.join(map(str, target_holes))} MHz \nGate Time: {tau_g} ns | Total Random Frequencies: {len(accumulated_freqs)}\n")
plt.tight_layout()
plt.show()

# %%