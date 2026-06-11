# %%
import numpy as np
import matplotlib.pyplot as plt

# 專業繪圖參數設定
plt.rcParams.update({
    'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 2.5,
    'xtick.major.width': 2, 'ytick.major.width': 2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 8, 'ytick.major.size': 8,
})

# ==========================================
# 1. 基礎參數設定 (Basics)
# ==========================================
DRAG_coef = 0.5
fs = 1.0                          # 取樣率 1 GHz (每 1 ns 一個點)
t_duration_list = [16, 20, 24, 40, 64, 128] # 不同的邏輯閘時間 (ns)
zero_padding_margin = 300       # 核心脈衝前後的最小留白 (ns)
amplitude = 1

# [核心新增] 預先計算一個足夠長的統一總時間視窗，用來承載所有脈衝
# 視窗長度 = 最大脈衝時間 + 兩倍的留白邊距
max_tg = max(t_duration_list)
total_view_duration = max_tg + 2 * zero_padding_margin
N_total = int(total_view_duration * fs)

# 4-Term Blackman-Harris 係數
a_coeffs = [0.35875, 0.48829, 0.14128, 0.01168] 

# 建立畫布
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

# ==========================================
# 2. 精簡與中間對齊優化迴圈
# ==========================================
for tg in t_duration_list:
    # --- A. 產生核心時域波形 ---
    t_core = np.arange(0, tg, 1/fs) 
    N_core = len(t_core)
    
    # 計算 Blackman-Harris 實部與導數
    wave_BH4T_core = amplitude * (a_coeffs[0] - a_coeffs[1]*np.cos(2*np.pi*t_core/N_core) + a_coeffs[2]*np.cos(4*np.pi*t_core/N_core) - a_coeffs[3]*np.cos(6*np.pi*t_core/N_core))
    wave_BH4T_der_core = amplitude * ((2*np.pi/N_core) * (a_coeffs[1]*np.sin(2*np.pi*t_core/N_core) - 2*a_coeffs[2]*np.sin(4*np.pi*t_core/N_core) + 3*a_coeffs[3]*np.sin(6*np.pi*t_core/N_core)))
    
    # 結合為 DRAG 複數訊號
    DRAG_BH4T_core = wave_BH4T_core + 1j * DRAG_coef * wave_BH4T_der_core
    
    # --- [核心修正] B. 計算非對稱零填充以實現中間對齊 ---
    # 目標：讓脈衝中心 (tg/2) 位於 total_view_duration / 2 的位置
    # 總需要的填充點數
    total_pad_samples = N_total - N_core
    # 計算脈衝前需要的填充（自動向下取整），剩下的全補在後面
    pad_before = total_pad_samples // 2
    pad_after = total_pad_samples - pad_before
    
    # 使用非對稱填充
    DRAG_BH4T_padded = np.pad(DRAG_BH4T_core, (pad_before, pad_after), mode='constant')
    
    # 計算時間軸 (N_total 已經是統一的了)
    t_absolute = np.arange(0, N_total, 1/fs)
    
    # --- [核心修正] C. 時域繪圖 (將時間軸轉換為「相對中心點」) ---
    # 參考中心點設為 0
    reference_center_time = total_view_duration / 2
    t_relative = t_absolute - reference_center_time
    
    # 實部（I Channel）形狀
    axes[0].plot(t_relative, np.real(DRAG_BH4T_padded), label=f"{tg} ns")
    
    # --- D. 頻域 FFT 計算與正規化 ---
    fft_y_shifted = np.fft.fftshift(np.fft.fft(DRAG_BH4T_padded))
    # 由於 N_total 統一，所有脈衝的頻率解析度也統一了，這對 CTS 比較更公平
    freqs_p = np.fft.fftshift(np.fft.fftfreq(N_total, 1/fs)) * 1000  # 轉換為 MHz
    
    # 你原本的雙邊頻譜縮放與 dB 轉換邏輯
    amplitudes_p = np.abs(fft_y_shifted) / N_total
    amplitudes_norm_p = (amplitudes_p - np.min(amplitudes_p)) / (np.max(amplitudes_p) - np.min(amplitudes_p) + 1e-12)
    db_vals_p = 20 * np.log10(np.maximum(amplitudes_norm_p, 1e-12))
    
    # --- E. 頻域繪圖 ---
    axes[1].plot(freqs_p, db_vals_p, label=f"{tg} ns")

# ==========================================
# 3. 統一格式化圖表 (移出迴圈)
# ==========================================
# --- 時域圖表美化 (對齊版) ---
axes[0].set_title("Time Domain (Center Aligned)")
axes[0].set_xlabel("Time relative to pulse center (ns)") # 修改 X 軸註釋
axes[0].set_ylabel("Amplitude [a.u.]")
# 將觀測視窗聚焦在脈衝主體，X 從 -max_tg/2 到 max_tg/2
observation_width = max_tg / 2 + 10
axes[0].set_xlim(-observation_width, observation_width) 
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --- 頻域圖表美化 (保持原樣，僅微調) ---
axes[1].set_title("Frequency Domain")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Amplitude [dB]")
axes[1].set_ylim(-100, 10)
axes[1].set_xlim(-300, 300)
axes[1].axvline(-180, color='gray', linestyle='--', alpha=0.7)  # Ref line
axes[1].axvline(180, color='gray', linestyle='--', alpha=0.7)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.suptitle("Center Aligned Sidelobe Comparison")
plt.tight_layout()
plt.show()
# %%