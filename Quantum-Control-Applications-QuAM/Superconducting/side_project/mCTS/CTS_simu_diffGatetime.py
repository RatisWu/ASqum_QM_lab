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
t_duration_list = [16, 32]        # 不同的邏輯閘時間 (ns)
zero_padding_margin = 300         # 核心脈衝前後的最小留白 (ns)
amplitude = 1

# 預先計算一個足夠長的統一總時間視窗，用來承載所有脈衝
max_tg = max(t_duration_list)
total_view_duration = max_tg + 2 * zero_padding_margin
N_total = int(total_view_duration * fs)

# 建立畫布
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

# ==========================================
# 2. 精簡與中間對齊優化迴圈
# ==========================================
for tg in t_duration_list:
    # --- A. 產生核心時域波形 (Cosine Pulse) ---
    t_core = np.arange(0, tg, 1/fs) 
    N_core = len(t_core)
    print(N_core)
    # [修改處] 計算 Cosine 主波形與其時域導數
    wave_cos_core = amplitude * np.cos(np.pi * (t_core - tg/2) / tg)
    wave_cos_der_core = -amplitude * (np.pi / tg) * np.sin(np.pi * (t_core - tg/2) / tg)
    
    # 結合為 DRAG 複數訊號
    DRAG_cos_core = wave_cos_core + 1j * DRAG_coef * wave_cos_der_core
    
    # --- B. 計算非對稱零填充以實現中間對齊 ---
    total_pad_samples = N_total - N_core
    pad_before = total_pad_samples // 2
    pad_after = total_pad_samples - pad_before
    
    # 使用非對稱填充
    DRAG_cos_padded = np.pad(DRAG_cos_core, (pad_before, pad_after), mode='constant')
    
    # 計算時間軸
    t_absolute = np.arange(0, N_total, 1/fs)
    
    # --- C. 時域繪圖 (將時間軸轉換為「相對中心點」) ---
    reference_center_time = total_view_duration / 2
    t_relative = t_absolute - reference_center_time
    
    # 實部（I Channel）形狀
    axes[0].plot(t_relative, np.real(DRAG_cos_padded), label=f"{tg} ns", marker='o')
    
    # --- D. 頻域 FFT 計算與正規化 ---
    fft_y_shifted = np.fft.fftshift(np.fft.fft(DRAG_cos_padded))
    freqs_p = np.fft.fftshift(np.fft.fftfreq(N_total, 1/fs)) * 1000  # 轉換為 MHz
    
    amplitudes_p = np.abs(fft_y_shifted) / N_total
    amplitudes_norm_p = (amplitudes_p - np.min(amplitudes_p)) / (np.max(amplitudes_p) - np.min(amplitudes_p) + 1e-12)
    db_vals_p = 20 * np.log10(np.maximum(amplitudes_norm_p, 1e-12))
    
    # --- E. 頻域繪圖 ---
    axes[1].plot(freqs_p, db_vals_p, label=f"{tg} ns")

# ==========================================
# 3. 統一格式化圖表
# ==========================================
# --- 時域圖表美化 ---
axes[0].set_title("Time Domain (Cosine Pulse - Center Aligned)")
axes[0].set_xlabel("Time relative to pulse center (ns)")
axes[0].set_ylabel("Amplitude [a.u.]")
observation_width = max_tg / 2 + 10
axes[0].set_xlim(-observation_width, observation_width) 
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --- 頻域圖表美化 ---
axes[1].set_title("Frequency Domain")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Amplitude [dB]")
axes[1].set_ylim(-100, 10)
axes[1].set_xlim(-300, 300)
axes[1].axvline(-180, color='gray', linestyle='--', alpha=0.7)  # Ref line
axes[1].axvline(180, color='gray', linestyle='--', alpha=0.7)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.suptitle("Cosine Pulse Center Aligned Sidelobe Comparison")
plt.tight_layout()
plt.show()
# %%