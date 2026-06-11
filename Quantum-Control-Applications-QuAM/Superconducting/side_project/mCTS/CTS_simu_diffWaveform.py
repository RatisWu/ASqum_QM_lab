# %%
import numpy as np
import matplotlib.pyplot as plt

# 設定與論文、報告同等級的專業繪圖樣式
plt.rcParams.update({
    'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 2.5,
    'xtick.major.width': 2, 'ytick.major.width': 2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 8, 'ytick.major.size': 8,
})

# ==========================================
# 1. 基礎參數設定 (Basics & Specs)
# ==========================================
fs = 1                  # 取樣率 2 GHz (每 0.5 ns 一個點)
t_duration = 20         # 邏輯閘時間 Gate time (ns)       
zero_padding = 300      # 前後零填充長度 (ns)
gaussian_sigma_factor = 4
drag_coef = 0.5
amplitude = 1

# 計算時間軸與取樣點數
t_core = np.arange(0, t_duration, 1/fs) 
N_core = len(t_core)
pad_samples = int(zero_padding * fs)

# 【核心精簡】自動合併 DRAG 修正並一鍵完成前後零填充的輔助函式
def finalize_pulse(i_channel, q_der_channel):
    complex_pulse = i_channel + 1j * drag_coef * q_der_channel
    return np.pad(complex_pulse, pad_samples, mode='constant')

# ==========================================
# 2. 各式波形產生器 (Waveform Generation)
# ==========================================
pulses = {}

# A: Sine / Cosine Wave
i_cos = amplitude * 0.5 * (1 - np.cos(2 * np.pi * t_core / t_duration))
q_cos = amplitude * 0.5 * (2 * np.pi / t_duration) * np.sin(2 * np.pi * t_core / t_duration)
pulses["Cosine"] = finalize_pulse(i_cos, q_cos)

# B: Gaussian Wave (含截斷修正)
sigma = N_core / gaussian_sigma_factor
amp_factor = 1 / (1 - np.exp(-((t_duration/2) ** 2) / (2 * sigma**2)))
i_gauss = amp_factor * amplitude * (np.exp(-((t_core - (t_duration/2)) ** 2) / (2 * sigma**2)) - np.exp(-((t_duration/2) ** 2) / (2 * sigma**2)))
q_gauss = amp_factor * amplitude * (-(t_core - t_duration/2) / (sigma**2)) * np.exp(-((t_core - t_duration/2) ** 2) / (2 * sigma**2))
pulses["Gaussian"] = finalize_pulse(i_gauss, q_gauss)

# C: 4-Term Blackman-Harris
a = [0.35875, 0.48829, 0.14128, 0.01168]
i_bh = amplitude * (a[0] - a[1]*np.cos(2*np.pi*t_core/t_duration) + a[2]*np.cos(4*np.pi*t_core/t_duration) - a[3]*np.cos(6*np.pi*t_core/t_duration))
q_bh = amplitude * ((2*np.pi/t_duration) * (a[1]*np.sin(2*np.pi*t_core/t_duration) - 2*a[2]*np.sin(4*np.pi*t_core/t_duration) + 3*a[3]*np.sin(6*np.pi*t_core/t_duration)))
pulses["Blackman-Harris"] = finalize_pulse(i_bh, q_bh)

# D: Cosine Flattop (精簡平滑版本)
flat_ratio = 0.5
flat_len = int(N_core * flat_ratio)
edge_len = (N_core - flat_len) // 2
fall_len = N_core - flat_len - edge_len
i_flat = np.concatenate([
    0.5 * (1 - np.cos(np.pi * np.arange(edge_len) / edge_len)),
    np.ones(flat_len),
    0.5 * (1 + np.cos(np.pi * np.arange(fall_len) / fall_len))
]) * amplitude
pulses["Cosine Flattop"] = finalize_pulse(i_flat, np.zeros_like(i_flat))

# E: Square Wave
pulses["Square"] = finalize_pulse(np.ones(N_core) * amplitude, np.zeros(N_core))

# ==========================================
# 3. 畫圖與傅立葉變換 (Plotting & FFT)
# ==========================================
total_samples = N_core + 2 * pad_samples
t_total = np.arange(0, total_samples) / fs
freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1/fs)) * 1000  # 轉換為 MHz

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

for title, pulse in pulses.items():
    # --- 時域繪圖 (Time Domain) ---
    axes[0].plot(t_total, np.real(pulse), label=title) # 實部即為 I channel
    
    # --- 頻域繪圖 (Frequency Domain) ---
    fft_y = np.fft.fftshift(np.fft.fft(pulse))
    amplitudes = np.abs(fft_y) / total_samples
    # 保持你原本的正規化縮放邏輯
    amplitudes_norm = (amplitudes - np.min(amplitudes)) / (np.max(amplitudes) - np.min(amplitudes) + 1e-12)
    db_vals = 20 * np.log10(np.maximum(amplitudes_norm, 1e-12))
    
    axes[1].plot(freqs, db_vals, label=title)

# --- 統一格式化時域圖表 (移出迴圈，大幅提升效率) ---
axes[0].set_title("Time Domain")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Amplitude [a.u.]")
axes[0].set_xlim(zero_padding - 10, zero_padding + t_duration + 10)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --- 統一格式化頻域圖表 ---
axes[1].set_title("Frequency Domain")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Amplitude [dB]")
axes[1].set_ylim(-100, 10)
axes[1].set_xlim(-300, 300)
axes[1].axvline(-180, color='gray', linestyle='--', alpha=0.7)  # 串擾抑制參考線
axes[1].axvline(180, color='gray', linestyle='--', alpha=0.7)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.suptitle(f"Waveform and Spectral Analysis, Gate time {int(t_duration)} ns")
plt.tight_layout()
plt.show()
# %%