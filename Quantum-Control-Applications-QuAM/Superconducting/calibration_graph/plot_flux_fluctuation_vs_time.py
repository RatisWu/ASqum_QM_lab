from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======= Configuration =======
excel_path = Path(r"d:\qm_code\as\qua-libs\Quantum-Control-Applications-QuAM\Superconducting\data\qubit_measurements_2025-10-21-09-40-00.xlsx")
base_dir = excel_path.parent
# =============================

# create output folder
out_dir = base_dir / excel_path.stem
out_dir.mkdir(parents=True, exist_ok=True)

# read Excel
xls = pd.ExcelFile(excel_path)
qubit_sheets = xls.sheet_names  # list of qubits

# for collecting time-series data across all qubits
func_data_dict = {}

for qname in qubit_sheets:
    df = pd.read_excel(xls, sheet_name=qname)

    # convert timestamp if needed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # identify function columns (exclude run_index, timestamp, and err columns)
    func_cols = [c for c in df.columns if c not in ("run_index", "timestamp") and not c.endswith("_err")]
    if not func_cols:
        continue

    # --- Distribution plot for each function (ignore err columns) ---
    for f in func_cols:
        data = df[f].dropna()
        if data.empty:
            continue
        fig, ax = plt.subplots()
        ax.hist(data, bins=10, alpha=0.7, color="skyblue", edgecolor="black")
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax.axvline(mean_val + std_val, color="green", linestyle=":", label=f"+1σ: {mean_val+std_val:.3f}")
        ax.axvline(mean_val - std_val, color="green", linestyle=":", label=f"-1σ: {mean_val-std_val:.3f}")
        ax.set_title(f"{qname} – {f} distribution")
        ax.set_xlabel(f)
        ax.set_ylabel("Counts")
        ax.legend()
        fig.savefig(out_dir / f"{qname}_{f}_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Collect data for time-series plot across qubits ---
    for f in func_cols:
        sub = df[["run_index", "timestamp", f]].dropna()
        if sub.empty:
            continue
        sub["qubit"] = qname
        sub["err"] = df[f"{f}_err"] if f"{f}_err" in df.columns else np.nan
        func_data_dict.setdefault(f, []).append(sub)

# --- Plot time series: all qubits together per function ---
for f, datasets in func_data_dict.items():
    fig, ax = plt.subplots()
    for sub in datasets:
        qname = sub["qubit"].iloc[0]
        yerr = sub["err"] if not sub["err"].isna().all() else None
        ax.errorbar(sub["run_index"], sub[f], yerr=yerr, marker="o", linestyle="-", label=qname)
    
    ax.set_title(f"All qubits – {f} vs run_index / timestamp")
    ax.set_xlabel("Run index")
    ax.set_ylabel(f)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.xticks(rotation=90)

    # --- Second X axis for timestamp ---
    if any(["timestamp" in sub.columns for sub in datasets]):
        ref = datasets[0]
        if "timestamp" in ref.columns and not ref["timestamp"].isna().all():
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ts = ref["timestamp"].reset_index(drop=True)
            run_idx = ref["run_index"].reset_index(drop=True)

            # === Auto tick spacing ===
            total_runs = len(run_idx)
            max_labels = 8  # you can tweak this (fewer labels if plot太擠)
            step = max(1, total_runs // max_labels)

            tick_idx = list(range(0, total_runs, step))
            if tick_idx[-1] != total_runs - 1:
                tick_idx.append(total_runs - 1)  # ensure last label included

            ts_labels = [ts[i].strftime("%H:%M:%S") for i in tick_idx]
            ax2.set_xticks(run_idx.iloc[tick_idx])
            ax2.set_xticklabels(ts_labels, rotation=90)
            ax2.set_xlabel("Timestamp (HH:MM:SS)")

    fig.tight_layout()
    fig.savefig(out_dir / f"all_qubits_{f}_line_double_x.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"✅ Done. Charts saved to: {out_dir.resolve()}")
