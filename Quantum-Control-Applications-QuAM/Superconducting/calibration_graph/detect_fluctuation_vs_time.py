import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from qualibrate import QualibrationNode, NodeParameters

from node_for_repeat import run_flux_offset, run_T1, run_T2_echo, run_ge

def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

# --- 可修改的參數 ---
num_runs = None   # None means keep running until Ctrl+C
cooldown_sec = 60  # CD time between runs in seconds

# --- node parameters ---
# custimize parameters for each function here

parameters_for_functions = {
    "flux_offset": NodeParameters(
        qubits = None
        num_averages = 500
        frequency_detuning_in_mhz = 8.0
        min_wait_time_in_ns = 16
        max_wait_time_in_ns = 500
        wait_time_step_in_ns = 10
        flux_span = 0.04
        flux_step = 0.001
        flux_point_joint_or_independent = "independent"
        simulate = False
        simulation_duration_ns = 2500
        timeout = 100
        load_data_id = None
        multiplexed = False
    ),
    "T1": NodeParameters(
        qubits = None
        num_averages = 300
        min_wait_time_in_nst = 16
        max_wait_time_in_ns = 90000
        wait_time_step_in_ns = 300
        flux_point_joint_or_independent_or_arbitrary = "independent"
        reset_type = "thermal"  #["active", "thermal"]
        use_state_discrimination = False
        simulatel = False
        simulation_duration_ns = 2500
        timeout = 100
        load_data_id = None
        multiplexed = True
    ),
    "T2_echo": NodeParameters(
        qubits= None
        num_averages = 300
        min_wait_time_in_ns = 16
        max_wait_time_in_ns = 50000
        wait_time_step_in_ns = 300
        flux_point_joint_or_independent_or_arbitrary = "arbitrary"
        reset_type = "thermal"  #["active", "thermal"]
        use_state_discrimination = True
        simulate = False
        simulation_duration_ns = 2500
        timeout = 100
        load_data_id = None
        multiplexed = True
    ),
    "GE": NodeParameters(
        qubits = None
        num_averages = 500
        use_two_state_discrimination = True
        simulate = False
        simulation_duration_ns = 2500
        timeout = 100
        load_data_id = None
        multiplexed = False
    ),
}

# --- node options, comment for skip ---
functions_to_run = {
    "flux_offset": run_flux_offset,
    "T1": run_T1,
    "T2_echo": run_T2_echo,
    "GE": run_ge,
}

# --- Data path ---
out_dir = Path(r'd:\qm_code\as\qua-libs\Quantum-Control-Applications-QuAM\Superconducting\data')
out_dir.mkdir(parents=True, exist_ok=True)

session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
excel_file = out_dir / f"qubit_measurements_{session_ts}.xlsx"

# --- initialization ---
qubit_dfs = {}  # DataFrames for each qubit
qubit_names_global = None  # initialize qubit names

print("Press Ctrl+C to stop running.\n")

i = 0
try:
    # while loop for a limited or unlimited runs 
    while True if num_runs is None else i < num_runs:
        i += 1
        ts = datetime.now().isoformat(timespec="seconds")
        run_label = f"{i}" if num_runs is None else f"{i:02d}/{num_runs}"
        print(f"[{run_label}] {ts} Starting run…")

        run_results = {}

        try:
            # --- execute node ---
            for fname, func in functions_to_run.items():
                params = parameters_for_functions.get(fname, None)
                result = func(params) if params is not None else func()
                *values, qubits_name = result
                run_results[fname] = values

            # --- 初始化 qubit_names_global ---
            if qubit_names_global is None:
                qubit_names_global = qubits_name
                for q in qubit_names_global:
                    qubit_dfs[q] = pd.DataFrame(columns=["run_index", "timestamp", *functions_to_run.keys()])

            # --- append this results ---
            for idx, q in enumerate(qubit_names_global):
                row = {"run_index": i, "timestamp": ts}
                for fname in functions_to_run.keys():
                    row[fname] = run_results[fname][idx]
                qubit_dfs[q] = pd.concat([qubit_dfs[q], pd.DataFrame([row])], ignore_index=True)

            print(f"[{run_label}] Run completed and appended.")

        except Exception as e:
            print(f"[{run_label}] error: {e!r}, continuing…")

        time.sleep(cooldown_sec)

except KeyboardInterrupt:
    print("\n🟡 Detected Ctrl+C — stopping gracefully…")

# --- to Excel ---
if qubit_dfs:
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for qname, df in qubit_dfs.items():
            df.to_excel(writer, sheet_name=qname, index=False)
    print(f"All runs finished. Saved to {excel_file}")
else:
    print("No data collected — nothing saved.")
