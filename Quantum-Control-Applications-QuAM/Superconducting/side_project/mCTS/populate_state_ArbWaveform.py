# %% {imports}
import numpy as np
import string
import secrets
from quam_libs.components import QuAM
from quam_libs.lib.pulses import ArbWaveformPulse
from quam_libs.lib.CTS_builder import WaveformEngineer
# %% 
""" Input Zone """
state_PATH:str|None = None
target_q:str = 'q2'

tg = 16
aiming_sup_power_dB = -40
avoid_freqs_MHz = np.array([30])


## Warning !!! You might change them but it's better not to do this.
method = 'CRAB'
optimizer = 'L-BFGS-B'


""" Input Zone end, do NOT touch below ! """



# %%
if state_PATH is not None:
    machine = QuAM.load(state_PATH)
else:
    machine = QuAM.load()

q = machine.qubits[target_q]

# Checking operation name doesn't exist
angle_maps = {'x180':0, 'y180':np.pi / 2, 'x90':0, 'y90':np.pi / 2, '-x90':np.pi, '-y90':-np.pi / 2}
# waveform cooking
WE = WaveformEngineer(tg-1, avoid_freqs_MHz, aiming_sup_power_dB)
WE.compare_all_methods(optimizer)
x180_I_samples, x180_Q_samples = WE.waveforms[method]["I"], WE.waveforms[method]["Q"]
# operation suffix remark
operation_note = f"AVOID relative IF {', '.join(str(x) for x in avoid_freqs_MHz)} MHz"
# Serial number
def generate_safe_random_code(length=5):
    """
    random string
    """

    safe_special_chars = "!@#%^&_+-="
    character_pool = string.ascii_letters + string.digits + safe_special_chars
    random_code = ''.join(secrets.choice(character_pool) for _ in range(length))
    
    return random_code
serial_number = generate_safe_random_code(5)


## basic x180 and x90
q.xy.operations[f'x180_{method}'] = ArbWaveformPulse(
    length=len(x180_I_samples), 
    digital_marker="ON",
    I_samples=x180_I_samples,
    Q_samples=x180_Q_samples,
    axis_angle=angle_maps['x180'],
    annotation=operation_note,
    sn=serial_number
)
q.xy.operations[f'x90_{method}'] = ArbWaveformPulse(
    length=len(x180_Q_samples), 
    digital_marker="ON",
    I_samples=(np.array(x180_I_samples)/2).tolist(),
    Q_samples=(np.array(x180_Q_samples)/2).tolist(),
    axis_angle=angle_maps['x90'],
    annotation=operation_note,
    sn=serial_number
)
## Others by refering
q.xy.operations[f'-x90_{method}'] = ArbWaveformPulse(
    length=f"#../x90_{method}/length", 
    digital_marker="ON",
    I_samples=f"#../x90_{method}/I_samples",
    Q_samples=f"#../x90_{method}/Q_samples",
    axis_angle=angle_maps['-x90'],
    annotation=operation_note,
    sn=serial_number
)
q.xy.operations[f'y180_{method}'] = ArbWaveformPulse(
    length=f"#../x180_{method}/length", 
    digital_marker="ON",
    I_samples=f"#../x180_{method}/I_samples",
    Q_samples=f"#../x180_{method}/Q_samples",
    axis_angle=angle_maps['y180'],
    annotation=operation_note,
    sn=serial_number
)
q.xy.operations[f'y90_{method}'] = ArbWaveformPulse(
    length=f"#../x90_{method}/length", 
    digital_marker="ON",
    I_samples=f"#../x90_{method}/I_samples",
    Q_samples=f"#../x90_{method}/Q_samples",
    axis_angle=angle_maps['y90'],
    annotation=operation_note,
    sn=serial_number
)
q.xy.operations[f'-y90_{method}'] = ArbWaveformPulse(
    length=f"#../x90_{method}/length", 
    digital_marker="ON",
    I_samples=f"#../x90_{method}/I_samples",
    Q_samples=f"#../x90_{method}/Q_samples",
    axis_angle=angle_maps['-y90'],
    annotation=operation_note,
    sn=serial_number
)

# %%
machine.save()