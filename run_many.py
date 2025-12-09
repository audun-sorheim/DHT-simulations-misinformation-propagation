import itertools
import subprocess
import time
from pathlib import Path

# === CONSTANT DEFAULTS ===
defaults = {
    "N": 4096,
    "K": 10,
    "NUM_ITERATIONS": 200,
    "NUM_SIMULATIONS": 20,
    "M": 5,
    "STD_DRAW": 0.75,
    "STD_LIKELIHOOD": 0.75,
    "FLEX_INTERVAL": "'0.0 0.0'",
    "CONFBIAS_BOOL": "True",
    "NUM_LOOPS": 10,
    "SAVE_ALL": "False",
    "GRAPH": "SQUARE",   # fixed graph type
}

# === PARAMETER ARRAYS ===
# FLEX_STRENGTH_values = [
#     0.005, 0.01, 0.02, 0.03, 0.04, 0.050, 0.06, 0.07, 0.08, 0.09,
#     0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
#     0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
#     0.850, 0.900, 0.950, 0.980, 0.985, 0.990, 0.995, 1.000
# ]
# FLEX_STRENGTH_values = [0.5]

# # S_values = [
# #     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.6, 2, 4, 8, 16
# # ]

# # S_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.15]
# start = 0.005
# end = 0.35
# # n = 401
# # step = (end - start) / (n - 1)
# step = 0.00075
# n = int((end - start) / step + 1)
# S_values = [round(start + i * step, 5) for i in range(n)]
# FLEX_STRENGTH_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09]

FLEX_STRENGTH_values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
# S_values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

# FLEX_STRENGTH_values = [0.98, 0.985, 0.99, 0.995, 0.999, 0.9995, 0.9999]
S_values = [10, 100, 1000, 10000, 100000, 1000000]

# start = 0.00025
# end = 0.34
# # n = 401
# # step = (end - start) / (n - 1)
# step = 0.00075
# n = int((end - start) / step + 1)
# print(n, type(n))
# FLEX_STRENGTH_values = [round(start + i * step, 5) for i in range(n)]
# FLEX_STRENGTH_values = [0.00025, 0.001, 0.00175, 0.0025, 0.00325, 0.004, 0.00475, 0.0055, 0.00625, 0.007, 0.00775, 0.0085, 0.00925, 0.01,
#                         0.31075, 0.3115, 0.31225, 0.313, 0.31375, 0.3145, 0.31525, 0.316, 0.31675, 0.3175, 0.31825, 0.319, 0.31975, 
#                         0.3205, 0.32125, 0.322, 0.32275, 0.3235, 0.32425, 0.325, 0.32575, 0.3265, 0.32725, 0.328, 0.32875, 0.3295, 
#                         0.33025, 0.331, 0.33175, 0.3325, 0.33325, 0.334, 0.33475, 0.3355, 0.33625, 0.337, 0.33775, 0.3385, 0.33925, 0.34]
# FLEX_STRENGTH_values = [0.0008, 0.0008, 0.0006, 0.0006, 0.0004, 0.0004, 0.0002, 0.00025, 0.0002, 0.0001, 0.0001]
# S_values = [1.0]

def encode_s(x):
    return f"{x:.5f}".replace('.', 'p')

def encode_f(x):
    return f"{x:5f}".replace('.', 'p')

DIR_base = "critical/SQUARE"

# === PARALLELISM SETTINGS ===
MAX_PARALLEL = 21  # run 30 simulations at the same time

# === BUILD ALL PARAMETER COMBINATIONS ===
combos = list(itertools.product(S_values, FLEX_STRENGTH_values))
print(f"Total simulations to run: {len(combos)}")  # should print 336

# === PREP LOG DIRECTORY ===
Path("logs").mkdir(exist_ok=True)
processes = []

# === LAUNCH LOOP ===
for i, (S, FLEX_STRENGTH) in enumerate(combos, start=1):

    s_token = encode_s(S)                              
    f_token = encode_f(FLEX_STRENGTH)

    env = defaults.copy()
    env.update({
        "S": S,
        "FLEX_STRENGTH": FLEX_STRENGTH,
        "DIR": f"{DIR_base}/s{s_token}"
    })

    out_file = Path("logs") / f"run_{i:03d}_SQUARE_s{s_token}_flex{f_token}.out"
    err_file = Path("logs") / f"run_{i:03d}_SQUARE_s{s_token}_flex{f_token}.err"

    cmd_vars = " ".join(f"{k}={v}" for k, v in env.items())
    full_cmd = f"bash -c \"{cmd_vars} ./run.sh\""

    print(f"[{i}/{len(combos)}] Launching: S={S}, FLEX_STRENGTH={FLEX_STRENGTH}")
    p = subprocess.Popen(full_cmd, shell=True,
                         stdout=open(out_file, "w"),
                         stderr=open(err_file, "w"))
    processes.append(p)

    # Maintain only MAX_PARALLEL active jobs
    while len(processes) >= MAX_PARALLEL:
        for proc in processes[:]:
            if proc.poll() is not None:
                processes.remove(proc)
        time.sleep(10)

# Wait for remaining to finish
for p in processes:
    p.wait()

print(f"\n✅ All {len(combos)} simulations complete. Check logs/ for output.")
