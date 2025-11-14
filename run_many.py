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
FLEX_STRENGTH_values = [
    0.005, 0.01, 0.02, 0.03, 0.04, 0.050, 0.06, 0.07, 0.08, 0.09,
    0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
    0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
    0.850, 0.900, 0.950, 0.980, 0.985, 0.990, 0.995, 1.000
]

# S_values = [
#     0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.6, 2, 4, 8, 16
# ]

S_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.15]
# FLEX_STRENGTH_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09]

DIR_base = "N4096/SQUARE"

# === PARALLELISM SETTINGS ===
MAX_PARALLEL = 30  # run 30 simulations at the same time

# === BUILD ALL PARAMETER COMBINATIONS ===
combos = list(itertools.product(S_values, FLEX_STRENGTH_values))
print(f"Total simulations to run: {len(combos)}")  # should print 336

# === PREP LOG DIRECTORY ===
Path("logs").mkdir(exist_ok=True)
processes = []

# === LAUNCH LOOP ===
for i, (S, FLEX_STRENGTH) in enumerate(combos, start=1):
    env = defaults.copy()
    env.update({
        "S": S,
        "FLEX_STRENGTH": FLEX_STRENGTH,
        "DIR": f"{DIR_base}/flex{str(FLEX_STRENGTH).replace('.', '')}"
    })

    out_file = Path("logs") / f"run_{i:03d}_BA_s{str(S).replace('.', '')}_flex{str(FLEX_STRENGTH).replace('.', '')}.out"
    err_file = Path("logs") / f"run_{i:03d}_BA_s{str(S).replace('.', '')}_flex{str(FLEX_STRENGTH).replace('.', '')}.err"

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

print("\n✅ All 384 simulations complete. Check logs/ for output.")
