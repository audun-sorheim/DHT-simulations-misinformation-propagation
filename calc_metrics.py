import numpy as np
import argparse
import os
from tqdm import tqdm
from metrics import truthfulness, cognitive_dissonance, calculate_beliefs

def main():
    parser = argparse.ArgumentParser(description="Calculate results from npz-files.")
    parser.add_argument("--filepath")
    parser.add_argument("--num_iterations")
    args = parser.parse_args()

    T0s = np.zeros((10, int(args.num_iterations) + 1))
    T1s = np.zeros((10, int(args.num_iterations) + 1))
    T2s = np.zeros((10, int(args.num_iterations) + 1))
    T3s = np.zeros((10, int(args.num_iterations) + 1))

    CDs = np.zeros((10, int(args.num_iterations) + 1))

    files = [f for f in os.listdir(args.filepath) if f.endswith('.npz')]

    for i, f in enumerate(tqdm(files, desc="Processing files")):
        if f.endswith('.npz'):
            name = os.path.splitext(f)[0]
            data = np.load(os.path.join(args.filepath, f))
            q = data['private']
            p = data['public']

            T0 = truthfulness(q, 0)
            T1 = truthfulness(q, 1)
            T2 = truthfulness(q, 2)
            T3 = truthfulness(q, 3)
            T0s[i] = T0
            T1s[i] = T1
            T2s[i] = T2
            T3s[i] = T3

            CD = cognitive_dissonance(q, p)
            CDs[i] = CD
    
    T0_f = np.mean(T0s, axis=0)
    T1_f = np.mean(T1s, axis=0)
    T2_f = np.mean(T2s, axis=0)
    T3_f = np.mean(T3s, axis=0)
    
    CD_f = np.mean(CDs, axis=0)

    np.savez_compressed("results" + name,
                        T0=T0_f,
                        T1=T1_f,
                        T2=T2_f,
                        T3=T3_f,
                        CD=CD_f)
    
    return None

if __name__=='__main__':
    main()