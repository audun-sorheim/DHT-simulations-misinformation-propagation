# DHT-simulations-misinformation-propagation
A github for my project and masters thesis, which is on the propagation of misinformation. 
This repository contains all code used for the project thesis delivered on 21.06.2025.

## Project thesis: "Beliefs and the propagation ofmisinformation in social networks"

The plots used in the project thesis paper are in the plots folder. 
The code used to produce the results shown in the plots lies in:

~~DHT_main_simulation_file.py~~ 
main.py

Which uses code existing in the following python-files to get results, and saves them to a .npz-file.

agents.py - to initialize agents and beliefs.
metrics.py - calculates all metrics used for results, also holds normalization functions.
networks.py - generates all networks used to get results.
plots.py - generates plots from results.
simulation.py - contains code to update beliefs and run one entire simulation.

The code is commented, but retains a lot of "old artifacts" no longer in use. Throughout the summer of 2025 redundant lines and functions might be removed, and readability might be improved, but the functional code used to produce results will stay untouched.

To produce the data on your own, simply follow the instructions in the method section of the paper and edit the parameters in the code accordingly.

# Simulation timing

## $N = 100$, ER-network $k=N/10$

Creating 200 graphs took less than one second. It then uses ~10-15 seconds to initialize everything. Then it uses 25 seconds to run 200 simulations.

In total: ~35-40 seconds.

## $N = 1000$, ER-network $k=N/10$

Creating 200 graphs took 34 seconds. Then it uses 200 seconds to run 200 simulations. Then it uses ~60 seconds to preocess and upload results.

In total: ~294 seconds.

## $N = 10000$, ER-network $k=N/10$

My computer ran out of memory when converting the graphs into adjacency-matrices of size $(10000, 10000)$. We should consider using sparse arrays if compatible with numba.