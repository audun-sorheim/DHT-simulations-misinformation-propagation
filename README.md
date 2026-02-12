# DHT-simulations-misinformation-propagation
A github for my project and masters thesis, which is on misinformation and belief dynamics. 
This repository contains all code used for the project thesis delivered on 21.06.2025 and master's thesis delivered on 26.01.2026.

## Master's thesis: "On Confirmation Bias and Cognitive Flexibility in Belief Dynamics"

The commits made on 12.02.2026 are only to clean up the repository, the code of the simulation program has _not_ been changed. Thus the final version of the repository commited on this date is valid for all results shown in my master's thesis. 

The plots and results are not included in the repository, but the code for producing them can be found in the .py and .ipynb files. NB! The paths to both data and output paths for plots are outdated in the code compared to the directory structure. Please read the methods chapter of "masters_thesis.pdf" for more information of how to produce results. The structure of the simulation program can be seen below. 

<img width="3375" height="3394" alt="Flowchart (1)" src="https://github.com/user-attachments/assets/54994541-416f-4b61-85fb-0041df534f63" /> 

When using large $N$ (here this means $N\geq256$) it is recommended to run the program on a remote computer preferably with more than 32 cores for parallellization.

To run simple results ($N=100$), just running run.sh is enough. When running $200$ simulations for $N=100$, the program typically uses $\sim 1$ minute to finish. If you want to plot the results on a square grid to visualize the belief dynamics of each agent, run _only_ $1$ simulation together with setting "save_all" in the args-list to true. 

When producing results for phase diagrams (large $N$, here I used $N=4096$). Run the file "run_many.py" and set the different configurations for the systems you want to run. 

### Simulation times for phase diagrams and lines

Outtake from Section 5.3 in "masters_thesis.pdf" for creating phase diagrams.

"One batch (20) of simulations took approximately ∼ 40−45 minutes. To reach 200 total simulations (one set), 10 batches were run for a total of ∼ 7 − 8 hours. Tables 5.3.3 and 5.3.4 contain the values used for s and ϕ for the phase diagrams. Multiplying the number of distinct values in the two respective subtables yields 32 · 21 = 672 sets of simulations for the phase diagram and 14 · 13 = 182 sets of simulations for the values in the four corners. A total of 854 sets of simulations. To compute this many, my co-supervisor, Astrid’s remote computer, "mindflayer," was used. It has an "AMD EPYC 7702P 64-Core" processor with 2 threads each [68]. On average, using 48 cores at a time yields a total run time of ∼ 5 − 6 days per phase diagram, for a total of 10 − 12 days across both the square and BA networks, excluding downtime between preparing new simulation sessions."

Outtake from Section 5.3 in "masters_thesis.pdf" for creating phase lines.

"For the lines in the phase diagram, an additional 930 sets of simulations were run on "mindflayer". Again, running on 48 cores on average yields a total run time of ∼ 6 − 7 days per network. For both the square and BA networks, this yields 2 weeks to complete the line simulations."

### Improvements needed

It is obvious from the above, that to run for larger systems, more parameters, or to average over more simulations. The program needs to be made more computationally effective. As of now, it is written in Python, with most of the computationally demanding code inside of numba-jitted functions. So in theory it should be fast. However, I am no computer scientist, so I expect there to exist more efficient ways to build up the program. Suggestions include translating the program into C++ or Julia or scrutinizing the way the systems are updated each timestep. 

## Project thesis: "Beliefs and the propagation ofmisinformation in social networks"

NB! The code used for the project thesis is outdated, to see the relevant repo, look for the last commit before august 2025.

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
