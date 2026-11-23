# Platoon simulation (ENPM 667 Project 1)

Python code for every simulation figure and table in the report
(Sections 9 and 10, Tables 8 and 10, and the block diagram in Fig. 4).
It is the simple Method 1 code with Method 2 added in the same style.

Requirements: Python 3 with numpy and matplotlib

    pip install numpy matplotlib

Files
- platoon_sim.py   vehicle model, leader, disturbances, noise, MCTH, sliding surfaces,
                   FTRE and the three controllers:
                   method="linear" (benchmark), "dob" (Method 1), "stismc" (Method 2)
- make_figures.py  runs every case (with and without noise), draws all figures,
                   writes the table values
- make_diagram.py  draws the Method 2 block diagram (fig_m2_block.png)

How to run (from this folder)

    python make_figures.py --rerun    # runs all simulations (a few minutes), then plots
    python make_figures.py            # re-plots only, reuses saved results in results/
    python make_diagram.py

Output: figures/*.png, figures/metrics.json, figures/metrics.tex (Table 8),
figures/metrics_table2.tex (Table 10). Simulation results are cached in results/*.npz.
