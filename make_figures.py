"""
Runs every simulation case used in the report, then makes the figures
(figures/*.png), the metrics file (figures/metrics.json) and the LaTeX table
rows (figures/metrics.tex for Table 8, figures/metrics_table2.tex for Table 10,
our version of Table II of the paper).

    python make_figures.py           re-plot; only simulates cases not saved yet
    python make_figures.py --rerun   re-run every simulation (a few minutes)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save figures to files, no window
import matplotlib.pyplot as plt

import platoon_sim as ps

# Folders (next to this file)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")


# plot configuration
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
INK = "#1f1f1c"        # text colour
MUTED = "#6b6a63"      # tick colour
GRID = "#e1e0d9"       # grid colour

plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.edgecolor"] = "#c3c2b7"
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["text.color"] = INK
plt.rcParams["xtick.color"] = MUTED
plt.rcParams["ytick.color"] = MUTED
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = GRID
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["lines.linewidth"] = 1.1
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


# Simulation cases
# Each case has a name, a controller ("dob" = Method 1, "stismc" = Method 2,
# "linear" = benchmark), a parameter set, and whether the MCTH spacing policy is used.
# Every case is run twice: with noise (saved as "name") and without noise (saved as "name_clean").
PAPER_PARAMS = ps.params()                                         # gains printed in the paper
TUNED_M1_PARAMS = ps.params(ks=50.0, alpha=50.0, ka1=5.0, ka2=0.5)   # retuned Method 1
TUNED_LIN_PARAMS = ps.params(l_lin=5.0, eta_lin=0.5)               # retuned benchmark
FIXED_M2_PARAMS = ps.params(adapt=False, Omega0=1.0)               # Method 2, fixed gains
THM3_M2_PARAMS = ps.params(kb1=2.0 * np.sqrt(2.0 * 3.3))           # Method 2, kb1 = 2*sqrt(2*kb2)

#          name         controller  parameters        use MCTH
CASES = [["m1_paper",  "dob",      PAPER_PARAMS,     True],
         ["m1_tuned",  "dob",      TUNED_M1_PARAMS,  True],
         ["lin_paper", "linear",   PAPER_PARAMS,     True],
         ["lin_tuned", "linear",   TUNED_LIN_PARAMS, True],
         ["m1_cth",    "dob",      PAPER_PARAMS,     False],
         # Method 2 (adaptive dual-layer STISMC)
         ["m2_paper",  "stismc",   PAPER_PARAMS,     True],
         ["m2_thm3",   "stismc",   THM3_M2_PARAMS,   True],
         ["m2_fixed",  "stismc",   FIXED_M2_PARAMS,  True],
         ["m2_cth",    "stismc",   PAPER_PARAMS,     False]]

# Text shown in legends and in the table
LABEL = {}
LABEL["m1_paper"] = "Method 1, paper gains"
LABEL["m1_tuned"] = "Method 1, retuned"
LABEL["lin_paper"] = "Linear DOB-ISMC, paper gains"
LABEL["lin_tuned"] = "Linear DOB-ISMC, retuned"
LABEL["m2_paper"] = "Method 2, paper gains"
LABEL["m2_thm3"] = r"Method 2, $k_{b1}=2\sqrt{2k_{b2}}$"
LABEL["m2_fixed"] = "Method 2, fixed gains"

# Controllers compared in the bar chart, the vehicle-1 plot and the table with one colour each
COMPARE = ["lin_paper", "m1_paper", "lin_tuned", "m1_tuned"]
COMPARE_COLORS = ["#c3c2b7", "#2a78d6", "#8d8c84", "#eb6834"]

# Controllers compared in the Method 2 part (Table II and the comparison figures)
COMPARE2 = ["lin_paper", "m1_paper", "m1_tuned", "m2_paper", "m2_thm3"]
COMPARE2_COLORS = ["#c3c2b7", "#2a78d6", "#eb6834", "#1baf7a", "#7a4fb8"]


# Running and measuring
def load_or_run(key, method, p, mcth, noise, rerun):
    """Load a saved result, or run the simulation (then save it)."""
    path = os.path.join(RESULTS_DIR, key + ".npz")

    if os.path.exists(path) and not rerun:
        # Read the saved file into a normal dictionary
        saved = np.load(path)
        hist = {}
        for name in saved.files:
            hist[name] = saved[name]
    else:
        print("simulating " + key + " ...")
        hist = ps.simulate(method=method, mcth=mcth, noise=noise, p=p)
        hist = ps.downsample(hist, every=10)

    np.savez_compressed(path, **hist)
    return hist


def compute_metrics(hist):
    """Peak and RMS of the true MCTH error, plus a few other peaks."""
    eb_true = hist["eb_true"]                  # actual (not measured) error

    peak_eb = np.abs(eb_true).max(axis=0)       # one value per vehicle
    rms_eb = np.sqrt((eb_true ** 2).mean(axis=0))

    metrics = {}
    metrics["peak_eb"] = peak_eb.tolist()
    metrics["rms_eb"] = rms_eb.tolist()
    metrics["peak_S"] = float(np.abs(hist["S"]).max())
    metrics["peak_Lt"] = float(np.abs(hist["Lt"]).max())
    metrics["peak_u"] = float(np.abs(hist["u"]).max())

    # RMS of the CTH error e_i (true positions and velocities), used for
    # Table II of the paper. It includes the start-up error.
    e = e_true(hist)
    metrics["rms_e"] = np.sqrt((e ** 2).mean(axis=0)).tolist()
    return metrics


def e_true(hist):
    """CTH error e_i computed from the true (noise-free) states."""
    # position of the vehicle in front: leader for vehicle 1, then vehicles 1..5
    x_front = np.column_stack([hist["x0"], hist["x"][:, :-1]])
    return x_front - hist["x"] - ps.PARAMS["xd"] - ps.PARAMS["h"] * hist["v"]


# Plotting helpers
def save(fig, name):
    """Save a figure as PNG in the figures folder and close it."""
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def plot_vehicles(ax, t, Y, ylabel, title):
    """Plot one line per vehicle (column i of Y is vehicle i+1)."""
    for i in range(6):
        ax.plot(t, Y[:, i], color=COLORS[i], ls=STYLES[i], label="$i=" + str(i + 1) + "$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xlim(t[0], t[-1])


# Figures
def plot_six_panel(hist, name):
    """Six panels: errors, sliding surface, L tilde, velocity, input."""
    t = hist["t"]
    fig, ax = plt.subplots(3, 2, figsize=(6.6, 7.4), sharex=True)

    plot_vehicles(ax[0, 0], t, hist["e"], r"$e_i$ [m]", r"(a) CTH spacing error $e_i$")
    plot_vehicles(ax[0, 1], t, hist["eb"], r"$\bar e_i$ [m]", r"(b) MCTH spacing error $\bar e_i$")
    plot_vehicles(ax[1, 0], t, hist["S"], r"$S_i$", r"(c) Coupled sliding surface $S_i$")
    plot_vehicles(ax[1, 1], t, hist["Lt"], r"$\tilde L_i$", r"(d) Lumped-uncertainty error $\tilde L_i$")
    plot_vehicles(ax[2, 0], t, hist["v"], r"$v_i$ [m/s]", r"(e) Velocity $v_i$")
    ax[2, 0].plot(t, hist["v0"], color=INK, lw=0.9, label="leader")
    plot_vehicles(ax[2, 1], t, hist["u"], r"$u_i$ [N/kg]", r"(f) Control input $u_i$")

    ax[2, 0].set_xlabel("time [s]")
    ax[2, 1].set_xlabel("time [s]")

    # One legend for the whole figure, taken from the velocity panel
    handles, labels = ax[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, name)


def plot_compare_peaks(metrics):
    """Bar chart of the peak MCTH error per vehicle, without and with noise."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    vehicles = np.arange(6)
    width = 0.2

    # Four bars per vehicle, one for each controller, shifted side by side
    for j in range(4):
        key = COMPARE[j]
        offset = (j - 1.5) * width
        clean_peaks = metrics[key + "_clean"]["peak_eb"]
        noisy_peaks = metrics[key]["peak_eb"]
        ax[0].bar(vehicles + offset, clean_peaks, width * 0.9,
                  color=COMPARE_COLORS[j], label=LABEL[key])
        ax[1].bar(vehicles + offset, noisy_peaks, width * 0.9,
                  color=COMPARE_COLORS[j])

    ax[0].set_ylabel(r"peak $|\bar e_i|$ [m]")
    ax[0].set_title("(a) No measurement noise", loc="left")
    ax[1].set_title("(b) With 5% velocity noise", loc="left")

    tick_names = ["1", "2", "3", "4", "5", "6"]
    for panel in [ax[0], ax[1]]:
        panel.set_xticks(vehicles, tick_names)
        panel.set_xlabel("vehicle $i$")
        panel.grid(axis="x", visible=False)

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "fig_compare_peaks")


def plot_compare_vehicle1(results):
    """Vehicle 1 (no noise) under all four controllers."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)

    for j in range(4):
        key = COMPARE[j]
        hist = results[key + "_clean"]
        ax[0].plot(hist["t"], hist["eb"][:, 0], color=COMPARE_COLORS[j], lw=1.2, label=LABEL[key])
        ax[1].plot(hist["t"], hist["Lt"][:, 0], color=COMPARE_COLORS[j], lw=1.2)

    ax[0].set_ylabel(r"$\bar e_1$ [m]")
    ax[0].set_title(r"(a) MCTH error of vehicle 1", loc="left")
    ax[1].set_ylabel(r"$\tilde L_1$")
    ax[1].set_title(r"(b) Lumped-uncertainty error, vehicle 1", loc="left")

    t_end = hist["t"][-1]
    for panel in [ax[0], ax[1]]:
        panel.set_xlabel("time [s]")
        panel.set_xlim(0, t_end)

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "fig_compare_v1")


def plot_mcth_vs_cth(results):
    """Method 1 (paper gains, no noise) with MCTH versus plain CTH."""
    mcth = results["m1_paper_clean"]
    cth = results["m1_cth_clean"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)

    plot_vehicles(ax[0], mcth["t"], mcth["eb"], r"spacing error [m]", r"(a) With MCTH: $\bar e_i$")
    plot_vehicles(ax[1], cth["t"], cth["e"], "", r"(b) Plain CTH ($\psi_i\equiv0$): $e_i$")
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "fig_mcth_vs_cth")


# Figures for Method 2
def put_legend_on_top(fig, ax, ncol):
    """One legend above the figure, taken from the lines of panel ax."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.07))


def plot_m2_mcth_vs_cth(results):
    """Method 2 (paper gains, no noise): CTH error e_i with and without MCTH."""
    mcth = results["m2_paper_clean"]
    cth = results["m2_cth_clean"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)

    plot_vehicles(ax[0], mcth["t"], mcth["e"], r"$e_i$ [m]", r"(a) With MCTH")
    plot_vehicles(ax[1], cth["t"], cth["e"], "", r"(b) Plain CTH ($\psi_i\equiv0$)")
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")

    put_legend_on_top(fig, ax[0], 6)
    fig.tight_layout()
    save(fig, "fig_m2_mcth_vs_cth")


def plot_m2_surfaces(hist):
    """Integral sliding surface s_i and coupled surface S_i, in millimetres."""
    t = hist["t"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    plot_vehicles(ax[0], t, 1000 * hist["s"], r"$s_i$ [mm]", r"(a) Integral sliding surface $s_i$")
    plot_vehicles(ax[1], t, 1000 * hist["S"], r"$S_i$ [mm]", r"(b) Coupled sliding surface $S_i$")
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")

    put_legend_on_top(fig, ax[0], 6)
    fig.tight_layout()
    save(fig, "fig_m2_surfaces")


def plot_m2_states(hist):
    """Positions and velocities under Method 2, with the leader."""
    t = hist["t"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    plot_vehicles(ax[0], t, hist["x"], r"$x_i$ [m]", r"(a) Position $x_i$")
    ax[0].plot(t, hist["x0"], color=INK, lw=0.9, label="leader")
    plot_vehicles(ax[1], t, hist["v"], r"$v_i$ [m/s]", r"(b) Velocity $v_i$")
    ax[1].plot(t, hist["v0"], color=INK, lw=0.9)
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")

    put_legend_on_top(fig, ax[0], 7)
    fig.tight_layout()
    save(fig, "fig_m2_states")


def plot_m2_gains(hist):
    """(a) Omega_2 against |xi_2| = |dL_2/dt|, (b) adaptive gain k_1i(t)."""
    t = hist["t"]
    xi = np.gradient(hist["L"][:, 1], t)            # xi_2 = dL_2/dt (numerical derivative)
    win = 50                                        # 50 samples = 50 ms moving average
    xi_smooth = np.convolve(np.abs(xi), np.ones(win) / win, mode="same")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax[0].plot(t, np.abs(xi), color="#c3c2b7", lw=0.6, label=r"$|\xi_2|$")
    ax[0].plot(t, xi_smooth, color=MUTED, lw=1.0, label=r"$|\xi_2|$, 50 ms average")
    ax[0].plot(t, hist["Omega"][:, 1], color=COLORS[0], lw=1.4, label=r"$\Omega_2$")
    ax[0].set_ylim(0, 1.3 * hist["Omega"][:, 1].max())
    ax[0].set_title(r"(a) $\Omega_2$ and $|\xi_2|=|\dot L_2|$, vehicle 2", loc="left")
    ax[0].legend(frameon=False, loc="upper right")

    plot_vehicles(ax[1], t, hist["k1"], r"$k_{1i}(t)$", r"(b) Adaptive gain $k_{1i}(t)$")
    ax[1].legend(frameon=False, ncol=2, loc="upper right")

    for panel in [ax[0], ax[1]]:
        panel.set_xlabel("time [s]")
        panel.set_xlim(t[0], t[-1])
    fig.tight_layout()
    save(fig, "fig_m2_gains")


def plot_m2_ftre(hist):
    """FTRE estimates: rear acceleration (vehicle 1) and delta_v (vehicles 1, 2)."""
    t = hist["t"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    # (a) vehicle 1 estimates the acceleration of vehicle 2
    ax[0].plot(t, hist["a"][:, 1], color=INK, lw=1.0, label=r"$a_2$")
    ax[0].plot(t, hist["ahat"][:, 0], color=COLORS[1], ls="--", lw=1.0, label=r"$\hat a_2$")
    ax[0].set_title(r"(a) Rear acceleration seen by vehicle 1", loc="left")
    ax[0].set_ylabel(r"[m/s$^2$]")

    # (b) mismatched disturbance of vehicles 1 and 2, true (solid) and estimate (dashed)
    ax[1].plot(t, hist["dv"][:, 0], color=COLORS[0], lw=1.0, label=r"$\delta_{v1}$")
    ax[1].plot(t, hist["dhat"][:, 0], color=COLORS[0], ls="--", lw=1.0, label=r"$\hat\delta_{v1}$")
    ax[1].plot(t, hist["dv"][:, 1], color=COLORS[2], lw=1.0, label=r"$\delta_{v2}$")
    ax[1].plot(t, hist["dhat"][:, 1], color=COLORS[2], ls="--", lw=1.0, label=r"$\hat\delta_{v2}$")
    ax[1].set_title(r"(b) Mismatched disturbance", loc="left")
    ax[1].set_ylabel("[m/s]")

    ax[0].legend(frameon=False, ncol=2, loc="upper right")
    ax[1].legend(frameon=False, ncol=2, loc="lower left")
    for panel in [ax[0], ax[1]]:
        panel.set_xlabel("time [s]")
        panel.set_xlim(t[0], t[-1])
    fig.tight_layout()
    save(fig, "fig_m2_ftre")


def plot_m2_fixed_vs_adaptive(results):
    """MCTH error with adaptive gains and with fixed gains (Fig. 8 of the paper)."""
    ad = results["m2_paper_clean"]
    fx = results["m2_fixed_clean"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)

    plot_vehicles(ax[0], ad["t"], 1000 * ad["eb"], r"$\bar e_i$ [mm]", r"(a) Adaptive gains")
    plot_vehicles(ax[1], fx["t"], 1000 * fx["eb"], "",
                  r"(b) Fixed gains $k_{1i}=k_{b1},\ k_{2i}=k_{b2}$")
    ax[0].set_xlabel("time [s]")
    ax[1].set_xlabel("time [s]")

    put_legend_on_top(fig, ax[0], 6)
    fig.tight_layout()
    save(fig, "fig_m2_fixed_vs_adaptive")


def plot_compare2_peaks(metrics):
    """Peak MCTH error per vehicle for all controllers, without and with noise."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    vehicles = np.arange(6)
    width = 0.16

    # Five bars per vehicle, one for each controller, shifted side by side
    for j in range(5):
        key = COMPARE2[j]
        offset = (j - 2) * width
        ax[0].bar(vehicles + offset, metrics[key + "_clean"]["peak_eb"], width * 0.9,
                  color=COMPARE2_COLORS[j], label=LABEL[key])
        ax[1].bar(vehicles + offset, metrics[key]["peak_eb"], width * 0.9,
                  color=COMPARE2_COLORS[j])

    ax[0].set_ylabel(r"peak $|\bar e_i|$ [m]")
    ax[0].set_title("(a) No measurement noise", loc="left")
    ax[1].set_title("(b) With 5% velocity noise", loc="left")

    tick_names = ["1", "2", "3", "4", "5", "6"]
    for panel in [ax[0], ax[1]]:
        panel.set_xticks(vehicles, tick_names)
        panel.set_xlabel("vehicle $i$")
        panel.grid(axis="x", visible=False)

    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save(fig, "fig_compare2_peaks")


def plot_compare2_vehicle1(results):
    """MCTH error of vehicle 1 for all controllers, without and with noise."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)

    for j in range(5):
        key = COMPARE2[j]
        clean = results[key + "_clean"]
        noisy = results[key]
        # left: no noise (this line also gives the legend entry), right: with noise
        ax[0].plot(clean["t"], clean["eb_true"][:, 0], color=COMPARE2_COLORS[j], lw=1.1,
                   label=LABEL[key])
        ax[1].plot(noisy["t"], noisy["eb_true"][:, 0], color=COMPARE2_COLORS[j], lw=1.1)

    ax[0].set_ylabel(r"$\bar e_1$ [m]")
    ax[0].set_title("(a) No measurement noise", loc="left")
    ax[1].set_title("(b) With 5% velocity noise", loc="left")
    t_end = noisy["t"][-1]
    for panel in [ax[0], ax[1]]:
        panel.set_xlabel("time [s]")
        panel.set_xlim(0, t_end)

    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save(fig, "fig_compare2_v1")


# Tables
def write_metrics_table(metrics):
    """LaTeX table rows: controller, noise, peak |eb| per vehicle, mean RMS [mm]."""
    lines = []
    for key in COMPARE:
        # first the run without noise, then the run with noise
        for run_key, noise_text in [[key + "_clean", "no"], [key, "yes"]]:
            m = metrics[run_key]

            row = LABEL[key] + " & " + noise_text
            for value in m["peak_eb"]:
                row = row + " & " + "%.3f" % value          # 3 decimals per vehicle
            mean_rms_mm = 1000 * np.mean(m["rms_eb"])
            row = row + " & " + "%.1f" % mean_rms_mm + " \\\\"
            lines.append(row)

    file = open(os.path.join(FIG_DIR, "metrics.tex"), "w")
    for row in lines:
        file.write(row + "\n")
    file.close()


def write_table2(metrics):
    """Table II of the paper: RMS of e_i in cm per vehicle, plus mean RMS of eb_i in mm."""
    lines = []
    for key in COMPARE2:
        # first the run without noise, then the run with noise
        for run_key, noise_text in [[key + "_clean", "no"], [key, "yes"]]:
            m = metrics[run_key]

            label = LABEL[key]
            if key == "m2_thm3":
                label = r"Method 2, $k_{b1i}=2\sqrt{2k_{b2i}}$"   # with the index i, as in the report

            row = label + " & " + noise_text
            for value in m["rms_e"]:
                row = row + " & " + "%.2f" % (100 * value)    # metres -> cm, 2 decimals
            mean_rms_eb_mm = 1000 * np.mean(m["rms_eb"])
            row = row + " & " + "%.2f" % mean_rms_eb_mm + " \\\\"
            lines.append(row)

    file = open(os.path.join(FIG_DIR, "metrics_table2.tex"), "w")
    for row in lines:
        file.write(row + "\n")
    file.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rerun = "--rerun" in sys.argv

    # 1) Run (or load) every case, with and without noise
    results = {}
    metrics = {}
    for case in CASES:
        name = case[0]
        method = case[1]
        p = case[2]
        mcth = case[3]

        for noise in [True, False]:
            if noise:
                key = name
            else:
                key = name + "_clean"
            results[key] = load_or_run(key, method, p, mcth, noise, rerun)
            metrics[key] = compute_metrics(results[key])
            print(key.ljust(16), "peak|eb| =", np.round(metrics[key]["peak_eb"], 3))

    file = open(os.path.join(FIG_DIR, "metrics.json"), "w")
    json.dump(metrics, file, indent=1)
    file.close()

    # 2) Figures (time histories use the noise-free runs)
    plot_six_panel(results["m1_paper_clean"], "fig_m1_paper")
    plot_six_panel(results["m1_tuned_clean"], "fig_m1_tuned")
    plot_six_panel(results["lin_paper_clean"], "fig_lin_paper")
    plot_compare_peaks(metrics)
    plot_compare_vehicle1(results)
    plot_mcth_vs_cth(results)

    # 3) Method 2 figures (time histories use the noise-free runs)
    m2 = results["m2_paper_clean"]
    plot_six_panel(m2, "fig_m2_paper")
    plot_m2_mcth_vs_cth(results)
    plot_m2_surfaces(m2)
    plot_m2_states(m2)
    plot_m2_gains(m2)
    plot_m2_ftre(m2)
    plot_m2_fixed_vs_adaptive(results)
    plot_compare2_peaks(metrics)
    plot_compare2_vehicle1(results)

    # 4) Tables for the report
    write_metrics_table(metrics)
    write_table2(metrics)


if __name__ == "__main__":
    main()
