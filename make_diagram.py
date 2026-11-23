"""
Draws the block diagram of Method 2 (adaptive dual-layer STISMC with the FTRE)
for the report: figures/fig_m2_block.png.   Run:  python make_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 9

fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, title, text, color="#e8f0fb"):
    """Rounded box with a bold title and a short line of maths."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=color, ec="#1f1f1c", lw=1.0))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", weight="bold", fontsize=8)
    ax.text(x + w / 2, y + h * 0.32, text, ha="center", va="center", fontsize=8.5)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f1f1c"))


def label(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=8.5)


# top row: surfaces -> super-twisting -> compensating input -> vehicle
box(0.2, 5.4, 3.0, 1.8, "MCTH + surfaces", r"$\bar e_i,\ s_i,\ S_i$")
box(4.3, 5.4, 3.2, 1.8, "Super-twisting law", r"$\bar u_i$, Eq. (30)")
box(8.6, 5.4, 3.4, 1.8, "Compensating input", r"$u_i$, Eq. (15)", color="#fdf0e6")
box(13.0, 5.4, 2.8, 1.8, "Vehicle $i$", r"Eq. (1)", color="#eeeeea")
arrow(3.3, 6.3, 4.2, 6.3); label(3.75, 6.6, r"$S_i$")
arrow(7.6, 6.3, 8.5, 6.3); label(8.05, 6.6, r"$\bar u_i$")
arrow(12.1, 6.3, 12.9, 6.3); label(12.5, 6.6, r"$u_i$")

# feedback of measured positions and velocities
ax.plot([14.4, 14.4, 1.7], [5.3, 4.5, 4.5], color="#1f1f1c", lw=1.0)
arrow(1.7, 4.5, 1.7, 5.3)
label(5.6, 4.8, r"measured $x,\ v$ (vehicles $i-1,\ i,\ i+1$)")

# FTRE below the compensating input
box(8.6, 1.6, 3.4, 1.8, "FTRE", r"$\hat a_{i+1},\ \hat\delta_{vi}$, Eqs. (12)-(13)", color="#e6f6ef")
ax.plot([10.3, 10.3], [4.5, 3.5], color="#1f1f1c", lw=1.0)
arrow(10.3, 4.5, 10.3, 3.5)
ax.plot([12.1, 12.6, 12.6, 11.4], [2.5, 2.5, 5.0, 5.0], color="#1f1f1c", lw=1.0)
arrow(11.4, 5.0, 11.4, 5.3)
label(13.4, 2.9, r"$\bar A_i,\ \bar B_i$")

# adaptation loop below the super-twisting law
box(0.2, 0.4, 3.0, 1.8, "Low-pass filter", r"$\nu_{fi}$, Eq. (34)", color="#f3ecfa")
box(4.3, 0.4, 3.2, 1.8, "Dual-layer adaptation", r"$\Omega_i,\ \rho_i$, Eq. (35)", color="#f3ecfa")
ax.plot([7.55, 7.9, 7.9], [6.0, 6.0, 3.3], color="#1f1f1c", lw=1.0)
ax.plot([7.9, 1.7], [3.3, 3.3], color="#1f1f1c", lw=1.0)
arrow(1.7, 3.3, 1.7, 2.3)
label(4.9, 3.6, r"$k_{2i}\,\mathrm{sgn}(S_i)$")
arrow(3.3, 1.3, 4.2, 1.3); label(3.75, 1.6, r"$\nu_{fi}$")
arrow(5.9, 2.3, 5.9, 5.3)
label(6.05, 4.0, r"$k_{1i}=\Omega_i^{1/2}k_{b1i},\ k_{2i}=\Omega_ik_{b2i}$", ha="left")

fig.savefig(os.path.join(HERE, "figures", "fig_m2_block.png"), dpi=200, bbox_inches="tight")
