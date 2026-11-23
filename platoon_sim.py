"""
ENPM 667 Project 1 - simulation of the bidirectional vehicle platoon from Boo & Chwa.

A virtual leader (vehicle 0) is followed by six vehicles. Three
controllers can be simulated:

  method="dob"     Proposed Method 1: nonlinear DOB-based ISMC (Eqs. 15-23),
                   with the finite-time robust exact differentiators (FTRE,
                   Eqs. 12-13) and a smoothed sign function (Remark 5).

  method="stismc"  Proposed Method 2: adaptive dual-layer super-twisting ISMC
                   (Eqs. 30-35), with the same FTRE and the same compensating
                   input (Eq. 15) as Method 1. Only u_bar is different.

  method="linear"  Linear DOB-based ISMC benchmark, rebuilt from how the paper
                   describes ref. [9]:
                     - rear-vehicle acceleration from a 50s/(s+50) filter,
                     - no compensation of the mismatched disturbance delta_v,
                     - a first-order linear DOB that only estimates the
                       matched disturbance delta_a.

The equations are integrated with forward Euler and a fixed step dt = 1e-4 s
(the controller contains sign functions, so a small fixed step is safer than
an adaptive solver).

Array convention used everywhere below:
  * arrays with n entries hold the six followers, index 0 = vehicle 1;
  * the arrays x and v hold n+1 entries, index 0 = the leader.

Run this file directly for a quick check:   python platoon_sim.py
To make the report figures, run make_figures.py instead.
"""
import numpy as np
PARAMS = {
    # number of followers, simulation length and time step
    "n": 6, "T": 15.0, "dt": 1e-4,

    # vehicle model, Eq. (1)
    "M": 1.0,         # mass
    "c": 0.008,       # aerodynamic drag coefficient
    "f": 0.001,       # rolling resistance
    "a_max": 5.0,     # actuator acceleration limit [m/s^2]
    "v_max": 50.0,    # velocity limit [m/s]

    # virtual leader and followers initial conditions, Eq. (52)
    "x0_0": 16.0, # m
    "v0_0": 2.0,  # m/s
    "x_init": np.array([14.0, 12.0, 10.0, 7.0, 4.2, 2.4]),
    "v_init": np.array([2.3, 1.7, 1.6, 2.8, 2.4, 2.3]),

    # spacing policy, Eqs. (2)-(4)
    "xd": 0.5,        # standstill distance
    "h": 1.0,         # time headway
    "vs": 5.0,        # decay rate of the MCTH correction psi

    # sliding surfaces, Eqs. (5)-(6)
    "kI": 10.0,
    "q": 0.9,

    # FTRE differentiators, Eqs. (12)-(13)
    "la": 5.0, "lam_a0": 1.5, "lam_a1": 1.1,   # rear-vehicle acceleration
    "ld": 0.2, "lam_d0": 1.5, "lam_d1": 1.1,   # mismatched disturbance

    # nonlinear DOB, Eqs. (20)-(21)
    "ks": 10.0,
    "alpha": 10.0,
    "mu": 1.0,

    # Method 1 control law, Eq. (23), and smoothing width (Remark 5)
    "ka1": 0.01,
    "ka2": 0.01,
    "eps": 0.001,

    # Method 2: super-twisting law, Eq. (30), and dual-layer adaptation,
    # Eqs. (34)-(35). Values printed in Sec. IV of the paper.
    "kb1": 2.6,       # k_1i(t) = sqrt(Omega_i) * kb1
    "kb2": 3.3,       # k_2i(t) = Omega_i * kb2
    "tau": 0.05,      # low-pass filter time constant, Eq. (34)
    "omega": 2.0,     # safety factor (> 1) in chi_i, Eq. (35)
    "eps_b": 2.0,     # offset eps_b in chi_i, Eq. (35)
    "gamma": 5.0,     # adaptation rate of rho_i, Eq. (35)
    "Omega0": 3.0,    # initial value Omega_i(0)
    "rho0": 0.0,      # initial value rho_i(0) (not printed in the paper)
    "Omega_min": 0.1, # lower limit that keeps Omega_i > 0 (k1 uses sqrt(Omega))
    "adapt": True,    # False: fixed gains k1 = kb1, k2 = kb2 (Omega = 1), Fig. 8(b)
    "antiwindup": True,  # freeze theta, Omega, rho while the actuator saturates
                         # (our addition; without it the plain-CTH run diverges)

    # linear DOB-ISMC benchmark [9]
    "l_lin": 0.1,     # linear sliding gain
    "eta_lin": 0.2,   # switching gain
    "l0_lin": 6.0,    # DOB bandwidth
    "wf": 50.0,       # filter bandwidth of 50s/(s+50)

    # velocity measurement noise: Gaussian, clipped at +-5 % of 1 m/s,
    # and held constant for noise_hold seconds
    "noise_max": 0.05,
    "noise_hold": 1e-2,
    "seed": 1,
}

# Gain sets used in the report (each one overrides some entries of PARAMS)
PAPER = {}                                                 # gains printed in Sec. IV
TUNED_M1 = {"ks": 50.0, "alpha": 50.0, "ka1": 5.0, "ka2": 0.5}   # retuned Method 1
TUNED_LIN = {"l_lin": 5.0, "eta_lin": 0.5}                 # baseline with matching sliding gains
# Method 2 with fixed gains (no adaptation, Omega = 1), as in Fig. 8(b) of the paper
FIXED_M2 = {"adapt": False, "Omega0": 1.0}
# Method 2 with kb1 = 2*sqrt(2*kb2), the gain rule of Theorem 3
THM3_M2 = {"kb1": 2.0 * np.sqrt(2.0 * 3.3)}


def params(**changes):
    """Return a copy of PARAMS with some values changed, e.g. params(ks=50.0)."""
    p = dict(PARAMS)
    p.update(changes)
    return p


# Leader motion, disturbances and small helpers
def leader_acc(t):
    """Leader acceleration: 1 m/s^2 between t = 2 s and t = 6 s, else 0."""
    if 2.0 <= t < 6.0:
        return 1.0
    return 0.0


def disturbances(t, n):
    """Matched (d_a) and mismatched (d_v) disturbances of every follower."""
    i = np.arange(1, n + 1)                       # vehicle numbers 1..n
    bump = np.exp(-(t - 5.0 - 0.2 * i) ** 2)      # Gaussian window around t = 5 s
    d_a = 1.5 * np.sin(3 * t) * bump              # matched (acceleration equation)
    d_v = (-1.0) ** i * 0.25 * np.sin(t) * bump   # mismatched (position equation)
    return d_a, d_v


def smooth_sign(x, eps):
    """Smoothed sign function of Remark 5: x / (|x| + eps)."""
    return x / (np.abs(x) + eps)


def sqrt_sign(x):
    """|x|^(1/2) * sign(x), used by the FTRE differentiators."""
    return np.sqrt(np.abs(x)) * np.sign(x)


def shift_from_front(a):
    """Value of the vehicle in front: [0, a_1, ..., a_{n-1}] (leader gets 0)."""
    return np.append(0.0, a[:-1])


# Main simulation
def simulate(method="dob", mcth=True, noise=True, p=PARAMS):
    """
    Simulate the platoon and return a dictionary of time histories.

    method : "dob" (Method 1), "stismc" (Method 2) or "linear" (benchmark [9])
    mcth   : True uses the MCTH spacing policy; False uses plain CTH (psi = 0)
    noise  : True adds velocity measurement noise
    p      : parameter dictionary (see PARAMS and params())
    """
    n, dt, T = p["n"], p["dt"], p["T"]
    q, h, kI, xd, vs = p["q"], p["h"], p["kI"], p["xd"], p["vs"]
    M, c, f = p["M"], p["c"], p["f"]
    eps = p["eps"]
    steps = int(round(T / dt)) + 1

    # ---- noise generator
    rng = np.random.default_rng(p["seed"])
    noise_sigma = p["noise_max"] / 3.0
    hold_steps = max(1, int(round(p["noise_hold"] / dt)))
    vel_noise = np.zeros(n)

    # ---- true states (index 0 = leader, 1..n = followers)
    x = np.concatenate(([p["x0_0"]], p["x_init"]))
    v = np.concatenate(([p["v0_0"]], p["v_init"]))

    # ---- MCTH constants of Eq. (4). They make the modified error start at 0.
    if mcth:
        e0 = x[:-1] - x[1:] - xd - h * v[1:]      # initial CTH error
        kappa = vs * e0 + v[:-1] - v[1:]
    else:
        e0 = np.zeros(n)                          # plain CTH: psi is always 0
        kappa = np.zeros(n)

    # ---- controller / observer states
    int_eb = np.zeros(n)          # integral of the MCTH error (Eq. 5)

    # Method 1: FTRE (12) estimates acceleration of the rear vehicle (2..n)
    v_hat = v[2:].copy()
    a_hat = np.zeros(n - 1)
    # Method 1: FTRE (13) estimates the mismatched disturbance (1..n)
    x_hat = x[1:].copy()
    dv_hat = np.zeros(n)
    # Method 1: nonlinear DOB (20)-(21)
    S_hat = np.zeros(n)
    L_hat = np.zeros(n)

    # Method 2: super-twisting integral state theta, filtered equivalent
    # control nu_f, adaptive variable Omega and its rate rho, Eqs. (30)-(35)
    theta = np.zeros(n)
    nu_f = np.zeros(n)
    Omega = np.full(n, p["Omega0"])
    rho = np.full(n, p["rho0"])
    chi0 = np.abs(Omega - p["eps_b"])          # chi_oi = |chi_i(0)| (nu_f(0) = 0)

    # Linear benchmark: low-pass filter state (a_rear = wf * (v - v_filt))
    v_filt = v[2:].copy()
    # Linear benchmark: DOB state z, with delta_a_hat = z + l0 * v (starts at 0)
    z = -p["l0_lin"] * v[1:]

    # ---- storage for the results
    hist = {"t": np.linspace(0, T, steps), "x0": np.zeros(steps), "v0": np.zeros(steps)}
    for key in ["x", "v", "e", "eb", "eb_true", "s", "S", "u", "L", "Lhat",
                "a", "dv", "dhat", "Omega", "k1", "nuf"]:
        hist[key] = np.zeros((steps, n))
    hist["ahat"] = np.zeros((steps, n - 1))

    for k in range(steps):
        t = k * dt
        d_a, d_v = disturbances(t, n)

        # ---------------------------------------------------------------
        # 1) Measured velocities (true velocity + noise for the followers)
        # ---------------------------------------------------------------
        if noise and k % hold_steps == 0:
            vel_noise = np.clip(rng.normal(0.0, noise_sigma, n),
                                -p["noise_max"], p["noise_max"])
        v_meas = v.copy()
        if noise:
            v_meas[1:] += vel_noise

        # ---------------------------------------------------------------
        # 2) Spacing errors and sliding surfaces, Eqs. (2)-(6)
        # ---------------------------------------------------------------
        e = x[:-1] - x[1:] - xd - h * v_meas[1:]          # CTH error
        decay = np.exp(-vs * t)
        psi = (e0 + kappa * t) * decay                     # MCTH correction
        psi_dot = -vs * (e0 + kappa * t) * decay + kappa * decay
        eb = e - psi                                       # MCTH error
        s = eb + kI * int_eb                               # Eq. (5)
        s_next = np.append(s[1:], 0.0)                     # s of the rear vehicle
        S = q * s - s_next                                 # coupled surface, Eq. (6)

        # ---------------------------------------------------------------
        # 3) Controller
        # ---------------------------------------------------------------
        # Acceleration estimate of the rear vehicle (vehicles 2..n)
        if method == "linear":
            a_rear = p["wf"] * (v_meas[2:] - v_filt)
        else:
            a_rear = a_hat                                 # FTRE (Methods 1 and 2)

        # Known feedback terms A_bar, Eq. (16).
        # The rear vehicle's terms are added for vehicles 1..n-1 only.
        rel_v = v_meas[:-1] - v_meas[1:]                   # v_{i-1} - v_i
        A_bar = q * (rel_v - psi_dot) + q * kI * eb
        A_bar[:-1] += -(rel_v[1:] - psi_dot[1:]) + h * a_rear - kI * eb[1:]

        if method != "linear":
            # Mismatched-disturbance compensation B_bar (Methods 1 and 2)
            B_bar = q * (shift_from_front(dv_hat) - dv_hat)
            B_bar[:-1] -= dv_hat[:-1] - dv_hat[1:]

        if method == "dob":
            L_est = L_hat
            # Method 1 control law, Eq. (23)
            u_bar = -p["ka1"] * S - p["ka2"] * smooth_sign(S, eps) - L_est
        elif method == "stismc":
            # Method 2: adaptive gains k1 = sqrt(Omega)*kb1, k2 = Omega*kb2
            k1 = np.sqrt(Omega) * p["kb1"]
            k2 = Omega * p["kb2"]
            # Dual-layer adaptation, Eq. (35): Omega follows omega*|nu_f| + eps_b
            chi = Omega - p["omega"] * np.abs(nu_f) - p["eps_b"]
            if p["adapt"]:
                Omega_dot = -rho * np.sign(chi)
            else:
                Omega_dot = np.zeros(n)
            # phi_i = -(Omega_dot/Omega) * S_i removes the effect of the changing
            # gain from the Lyapunov analysis (Eq. 37 of the paper)
            phi = -(Omega_dot / Omega) * S
            # Super-twisting law, Eq. (30). -theta is the estimate of L_i.
            u_bar = -k1 * sqrt_sign(S) + theta + phi
            L_est = -theta
        else:
            B_bar = np.zeros(n)                            # [9] ignores delta_v
            da_hat = z + p["l0_lin"] * v_meas[1:]          # matched disturbance estimate
            L_est = -q * h * da_hat                        # only the matched part of L
            u_bar = -p["l_lin"] * S - p["eta_lin"] * smooth_sign(S, eps) - L_est

        # Control input, Eq. (15)
        u = c * v_meas[1:] ** 2 + f + M / (q * h) * (A_bar + B_bar - u_bar)

        # ---------------------------------------------------------------
        # 4) Vehicle acceleration (with actuator limit and disturbance)
        # ---------------------------------------------------------------
        acc_cmd = (u - c * v[1:] ** 2 - f) / M
        saturated = np.abs(acc_cmd) > p["a_max"]           # actuator at its limit
        acc_cmd = np.clip(acc_cmd, -p["a_max"], p["a_max"])
        acc = acc_cmd + d_a

        # ---------------------------------------------------------------
        # 5) True lumped uncertainty L, Eq. (19) (only used for plots)
        # ---------------------------------------------------------------
        if method != "linear":
            dv_err = d_v - dv_hat                          # delta_v tilde
        else:
            dv_err = d_v.copy()                            # nothing is estimated
        L = q * (shift_from_front(dv_err) - dv_err) - q * h * d_a
        L[:-1] += h * (acc[1:] - a_rear) - (dv_err[:-1] - dv_err[1:])

        # ---------------------------------------------------------------
        # 6) Save this time step
        # ---------------------------------------------------------------
        hist["x0"][k] = x[0]
        hist["v0"][k] = v[0]
        hist["x"][k] = x[1:]
        hist["v"][k] = v[1:]
        hist["e"][k] = e
        hist["eb"][k] = eb
        hist["eb_true"][k] = x[:-1] - x[1:] - xd - h * v[1:] - psi   # noise-free error
        hist["s"][k] = s
        hist["S"][k] = S
        hist["u"][k] = u
        hist["L"][k] = L
        hist["Lhat"][k] = L_est
        hist["a"][k] = acc
        hist["ahat"][k] = a_rear
        hist["dv"][k] = d_v
        hist["dhat"][k] = dv_hat
        hist["Omega"][k] = Omega
        hist["k1"][k] = np.sqrt(Omega) * p["kb1"]
        hist["nuf"][k] = nu_f

        if k == steps - 1:
            break

        # ---------------------------------------------------------------
        # 7) Update observers and vehicles one Euler step
        # ---------------------------------------------------------------
        if method != "linear":
            # FTRE (12): acceleration of the rear vehicle from its velocity
            err_v = v_hat - v_meas[2:]
            phi_a = -p["lam_a0"] * np.sqrt(p["la"]) * sqrt_sign(err_v) + a_hat
            a_hat_dot = -p["lam_a1"] * p["la"] * np.sign(a_hat - phi_a)
            v_hat += dt * phi_a
            a_hat += dt * a_hat_dot

            # FTRE (13): mismatched disturbance from position and velocity
            err_x = x_hat - x[1:]
            phi_d = -p["lam_d0"] * np.sqrt(p["ld"]) * sqrt_sign(err_x) + dv_hat
            dv_hat_dot = -p["lam_d1"] * p["ld"] * np.sign(dv_hat - phi_d)
            x_hat += dt * (v_meas[1:] + phi_d)
            dv_hat += dt * dv_hat_dot

        if method == "dob":
            # Nonlinear DOB (20)-(21)
            err_S = S - S_hat
            L_hat_dot = (-(p["ks"] + p["alpha"]) * L_hat + p["mu"] * np.sign(err_S)
                         + p["alpha"] * p["ks"] * err_S)
            S_hat += dt * (u_bar + L_hat)
            L_hat += dt * L_hat_dot
        elif method == "stismc":
            # Anti-windup: while a vehicle's actuator is saturated its integrators
            # (theta, Omega, rho) are frozen, so the gains cannot wind up.
            if p["antiwindup"]:
                run = ~saturated                       # True = integrators may run
            else:
                run = np.ones(n, dtype=bool)           # always run
            # Super-twisting integral term, Eq. (30)
            theta += dt * run * (-k2 * np.sign(S))
            # Low-pass filter for the equivalent control, Eq. (34)
            nu_f += dt * (k2 * np.sign(S) - nu_f) / p["tau"]
            # Dual-layer adaptation, Eq. (35)
            if p["adapt"]:
                # rho_i only grows while |chi_i| is larger than chi_oi
                for i in range(n):
                    if run[i] and abs(chi[i]) > chi0[i]:
                        rho[i] += dt * (p["gamma"] * abs(chi[i]))
                Omega = np.maximum(Omega + dt * run * Omega_dot, p["Omega_min"])
        else:
            # Low-pass filter for the rear-vehicle acceleration
            v_filt += dt * p["wf"] * (v_meas[2:] - v_filt)
            # Linear DOB of [9]: d(da_hat)/dt = l0 * (delta_a - da_hat)
            z += dt * (-p["l0_lin"] * (da_hat + acc_cmd))

        int_eb += dt * eb

        # Vehicle dynamics: x' = v + d_v (followers only), v' = acceleration
        acc_all = np.concatenate(([leader_acc(t)], acc))
        dv_all = np.concatenate(([0.0], d_v))
        x = x + dt * (v + dv_all)
        v = np.clip(v + dt * acc_all, -p["v_max"], p["v_max"])

    # Error of the lumped-uncertainty estimate
    hist["Lt"] = hist["L"] - hist["Lhat"]
    return hist


def downsample(hist, every=10):
    """Keep every 10th sample, so the saved result files stay small."""
    small = {}
    for key, value in hist.items():
        small[key] = value[::every]
    return small


if __name__ == "__main__":
    hist = simulate("dob")
    print("Method 1, paper gains: max |eb| =", np.abs(hist["eb"]).max())
    hist = simulate("stismc")
    print("Method 2, paper gains: max |eb| =", np.abs(hist["eb"]).max())
