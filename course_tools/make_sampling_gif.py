"""Render assets/bayesian_sampling.gif: a Metropolis-Hastings walk on a 2D posterior.

Left panel shows the target density with the chain walking over it and each
proposal marked accepted/rejected. Right panels show the trace of theta_1 and
its marginal histogram filling in towards the analytic marginal.

Run with: python course_tools/make_sampling_gif.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

# --- palette (validated categorical slots 1-2 + single-hue sequential) --------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
ACCEPT = "#2a78d6"  # categorical slot 1
REJECT = "#eb6834"  # categorical slot 2
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#c9dcf2", "#7fabdf", "#2a78d6", "#123a6b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": "#d8d7d2",
    "font.size": 9,
})


# --- target posterior: a banana-shaped (non-Gaussian, correlated) density -----
def log_post(x1, x2):
    return -(x1**2) / (2 * 1.6**2) - ((x2 - 0.5 * x1**2 + 1.0) ** 2) / (2 * 0.55**2)


X1LIM, X2LIM = (-4.2, 4.2), (-3.0, 6.0)
g1 = np.linspace(*X1LIM, 320)
g2 = np.linspace(*X2LIM, 320)
G1, G2 = np.meshgrid(g1, g2)
DENS = np.exp(log_post(G1, G2))

# analytic marginal of theta_1 by numerical integration over theta_2
MARG = DENS.sum(axis=0)
MARG /= np.trapezoid(MARG, g1)


# --- run the chain ------------------------------------------------------------
def run_chain(n, step=0.6, seed=7):
    rng = np.random.default_rng(seed)
    x = np.array([-3.2, 4.6])  # start away from the mode
    lp = log_post(*x)
    steps = []
    for _ in range(n):
        prop = x + step * rng.standard_normal(2)
        lp_prop = log_post(*prop)
        accepted = np.log(rng.uniform()) < lp_prop - lp
        steps.append((x.copy(), prop.copy(), accepted))
        if accepted:
            x, lp = prop, lp_prop
    return steps


# frame schedule: one proposal per frame at first, then batches
SLOW, FAST_FRAMES, BATCH = 55, 145, 22
SCHEDULE = [1] * SLOW + [BATCH] * FAST_FRAMES
N_ITER = sum(SCHEDULE)
STEPS = run_chain(N_ITER)
ENDS = np.cumsum(SCHEDULE)

# chain state after every iteration, for cheap slicing per frame
CHAIN = np.array([s[0] for s in STEPS] + [STEPS[-1][1] if STEPS[-1][2] else STEPS[-1][0]])
ACC_FLAGS = np.array([s[2] for s in STEPS])


# --- figure -------------------------------------------------------------------
fig = plt.figure(figsize=(9.2, 4.3), dpi=80)
gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1], height_ratios=[1, 1],
                      left=0.025, right=0.98, top=0.90, bottom=0.03,
                      wspace=0.10, hspace=0.28)
ax_p = fig.add_subplot(gs[:, 0])
ax_t = fig.add_subplot(gs[0, 1])
ax_h = fig.add_subplot(gs[1, 1])

fig.text(0.028, 0.975, "Sampling a posterior with Metropolis-Hastings",
         fontsize=13, color=INK, weight="semibold", ha="left", va="top")

# posterior panel
ax_p.contourf(G1, G2, DENS, levels=12, cmap=SEQ, alpha=0.9)
ax_p.contour(G1, G2, DENS, levels=6, colors="#ffffff", linewidths=0.6, alpha=0.5)
ax_p.set_xlim(*X1LIM)
ax_p.set_ylim(*X2LIM)
ax_p.text(0.98, 0.98, "posterior  $p(\\theta_1,\\theta_2\\,|\\,d)$", transform=ax_p.transAxes,
          ha="right", va="top", color=INK_2, fontsize=9)

(path_old,) = ax_p.plot([], [], lw=1.0, color="#ffffff", alpha=0.55, zorder=3)
(path_new,) = ax_p.plot([], [], lw=2.0, color="#ffffff", alpha=0.95, zorder=4)
(pts_acc,) = ax_p.plot([], [], "o", ms=2.6, color=ACCEPT, mec=SURFACE, mew=0.4, alpha=0.85, zorder=5)
(prop_line,) = ax_p.plot([], [], lw=2.0, zorder=6)
(prop_pt,) = ax_p.plot([], [], "o", ms=8, mec=SURFACE, mew=2.0, zorder=7)
(cur_pt,) = ax_p.plot([], [], "o", ms=9, color=ACCEPT, mec=SURFACE, mew=2.0, zorder=8)
verdict = ax_p.text(0.98, 0.035, "", transform=ax_p.transAxes, ha="right", va="bottom",
                    fontsize=10, weight="semibold")
counter = ax_p.text(0.025, 0.03, "", transform=ax_p.transAxes, ha="left", va="bottom",
                    fontsize=9, color=INK, bbox=dict(fc=SURFACE, ec="#d8d7d2", pad=3.5))

# trace panel
ax_t.set_xlim(0, N_ITER)
ax_t.set_ylim(*X1LIM)
ax_t.text(0.0, 1.04, r"trace of $\theta_1$", transform=ax_t.transAxes,
          ha="left", va="bottom", color=INK_2, fontsize=9)
(trace_line,) = ax_t.plot([], [], lw=1.0, color=ACCEPT)

# marginal panel
ax_h.set_xlim(*X1LIM)
ax_h.set_ylim(0, MARG.max() * 1.35)
ax_h.text(0.0, 1.04, r"marginal $p(\theta_1\,|\,d)$", transform=ax_h.transAxes,
          ha="left", va="bottom", color=INK_2, fontsize=9)
ax_h.plot(g1, MARG, lw=2.0, color=INK, zorder=4)
ax_h.text(0.975, 0.9, "exact", transform=ax_h.transAxes, ha="right", va="top",
          fontsize=8.5, color=INK)
ax_h.text(0.975, 0.74, "samples", transform=ax_h.transAxes, ha="right", va="top",
          fontsize=8.5, color=ACCEPT)
BINS = np.linspace(*X1LIM, 34)
bars = ax_h.bar(BINS[:-1], np.zeros(len(BINS) - 1), width=np.diff(BINS) * 0.88,
                align="edge", color=ACCEPT, alpha=0.55, zorder=3)
for ax in (ax_p, ax_t, ax_h):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def update(frame):
    i = ENDS[frame] - 1  # index of the last proposal shown this frame
    cur, prop, ok = STEPS[i]
    chain = CHAIN[: i + 2]

    tail = max(0, len(chain) - 45)
    path_old.set_data(chain[: tail + 1, 0], chain[: tail + 1, 1])
    path_new.set_data(chain[tail:, 0], chain[tail:, 1])
    pts_acc.set_data(chain[:, 0], chain[:, 1])

    show_prop = SCHEDULE[frame] == 1
    col = ACCEPT if ok else REJECT
    prop_line.set_data(*(([cur[0], prop[0]], [cur[1], prop[1]]) if show_prop else ([], [])))
    prop_line.set_color(col)
    prop_line.set_linestyle("-" if ok else (0, (2, 1.6)))
    prop_pt.set_data(*(([prop[0]], [prop[1]]) if show_prop else ([], [])))
    prop_pt.set_color(col)
    cur_pt.set_data([chain[-1, 0]], [chain[-1, 1]])
    verdict.set_text(("proposal accepted" if ok else "proposal rejected") if show_prop else "")
    verdict.set_color(col)

    n_acc = int(ACC_FLAGS[: i + 1].sum())
    counter.set_text(f"{i + 1} proposals   {n_acc} accepted   ({n_acc / (i + 1):.0%})")

    trace_line.set_data(np.arange(len(chain)), chain[:, 0])

    # normalised by the *final* chain length, so the bars fill in over time and
    # land on the exact marginal at the last frame instead of spiking early
    h, _ = np.histogram(chain[:, 0], bins=BINS)
    h = h / ((N_ITER + 1) * np.diff(BINS))
    for bar, ht in zip(bars, h):
        bar.set_height(ht)

    return ()


anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=1)
out = "assets/bayesian_sampling.gif"
anim.save(out, writer=PillowWriter(fps=18), dpi=100)
print("wrote", out)
