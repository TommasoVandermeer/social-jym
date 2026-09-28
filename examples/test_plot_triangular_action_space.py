import matplotlib.pyplot as plt
import os
from matplotlib import rc, rcParams
font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

figure, ax = plt.subplots(1,1, figsize=(10, 3))
figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.2)
ax.add_patch(
    plt.Polygon(
        [   
            [2*1./0.7,0],
            [-2*1./0.7,0],
            [0.,1.],
        ],
        closed=True,
        fill=True,
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=2,
        zorder=2,
    ),
)
ax.set_xticks([2*1./0.7, 0., -2*1./0.7])
ax.set_xticklabels([r"$\overline{\omega}$", "0", r"$-\overline{\omega}$"])
ax.set_yticks([0.,1.])
ax.set_yticklabels(["0", r"$\overline{v}$"])
ax.set_ylim(-0.1, 1. + 0.1)
ax.set_xlim(-2*1./0.7 - 0.3, 2*1./0.7 + 0.3)
ax.set_ylabel("$v$ (m/s)", labelpad=-15)
ax.set_xlabel("$\omega$ (rad/s)", labelpad=-5)
ax.grid()
figure.savefig(os.path.join(os.path.dirname(__file__), 'action_space.eps'), format='eps')