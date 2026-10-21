import numpy as np
import jax.numpy as jnp
from matplotlib import rc, rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

class TrajectoryAnimator:
    """
    Figure, axes and playback shared by the *_fast animate methods of the policies.
    It reproduces the layout of the old animate methods: simulation axes on the left (a grid of sim_rows x sim_cols) and the
    action space on the right. Axes are set up once, layers (see layers.py) create their artists once and are only updated at
    each frame, and the interactive playback uses blitting, so that only the data artists are redrawn.

    Interactive controls: mouse click or space bar to pause/resume, left/right arrows to step one frame back/forward when paused.
    """
    def __init__(
        self,
        n_frames:int,
        dt:float,
        sim_rows:int=1,
        sim_cols:int=1,
        left:float=0.05,
        x_lims:jnp.ndarray=None,
        y_lims:jnp.ndarray=None,
        ylabel_pad:float=-13,
        xlabel_only_bottom_row:bool=False,
    ) -> None:
        self.n_frames = n_frames
        self.dt = dt
        self.layers = []
        # Set matplotlib fonts
        rc('font', weight='regular', size=20)
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        # Figure layout
        self.fig = plt.figure(figsize=(21.43,13.57))
        self.fig.subplots_adjust(left=left, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        outer_gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 0.4], wspace=0.09)
        if sim_rows == 1 and sim_cols == 1:
            self.sim_axs = [self.fig.add_subplot(outer_gs[0])]
        else:
            gs_left = outer_gs[0].subgridspec(sim_rows, sim_cols, wspace=0.0, hspace=0.0)
            self.sim_axs = [self.fig.add_subplot(gs_left[r, c]) for r in range(sim_rows) for c in range(sim_cols)]
        self.action_ax = self.fig.add_subplot(outer_gs[1])
        # Simulation axes
        self.x_lims = x_lims if x_lims is not None else [-10,10]
        self.y_lims = y_lims if y_lims is not None else [-10,10]
        for i, ax in enumerate(self.sim_axs):
            ax.set(xlim=self.x_lims, ylim=self.y_lims)
            if xlabel_only_bottom_row and i < len(self.sim_axs) - sim_cols:
                ax.set_xticks([])
            else:
                ax.set_xlabel('X', labelpad=-5)
            if i % 2 == 0:
                ax.set_ylabel('Y', labelpad=ylabel_pad)
            else:
                ax.set_yticks([])
            ax.set_aspect('equal', adjustable='datalim')
        # Playback state
        self._frame = 0
        self._paused = False
        self._anim = None
        self._blit = False
        self._drawn_frame = None
        self._drawn_artists = []
        self._sim_views = None

    def setup_action_space(
        self,
        v_max:float,
        wheels_distance:float,
        edgecolor:str,
        facecolor:str,
        adaptive_xticks:bool=False,
        title:str=None
    ) -> None:
        """Static part of the action space axes: labels, ticks, grid and feasible (triangular) action space."""
        ax = self.action_ax
        ax.set_xlabel("$v$ (m/s)")
        ax.set_ylabel(r"$\omega$ (rad/s)", labelpad=-15)
        ax.set_xlim(-0.1, v_max + 0.1)
        ax.set_ylim(-2*v_max/wheels_distance - 0.3, 2*v_max/wheels_distance + 0.3)
        if (not adaptive_xticks) or (v_max % 0.2 == 0):
            ax.set_xticks(jnp.arange(0, v_max+0.2, 0.2))
            ax.set_xticklabels([round(i,1) for i in jnp.arange(0, v_max, 0.2)] + [r"$\overline{v}$"])
        elif v_max % 0.1 == 0:
            ax.set_xticks(jnp.arange(0, v_max+0.1, 0.1))
            ax.set_xticklabels([round(i,1) for i in jnp.arange(0, v_max, 0.1)] + [r"$\overline{v}$"])
        else:
            ax.set_xticks(jnp.array([0.,v_max]))
            ax.set_xticklabels([round(0.0,1)] + [r"$\overline{v}$"])
        ax.set_yticks(jnp.arange(-2,3,1).tolist() + [2*v_max/wheels_distance,-2*v_max/wheels_distance])
        ax.set_yticklabels([round(i) for i in jnp.arange(-2,3,1).tolist()] + [r"$\overline{\omega}$", r"$-\overline{\omega}$"])
        ax.grid()
        ax.add_patch(
            plt.Polygon(
                [
                    [0,2*v_max/wheels_distance],
                    [0,-2*v_max/wheels_distance],
                    [v_max,0],
                ],
                closed=True,
                fill=True,
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=2,
                zorder=2,
            ),
        )
        if title is not None:
            ax.set_title(title)

    def add(self, *layers) -> None:
        self.layers.extend(layers)

    def update(self, frame:int) -> list:
        """Move all the layers to the given frame and return the artists to redraw."""
        artists = []
        for layer in self.layers:
            artists.extend(layer.update(frame))
        self._update_sim_limits(frame)
        return artists

    def _update_sim_limits(self, frame:int) -> None:
        """
        Reproduce what the old animate methods did at each frame on the simulation axes: ax.clear() and ax.set(xlim, ylim),
        so the data limits only include the data of the current frame and the view is reset to the given limits.
        Then, at draw time, set_aspect('equal', adjustable='datalim') expands one of the two ranges, depending on the data limits.
        """
        for ax in self.sim_axs:
            points = [p for layer in self.layers if layer.ax is ax and (p := layer.datalim_points(frame)) is not None]
            ax.dataLim.set_points(Bbox.null().get_points())
            ax.ignore_existing_data_limits = True
            if len(points) > 0:
                ax.update_datalim(np.concatenate(points))
            ax.set(xlim=self.x_lims, ylim=self.y_lims)

    def render_frame(self, frame:int) -> np.ndarray:
        """Full (non-blitted) draw of one frame. Returns the RGBA buffer of the figure (used to compare with the old animations)."""
        self.update(frame)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba()).copy()

    def run(self, save_video:bool=False, save_path:str=None, dpi:int=300, speed:float=1., blit:bool=True) -> None:
        """Save the video (if requested) and then show the animation, looping, in real time (or speed times faster)."""
        self._anim = FuncAnimation(
            self.fig,
            self._draw_frame,
            frames=self._frames,
            init_func=lambda: self.update(self._frame),
            interval=self.dt*1000/speed,
            blit=blit,
            save_count=self.n_frames,
            cache_frame_data=False,
        )
        if save_video:
            if not writers.is_available('ffmpeg'):
                raise RuntimeError("ffmpeg is needed to save the video, but it is not installed")
            writer_video = FFMpegWriter(fps=int(1/self.dt), bitrate=1800)
            self._anim.save(save_path, writer=writer_video, dpi=dpi)
            self._frame = 0
        if blit:
            # Blitting restores the background only inside the axes: animated texts must not exceed them, or they leave trails
            for artist in self.update(self._frame):
                if isinstance(artist, Text):
                    artist.set_clip_on(True)
        self._blit = blit
        self.fig.canvas.mpl_connect('button_press_event', self._toggle_pause)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show()

    def _frames(self):
        while True:
            yield self._frame
            if not self._paused:
                self._frame = (self._frame + 1) % self.n_frames

    def _draw_frame(self, frame:int) -> list:
        if frame == self._drawn_frame:
            # Paused: leave the frame untouched (so that, e.g., zooming while paused works)
            return self._drawn_artists
        self._drawn_frame = frame
        self._drawn_artists = self.update(frame)
        if self._blit:
            # Blitting only redraws the data artists over the cached background: if the axes limits of the new frame changed
            # (see _update_sim_limits), the background (ticks) must be redrawn (FuncAnimation re-caches it when the view changes)
            for ax in self.sim_axs:
                ax.apply_aspect()
            sim_views = [ax.viewLim.frozen().get_points() for ax in self.sim_axs]
            if self._sim_views is not None and any(not np.array_equal(v1, v2) for v1, v2 in zip(sim_views, self._sim_views)):
                self.fig.canvas.draw()
            self._sim_views = sim_views
        return self._drawn_artists

    def _toggle_pause(self, *args, **kwargs) -> None:
        self._paused = not self._paused

    def _on_key(self, event) -> None:
        if event.key == ' ':
            self._toggle_pause()
        elif self._paused and event.key in ('left', 'right'):
            self._frame = (self._frame + (1 if event.key == 'right' else -1)) % self.n_frames
