"""
Reusable pieces of the policies animations.
Each layer creates its artists ONCE (with the same style and in the same order as the old animate methods) and update(frame)
only changes their data with numpy setters, returning the artists to redraw. All the data is expected to be numpy arrays with
the frame index as first dimension (see preprocessing.py).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection, EllipseCollection, PolyCollection

from socialjym.utils.animation.preprocessing import heading_points

class Layer:
    """
    Base class. update(frame) returns ALL the artists of the layer, also the ones that do not change: when blitting, returned
    artists are drawn above the static background, so static data (e.g. obstacles) must be redrawn to keep the original zorder.
    datalim_points(frame) returns the points (n, 2) that the artists of the old animate methods added to the axes data limits
    at that frame (the old methods cleared the axes at each frame, and with aspect='equal' and adjustable='datalim' the data 
    limits decide which axis range is expanded, so the axes limits of the old animations change from frame to frame).
    """
    def __init__(self, ax) -> None:
        self.ax = ax
        self.artists = []

    def update(self, frame:int) -> list:
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return None

class LineLayer(Layer):
    """A Line2D (single marker if points is (n_steps, 2), polyline if points is (n_steps, n_points, 2)) moving at each frame."""
    def __init__(self, ax, points:np.ndarray, **kwargs) -> None:
        super().__init__(ax)
        self.points = points[:,None,:] if points.ndim == 2 else points
        self.line, = ax.plot(self.points[0,:,0], self.points[0,:,1], **kwargs)
        self.artists = [self.line]

    def update(self, frame:int) -> list:
        self.line.set_data(self.points[frame,:,0], self.points[frame,:,1])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return self.points[frame]

class ScatterLayer(Layer):
    """
    A scatter whose points change at each frame. If mask (n_steps, n_points) is given, only the masked points are shown.
    If values (n_steps, n_points) are given they are colormapped and the color limits are rescaled at each frame to the shown
    values, as plt.scatter(c=values) does.
    """
    def __init__(self, ax, offsets:np.ndarray, mask:np.ndarray=None, values:np.ndarray=None, **kwargs) -> None:
        super().__init__(ax)
        self.offsets = offsets
        self.mask = mask
        self.values = values
        if values is not None:
            kwargs['c'] = values[0]
        self.scatter = ax.scatter(offsets[0,:,0], offsets[0,:,1], **kwargs)
        self.artists = [self.scatter]
        self.update(0)

    def update(self, frame:int) -> list:
        offsets = self.offsets[frame]
        values = self.values[frame] if self.values is not None else None
        if self.mask is not None:
            offsets = offsets[self.mask[frame]]
            values = values[self.mask[frame]] if values is not None else None
        self.scatter.set_offsets(offsets)
        if values is not None:
            self.scatter.set_array(values)
            shown_values = self.scatter.get_array() # Masked where invalid, as in plt.scatter
            if shown_values.count() > 0:
                self.scatter.set_clim(shown_values.min(), shown_values.max())
        self.scatter.set_visible(len(offsets) > 0)
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return self.offsets[frame] if self.mask is None else self.offsets[frame][self.mask[frame]]

class PolygonLayer(Layer):
    """A polygon whose vertices (n_steps, n_vertices, 2) change at each frame."""
    def __init__(self, ax, vertices:np.ndarray, **kwargs) -> None:
        super().__init__(ax)
        self.vertices = vertices
        self.polygon = plt.Polygon(vertices[0], closed=True, **kwargs)
        ax.add_patch(self.polygon)
        self.artists = [self.polygon]

    def update(self, frame:int) -> list:
        self.polygon.set_xy(self.vertices[frame])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return self.vertices[frame]

class LinesLayer(Layer):
    """
    Many polylines (n_steps, n_lines, n_points, 2) drawn as one LineCollection (e.g. LiDAR rays, sampled trajectories).
    Optional per-frame mask (n_steps, n_lines), RGBA colors (n_steps, n_lines, 4) and linewidths (n_steps, n_lines).
    """
    def __init__(self, ax, segments:np.ndarray, mask:np.ndarray=None, colors:np.ndarray=None, linewidths:np.ndarray=None, **kwargs) -> None:
        super().__init__(ax)
        self.segments = segments
        self.mask = mask
        self.colors = colors
        self.linewidths = linewidths
        # Same cap and join styles of Line2D, which was used for each line in the old animations
        self.collection = LineCollection([], capstyle='projecting', joinstyle='round', **kwargs)
        ax.add_collection(self.collection, autolim=False)
        self.artists = [self.collection]
        self.update(0)

    def update(self, frame:int) -> list:
        lines = slice(None) if self.mask is None else self.mask[frame]
        self.collection.set_segments(self.segments[frame][lines])
        if self.colors is not None:
            self.collection.set_color(self.colors[frame][lines])
        if self.linewidths is not None:
            self.collection.set_linewidth(self.linewidths[frame][lines])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        lines = slice(None) if self.mask is None else self.mask[frame]
        return self.segments[frame][lines].reshape(-1, 2)

class EllipsesLayer(Layer):
    """Ellipses with centers (n_steps, n, 2), widths, heights and angles in degrees (n_steps, n), drawn as one EllipseCollection."""
    def __init__(self, ax, centers:np.ndarray, widths:np.ndarray, heights:np.ndarray, angles:np.ndarray, mask:np.ndarray=None, **kwargs) -> None:
        super().__init__(ax)
        self.centers, self.widths, self.heights, self.angles = centers, widths, heights, angles
        self.mask = mask
        self.collection = EllipseCollection(widths[0], heights[0], angles[0], units='xy', offsets=centers[0], offset_transform=ax.transData, **kwargs)
        ax.add_collection(self.collection, autolim=False)
        self.artists = [self.collection]
        self.update(0)

    def update(self, frame:int) -> list:
        shown = np.ones(self.widths.shape[1], dtype=bool) if self.mask is None else self.mask[frame]
        self.collection.set_offsets(self.centers[frame][shown])
        self.collection.set_widths(self.widths[frame][shown])
        self.collection.set_heights(self.heights[frame][shown])
        self.collection.set_angles(self.angles[frame][shown])
        self.collection.set_visible(np.any(shown))
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        # Axis-aligned bounding box of each shown ellipse
        shown = np.ones(self.widths.shape[1], dtype=bool) if self.mask is None else self.mask[frame]
        centers = self.centers[frame][shown]
        angles = np.radians(self.angles[frame][shown])
        half_widths, half_heights = self.widths[frame][shown] / 2, self.heights[frame][shown] / 2
        extents = np.stack((
            np.sqrt((half_widths * np.cos(angles))**2 + (half_heights * np.sin(angles))**2),
            np.sqrt((half_widths * np.sin(angles))**2 + (half_heights * np.cos(angles))**2),
        ), axis=-1)
        return np.concatenate((centers - extents, centers + extents))

class ArrowsLayer(Layer):
    """One FancyArrow per object, from origins (n_steps, n, 2) along deltas (n_steps, n, 2). Masked-out arrows are hidden."""
    def __init__(self, ax, origins:np.ndarray, deltas:np.ndarray, mask:np.ndarray=None, **kwargs) -> None:
        super().__init__(ax)
        self.origins, self.deltas, self.mask = origins, deltas, mask
        self.artists = [ax.arrow(origins[0,i,0], origins[0,i,1], deltas[0,i,0], deltas[0,i,1], **kwargs) for i in range(origins.shape[1])]
        self.update(0)

    def update(self, frame:int) -> list:
        for i, arrow in enumerate(self.artists):
            arrow.set_data(x=self.origins[frame,i,0], y=self.origins[frame,i,1], dx=self.deltas[frame,i,0], dy=self.deltas[frame,i,1])
            if self.mask is not None:
                arrow.set_visible(self.mask[frame,i])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        shown_arrows = [arrow.get_xy() for arrow in self.artists if arrow.get_visible()]
        return np.concatenate(shown_arrows) if shown_arrows else None

class TextsLayer(Layer):
    """One Text per object, with positions (n_steps, n, 2) and strings (n_steps, n). Masked-out texts are hidden."""
    def __init__(self, ax, positions:np.ndarray, strings:np.ndarray, mask:np.ndarray=None, **kwargs) -> None:
        super().__init__(ax)
        self.positions, self.strings, self.mask = positions, strings, mask
        self.artists = [ax.text(positions[0,i,0], positions[0,i,1], strings[0][i], **kwargs) for i in range(positions.shape[1])]
        self.update(0)

    def update(self, frame:int) -> list:
        for i, text in enumerate(self.artists):
            text.set_position(self.positions[frame,i])
            text.set_text(self.strings[frame][i])
            if self.mask is not None:
                text.set_visible(self.mask[frame,i])
        return self.artists

class RobotLayer(Layer):
    """Robot body and heading (and goal, if given)."""
    def __init__(self, ax, robot_poses:np.ndarray, robot_radius:float, robot_goals:np.ndarray=None) -> None:
        super().__init__(ax)
        self.robot_poses = robot_poses
        self.heads = heading_points(robot_poses, robot_radius)
        self.robot_goals = robot_goals
        self.head = plt.Circle(self.heads[0], 0.1, color='black', zorder=1)
        ax.add_patch(self.head)
        self.body = plt.Circle(robot_poses[0,:2], robot_radius, edgecolor="black", facecolor="red", fill=True, zorder=3)
        ax.add_patch(self.body)
        self.artists = [self.head, self.body]
        if robot_goals is not None:
            self.goal, = ax.plot(robot_goals[0,0], robot_goals[0,1], marker='*', markersize=7, color='red', zorder=5)
            self.artists.append(self.goal)

    def update(self, frame:int) -> list:
        self.head.set_center(self.heads[frame])
        self.body.set_center(self.robot_poses[frame,:2])
        if self.robot_goals is not None:
            self.goal.set_data([self.robot_goals[frame,0]], [self.robot_goals[frame,1]])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        points = [*circle_extents(self.heads[frame][None], 0.1), *circle_extents(self.robot_poses[frame,None,:2], self.body.get_radius())]
        if self.robot_goals is not None:
            points.append(self.robot_goals[frame,None])
        return np.concatenate(points)

class HumansLayer(Layer):
    """
    Humans bodies, headings, legs (optional) and velocity arrows (optional).
    If visibility (n_steps, n_humans) is given, non visible humans are grey and more transparent.
    Static humans (n_humans,) are drawn as black disks without heading nor velocity (columns in CCSO).
    """
    def __init__(
        self,
        ax,
        humans_poses:np.ndarray,
        humans_radii:np.ndarray,
        humans_velocities:np.ndarray=None,
        visibility:np.ndarray=None,
        static:np.ndarray=None,
        legs_states:np.ndarray=None,
        legs_radii:np.ndarray=None,
    ) -> None:
        super().__init__(ax)
        n_humans = humans_poses.shape[1]
        self.poses = humans_poses
        self.radii = humans_radii
        self.heads = heading_points(humans_poses, humans_radii)
        self.velocities = humans_velocities
        self.visibility = visibility
        self.static = np.zeros(n_humans, dtype=bool) if static is None else static
        self.legs = (legs_states is not None) and (legs_radii is not None)
        self.legs_states, self.legs_radii = legs_states, legs_radii
        self.moving_humans = np.where(~self.static)[0]
        # Create artists in the same order of the old animate methods (it matters for overlapping artists with same zorder)
        self.static_bodies, self.parts, self.arrows = {}, {}, {}
        for h in range(n_humans):
            if self.static[h]:
                self.static_bodies[h] = plt.Circle(humans_poses[0,h,:2], humans_radii[0,h], edgecolor='black', facecolor='black', alpha=1., fill=True, zorder=1)
                ax.add_patch(self.static_bodies[h])
                self.artists.append(self.static_bodies[h])
            else:
                parts = {}
                if self.legs:
                    parts['l_leg'] = plt.Circle(legs_states[0,h,0:2], legs_radii[0,h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                    parts['r_leg'] = plt.Circle(legs_states[0,h,3:5], legs_radii[0,h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                parts['head'] = plt.Circle(self.heads[0,h], 0.1, color='black', alpha=0.6, zorder=1)
                parts['body'] = plt.Circle(humans_poses[0,h,:2], humans_radii[0,h], edgecolor='black', facecolor='blue', alpha=0.6, fill=True, zorder=1)
                for part in parts.values():
                    ax.add_patch(part)
                    self.artists.append(part)
                self.parts[h] = parts
        if humans_velocities is not None:
            for h in self.moving_humans:
                self.arrows[h] = ax.arrow(
                    humans_poses[0,h,0],
                    humans_poses[0,h,1],
                    humans_velocities[0,h,0],
                    humans_velocities[0,h,1],
                    head_width=0.15,
                    head_length=0.15,
                    fc='blue',
                    ec='blue',
                    alpha=0.6,
                    zorder=30,
                )
                self.artists.append(self.arrows[h])
        self.update(0)

    def update(self, frame:int) -> list:
        for h, body in self.static_bodies.items():
            body.set_center(self.poses[frame,h,:2])
            body.set_radius(self.radii[frame,h])
        for h, parts in self.parts.items():
            visible = True if self.visibility is None else bool(self.visibility[frame,h])
            color = 'blue' if visible else 'grey'
            alpha = 0.6 if visible else 0.3
            if self.legs:
                for leg, idxs in (('l_leg', slice(0,2)), ('r_leg', slice(3,5))):
                    parts[leg].set_center(self.legs_states[frame,h,idxs])
                    parts[leg].set_radius(self.legs_radii[frame,h])
                    parts[leg].set_facecolor(color)
                    parts[leg].set_alpha(alpha)
            body_alpha = 0.3 if self.legs else alpha
            parts['head'].set_center(self.heads[frame,h])
            parts['head'].set_alpha(body_alpha)
            parts['body'].set_center(self.poses[frame,h,:2])
            parts['body'].set_radius(self.radii[frame,h])
            parts['body'].set_facecolor(color)
            parts['body'].set_alpha(body_alpha)
            if h in self.arrows:
                self.arrows[h].set_data(x=self.poses[frame,h,0], y=self.poses[frame,h,1], dx=self.velocities[frame,h,0], dy=self.velocities[frame,h,1])
                self.arrows[h].set_facecolor(color)
                self.arrows[h].set_edgecolor(color)
                self.arrows[h].set_alpha(alpha)
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        static, moving = np.where(self.static)[0], self.moving_humans
        points = [
            *circle_extents(self.poses[frame,static,:2], self.radii[frame,static]),
            *circle_extents(self.heads[frame,moving], 0.1),
            *circle_extents(self.poses[frame,moving,:2], self.radii[frame,moving]),
        ]
        if self.legs:
            points += [
                *circle_extents(self.legs_states[frame,moving,0:2], self.legs_radii[frame,moving]),
                *circle_extents(self.legs_states[frame,moving,3:5], self.legs_radii[frame,moving]),
            ]
        points += [arrow.get_xy() for arrow in self.arrows.values()]
        return np.concatenate(points)

class ObstaclesLayer(Layer):
    """Static obstacles (n_steps, n_obstacles, n_edges, 2, 2): filled polygons, or segments if they have one edge."""
    def __init__(self, ax, static_obstacles:np.ndarray) -> None:
        super().__init__(ax)
        self.obstacles = static_obstacles
        self.constant = all(np.array_equal(static_obstacles[0], o, equal_nan=True) for o in static_obstacles)
        self.polygons = static_obstacles.shape[2] > 1
        for o in static_obstacles[0]:
            if self.polygons: # ax.fill draws one polygon per column (edges starting and ending points), as in the old animations
                self.artists += ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
            else:
                self.artists += ax.plot(o[0,:,0],o[0,:,1], color='black', linewidth=2, zorder=3)

    def update(self, frame:int) -> list:
        if not self.constant:
            for i, o in enumerate(self.obstacles[frame]):
                if self.polygons:
                    self.artists[2*i].set_xy(o[:,0,:])
                    self.artists[2*i+1].set_xy(o[:,1,:])
                else:
                    self.artists[i].set_data(o[0,:,0], o[0,:,1])
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return self.obstacles[frame].reshape(-1, 2)

class GridCellsLayer(Layer):
    """Square grid cells (n_rows, n_cols, 2) of the given size, red where occupied (n_steps, n_rows, n_cols)."""
    def __init__(self, ax, grid_cells:np.ndarray, grid_cells_size:float, occupancy_grids:np.ndarray=None) -> None:
        super().__init__(ax)
        self.occupancy_grids = occupancy_grids
        centers = grid_cells.reshape(-1, 2)
        corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) * grid_cells_size / 2
        self.cells_corners = centers[:,None,:] + corners[None]
        self.collection = PolyCollection(self.cells_corners, facecolors='white', edgecolors='black', alpha=0.5, zorder=0)
        ax.add_collection(self.collection, autolim=False)
        self.artists = [self.collection]
        self.update(0)

    def update(self, frame:int) -> list:
        if self.occupancy_grids is not None:
            occupied = np.asarray(self.occupancy_grids[frame]).reshape(-1).astype(bool)
            self.collection.set_facecolor(np.where(occupied, 'red', 'white'))
        return self.artists

    def datalim_points(self, frame:int) -> np.ndarray:
        return self.cells_corners.reshape(-1, 2)

def scene_layers(
    ax,
    robot_poses:np.ndarray,
    robot_radius:float,
    robot_goals:np.ndarray=None,
    humans_poses:np.ndarray=None,
    humans_velocities:np.ndarray=None,
    humans_radii:np.ndarray=None,
    static_obstacles:np.ndarray=None,
    **humans_kwargs,
) -> list:
    """Humans, robot, goal and static obstacles of one simulation axes (in the drawing order of the old animate methods)."""
    layers = []
    if humans_poses is not None:
        layers.append(HumansLayer(ax, humans_poses, humans_radii, humans_velocities, **humans_kwargs))
    layers.append(RobotLayer(ax, robot_poses, robot_radius, robot_goals))
    if static_obstacles is not None:
        layers.append(ObstaclesLayer(ax, static_obstacles))
    return layers

def point_cloud_layers(ax, point_clouds:np.ndarray, n_stack:int) -> list:
    """Stack of point clouds (n_steps, n_clouds, n_points, 2): older clouds (higher index) are more faded."""
    layers = []
    n_points = point_clouds.shape[2]
    for j in range(point_clouds.shape[1]):
        t = (1 - j / (n_stack - 1)) if n_stack > 1 else 1.  # in [0,1]
        layers.append(ScatterLayer(
            ax,
            point_clouds[:,j],
            c=np.full((n_points,), 0.3 + 0.7 * t),
            cmap='Reds',
            vmin=0.0,
            vmax=1.0,
            alpha=0.3 + 0.7 * t,
            zorder=20 + n_stack - j,
        ))
    return layers

def perception_points_layers(ax, positions:np.ndarray, velocities:np.ndarray, mask:np.ndarray) -> list:
    """Detected humans (world frame positions and velocities (n_steps, n_humans, 2)) as red dots with velocity arrows."""
    layers = []
    for h in range(positions.shape[1]):
        layers.append(ScatterLayer(ax, positions[:,h:h+1], mask=mask[:,h:h+1], color='red', s=30, alpha=1, zorder=20))
        layers.append(ArrowsLayer(ax, positions[:,h:h+1], velocities[:,h:h+1], mask=mask[:,h:h+1], head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=1, zorder=20))
    return layers

def circle_extents(centers:np.ndarray, radii) -> tuple:
    """Lower-left and upper-right corners of the bounding boxes of circles with centers (n, 2) and radii (n) or scalar."""
    radii = np.broadcast_to(np.asarray(radii, dtype=float), centers.shape[:1])[:,None]
    return centers - radii, centers + radii

def rgba(color, alpha:float=None) -> np.ndarray:
    return np.array(mcolors.to_rgba(color, alpha))
