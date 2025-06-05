import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax, jit, vmap, debug
from functools import partial
import time
from shapely.geometry import LineString, box
from shapely.affinity import rotate, translate

vmax = 1.
wheels_distance = 0.7
samples = 205
dt = .25
robot_radius = 0.3
robot_pose = jnp.array([1., 1., jnp.pi/4.]) # Robot pose in the form [x, y, theta]
iterations = 1
epsilon = 1e-3
# Obstacles
obstacles = jnp.array([
    [[[0.47,0.015],[0.57,0.015]],[[0.57,0.015],[0.57,0.1]],[[0.57,0.1],[0.47,0.015]]],
    [[[0.45,-0.015],[0.55,0.]],[[0.55,0.],[0.6,-0.1]],[[0.6,-0.1],[0.45,-0.015]]],
    # [[[0.08,-0.305],[0.07,-0.35]],[[0.07,-0.35],[0.13,-0.4]],[[0.13,-0.4],[0.08,-0.305]]],
    [[[0.57,0.17],[0.35,0.35]],[[0.35,0.35],[0.4,0.4]],[[0.4,0.4],[0.57,0.17]]],
    [[[-0.25,0.2],[-0.25,0.35]],[[-0.25,0.35],[-0.29,0.35]],[[-0.29,0.35],[-0.25,0.2]]],
    [[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]],[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]],[[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]]],
])
obstacles = obstacles.at[:, :, :, 0].set(obstacles[:, :, :, 0] + 1)
obstacles = obstacles.at[:, :, :, 1].set(obstacles[:, :, :, 1] + 1)
segments = jnp.reshape(obstacles, (obstacles.shape[0] * obstacles.shape[1], 2, 2))
# Filter out segments containing NaN values before creating shapely LineStrings
valid_mask = ~jnp.isnan(segments).any(axis=(1,2))
filtered_segments = segments[valid_mask]
shapely_segments = [LineString(segment) for segment in filtered_segments]
print("N° of obstacles: ", obstacles.shape[0])
print("N° of segments per obstacle: ", obstacles.shape[1], "\n")

### JAX BASED ALGORITHM (Liang-Barsky fline clipping algorithm)
# Segment-rectangle intersection function
@jit
def segment_rectangle_intersection(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
    @jit
    def _nan_segment(val):
        return False, jnp.array([jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan])
    @jit
    def _not_nan_segment(val):
        x1, y1, x2, y2, xmin, xmax, ymin, ymax = val
        dx = x2 - x1
        dy = y2 - y1
        p = jnp.array([-dx, dx, -dy, dy])
        q = jnp.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1])
        @jit
        def loop_body(i, tup):
            t, p, q = tup
            t0, t1 = t
            t0, t1 = lax.switch(
                (jnp.sign(p[i])+1).astype(jnp.int32),
                [
                    lambda t: lax.cond(q[i]/p[i] > t[1], lambda _: (2.,1.), lambda x: (jnp.max(jnp.array([x[0],q[i]/p[i]])), x[1]), t),  # p[i] < 0
                    lambda t: lax.cond(q[i] < 0, lambda _: (2.,1.), lambda x: x, t),  # p[i] == 0
                    lambda t: lax.cond(q[i]/p[i] < t[0], lambda _: (2.,1.), lambda x: (x[0], jnp.min(jnp.array([x[1],q[i]/p[i]]))), t),  # p[i] > 0
                ],
                (t0, t1),
            )
            # debug.print("t0: {x}, t1: {y}, switch_case: {z}", x=t0, y=t1, z=(jnp.sign(p[i])+1).astype(jnp.int32))
            return ((t0, t1), p ,q)
        t, p, q = lax.fori_loop(
            0, 
            4,
            loop_body,
            ((0., 1.), p, q),
        )
        t0, t1 = t
        inside_or_intersects = ~(t0 > t1)
        intersection_point_0 = lax.switch(
            jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t0 == 0), (inside_or_intersects) & (t0 > 0)])),
            [
                lambda _: jnp.array([jnp.nan, jnp.nan]),
                lambda _: jnp.array([x1, y1]),
                lambda _: jnp.array([x1 + t0 * dx, y1 + t0 * dy]),
            ],
            None,
        )
        intersection_point_1 = lax.switch(
            jnp.argmax(jnp.array([~(inside_or_intersects), (inside_or_intersects) & (t1 == 1), (inside_or_intersects) & (t1 < 1)])),
            [
                lambda _: jnp.array([jnp.nan, jnp.nan]),
                lambda _: jnp.array([x2, y2]),
                lambda _: jnp.array([x1 + t1 * dx, y1 + t1 * dy]),
            ],
            None,
        )
        return inside_or_intersects, intersection_point_0, intersection_point_1
    return lax.cond(
        jnp.any(jnp.isnan(jnp.array([x1, y1, x2, y2]))),
        _nan_segment,
        _not_nan_segment,
        (x1, y1, x2, y2, xmin, xmax, ymin, ymax),
    )
@jit
def batch_segment_rectangle_intersection(x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax):
    return vmap(segment_rectangle_intersection, in_axes=(0,0,0,0,None,None,None,None))(x1s, y1s, x2s, y2s, xmin, xmax, ymin, ymax)
# Bound action space function
@partial(jit, static_argnames=("vmax", "wheels_distance", "dt", "robot_radius"))
def bound_action_space(obstacle_segments, robot_pose, vmax, wheels_distance, dt, robot_radius):
    """
    Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
    """
    # Convert obstacle segments to absolute coordinates
    # Translate segments to robot frame
    obstacle_segments = obstacle_segments.at[:, :, 0].set(obstacle_segments[:, :, 0] - robot_pose[0])
    obstacle_segments = obstacle_segments.at[:, :, 1].set(obstacle_segments[:, :, 1] - robot_pose[1])
    # Rotate segments by -robot_pose[2]
    c, s = jnp.cos(-robot_pose[2]), jnp.sin(-robot_pose[2])
    rot = jnp.array([[c, -s], [s, c]])
    obstacle_segments = jnp.einsum('ij,klj->kli', rot, obstacle_segments)
    # Lower ALPHA
    _, intersection_points0, intersection_points1 = batch_segment_rectangle_intersection(
        obstacle_segments[:,0,0],
        obstacle_segments[:,0,1],
        obstacle_segments[:,1,0],
        obstacle_segments[:,1,1],
        0., # xmin
        vmax * dt + robot_radius, # xmax
        -robot_radius, # ymin
        robot_radius, # ymax
    )
    intersection_points = jnp.vstack((intersection_points0, intersection_points1))
    min_x = jnp.nanmin(intersection_points[:,0])
    new_alpha = lax.cond(
        ~jnp.isnan(min_x),
        lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (vmax * dt),
        lambda _: 1.,
        None,
    )
    @jit
    def _lower_beta_and_gamma(tup:tuple):
        obstacle_segments, new_alpha, vmax, wheels_distance, dt, robot_radius = tup
        # Lower BETA
        _, intersection_points0, intersection_points1 = batch_segment_rectangle_intersection(
            obstacle_segments[:,0,0],
            obstacle_segments[:,0,1],
            obstacle_segments[:,1,0],
            obstacle_segments[:,1,1],
            -robot_radius, # xmin
            new_alpha * vmax * dt + robot_radius, # xmax
            robot_radius, # ymin
            robot_radius + (new_alpha*dt**2*vmax/(4*wheels_distance)), # ymax
        )
        intersection_points = jnp.vstack((intersection_points0, intersection_points1))
        min_y = jnp.nanmin(intersection_points[:,1])
        new_beta = lax.cond(
            ~jnp.isnan(min_y),
            lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax * dt**2 * new_alpha),
            lambda _: 1.,
            None,
        )
        # Lower GAMMA
        _, intersection_points0, intersection_points1 = batch_segment_rectangle_intersection(
            obstacle_segments[:,0,0],
            obstacle_segments[:,0,1],
            obstacle_segments[:,1,0],
            obstacle_segments[:,1,1],
            -robot_radius, # xmin
            new_alpha * vmax * dt + robot_radius, # xmax
            -robot_radius - (new_alpha*dt**2*vmax/(4*wheels_distance)), # ymin
            -robot_radius, # ymax
        )
        intersection_points = jnp.vstack((intersection_points0, intersection_points1))
        max_y = jnp.nanmax(intersection_points[:,1])
        new_gamma = lax.cond(
            ~jnp.isnan(max_y),
            lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax * dt**2 * new_alpha),
            lambda _: 1.,
            None,
        )
        return new_beta, new_gamma
    new_beta, new_gamma = lax.cond(
        new_alpha == 0.,
        lambda _: (1., 1.),
        _lower_beta_and_gamma,
        (obstacle_segments, new_alpha, vmax, wheels_distance, dt, robot_radius)
    )
    # Apply lower blound to new_alpha, new_beta, new_gamma
    new_alpha = jnp.max(jnp.array([epsilon, new_alpha]))
    new_beta = jnp.max(jnp.array([epsilon, new_beta]))
    new_gamma = jnp.max(jnp.array([epsilon, new_gamma]))
    return new_alpha, new_beta, new_gamma

### SHAPELY BASED ALGORITHM
# Segment-rectangle intersection function using Shapely
def segment_rectangle_intersection_shapely(line:LineString, xmin, xmax, ymin, ymax):
    """
    Compute the intersection of a segment with a rectangle defined by (xmin, xmax, ymin, ymax).
    Returns a tuple (inside_or_intersects, intersection_point_0, intersection_point_1).
    """
    rect = box(xmin, ymin, xmax, ymax)
    if line.intersection(rect):
        points = jnp.array(line.intersection(rect).coords)
        return True, points[0], points[-1]
    else:
        return False, jnp.array([jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan])
def bound_action_space_shapely(obstacle_segments, robot_pose, vmax, wheels_distance, dt, robot_radius):
    """
    Compute the bounds of the action space based on the control parameters alpha, beta, gamma.
    """
    # Convert obstacle segments to local robot frame
    # Translate segments to robot frame
    obstacle_segments = [translate(segment, float(-robot_pose[0]), float(-robot_pose[1])) for segment in obstacle_segments]
    # Rotate segments by -robot_pose[2]
    obstacle_segments = [rotate(segment, float(-jnp.rad2deg(robot_pose[2])), origin=(0, 0)) for segment in obstacle_segments]
    # Lower ALPHA
    intersection_points = []
    for line in obstacle_segments:
        _, intersection_points0, intersection_points1 = segment_rectangle_intersection_shapely(
            line,
            0., # xmin
            vmax * dt + robot_radius, # xmax
            -robot_radius, # ymin
            robot_radius, # ymax
        )
        intersection_points.append(intersection_points0)
        intersection_points.append(intersection_points1)
    intersection_points = jnp.array(intersection_points)
    min_x = jnp.nanmin(intersection_points[:,0])
    new_alpha = lax.cond(
        ~jnp.isnan(min_x),
        lambda _: jnp.max(jnp.array([0, min_x - robot_radius])) / (vmax * dt),
        lambda _: 1.,
        None,
    )
    if new_alpha == 0.:
        return epsilon, 1., 1.
    # Lower BETA
    intersection_points = []
    for line in obstacle_segments:
        _, intersection_points0, intersection_points1 = segment_rectangle_intersection_shapely(
            line,
            -robot_radius, # xmin
            new_alpha * vmax * dt + robot_radius, # xmax
            robot_radius, # ymin
            robot_radius + (new_alpha*dt**2*vmax/(4*wheels_distance)), # ymax
        )
        intersection_points.append(intersection_points0)
        intersection_points.append(intersection_points1)
    intersection_points = jnp.array(intersection_points)
    min_y = jnp.nanmin(intersection_points[:,1])
    new_beta = lax.cond(
        ~jnp.isnan(min_y),
        lambda _: (min_y - robot_radius) * 4 * wheels_distance / (vmax * dt**2 * new_alpha),
        lambda _: 1.,
        None,
    )
    # Lower GAMMA
    intersection_points = []
    for line in obstacle_segments:
        _, intersection_points0, intersection_points1 = segment_rectangle_intersection_shapely(
            line,
            -robot_radius, # xmin
            new_alpha * vmax * dt + robot_radius, # xmax
            -robot_radius - (new_alpha*dt**2*vmax/(4*wheels_distance)), # ymin
            -robot_radius, # ymax
        )
        intersection_points.append(intersection_points0)
        intersection_points.append(intersection_points1)
    intersection_points = jnp.array(intersection_points)
    max_y = jnp.nanmax(intersection_points[:,1])
    new_gamma = lax.cond(
        ~jnp.isnan(max_y),
        lambda _: (-max_y - robot_radius) * 4 * wheels_distance / (vmax * dt**2 * new_alpha),
        lambda _: 1.,
        None,
    )
    return new_alpha, new_beta, new_gamma

### Measure computation time of single segment-rectangle intersection with JAX
_ = segment_rectangle_intersection(segments[0,0,0], segments[0,0,1], segments[0,1,0], segments[0,1,1], 0., vmax * dt + robot_radius, -robot_radius, robot_radius)
start_time = time.time()
for _ in range(iterations):
    intersects, intersection_point_0, intersection_point_1 = segment_rectangle_intersection(
        segments[0,0,0], segments[0,0,1], segments[0,1,0], segments[0,1,1], 0., vmax * dt + robot_radius, -robot_radius, robot_radius
    )
    intersects.block_until_ready()
    intersection_point_0.block_until_ready()
    intersection_point_1.block_until_ready()
end_time = time.time()
print(f"JAX - Single Segment-Rectangle Intersection - Average time taken for one iteration: {(end_time - start_time)/iterations:.8f} seconds\n")

### Measure computation time of for single segment-rectangle intersection with SHAPELY
start_time = time.time()
for _ in range(iterations):
    intersects, intersection_point_0, intersection_point_1 = segment_rectangle_intersection_shapely(
        shapely_segments[0], 0., vmax * dt + robot_radius, -robot_radius, robot_radius
    )
end_time = time.time()
print(f"SHAPELY - Single Segment-Rectangle Intersection - Average time taken for one iteration: {(end_time - start_time)/iterations:.8f} seconds\n")

### Measure computation time of JAX BASED ALGORITHM during multiple iterations
_ = bound_action_space(segments, robot_pose, vmax, wheels_distance, dt, robot_radius)
start_time = time.time()
for _ in range(iterations):
    new_alpha, new_beta, new_gamma = bound_action_space(segments, robot_pose, vmax, wheels_distance, dt, robot_radius)
    new_alpha.block_until_ready()
    new_beta.block_until_ready()
    new_gamma.block_until_ready()
end_time = time.time()
print(f"JAX Compelete Algorithm - Average time taken for one iteration: {(end_time - start_time)/iterations:.8f} seconds\n")

### Measure computation time of SHAPELY BASED ALGORITHM during multiple iterations
start_time = time.time()
for _ in range(iterations):
    new_alpha, new_beta, new_gamma = bound_action_space_shapely(shapely_segments, robot_pose, vmax, wheels_distance, dt, robot_radius)
end_time = time.time()
print(f"SHAPELY Complete Algorithm - Average time taken for one iteration: {(end_time - start_time)/iterations:.8f} seconds\n")

### Final output
# new_alpha, new_beta, new_gamma = bound_action_space(segments, robot_pose, vmax, wheels_distance, dt, robot_radius)
new_alpha, new_beta, new_gamma = bound_action_space_shapely(shapely_segments, robot_pose, vmax, wheels_distance, dt, robot_radius)
print(f"New alpha (for collision avoidance): {new_alpha}")
print(f"New beta (for collision avoidance): {new_beta}")
print(f"New gamma (for collision avoidance): {new_gamma}")
# Plotting the feasible region
plt.gca().add_artist(plt.Circle((robot_pose[0], robot_pose[1]), robot_radius, color='black', fill=False, zorder=3, linewidth=2))
plt.gca().add_artist(plt.Circle((robot_pose[0] + jnp.cos(robot_pose[2]) * robot_radius, robot_pose[1] + jnp.sin(robot_pose[2]) * robot_radius), robot_radius/10, color='black', fill=False, zorder=3, linewidth=2, linestyle='--'))
plt.gca().add_artist(plt.Rectangle(
    (robot_pose[0] - robot_radius, robot_pose[1] - dt**2*vmax/(4*wheels_distance) - robot_radius), 
    vmax*dt + 2 * robot_radius,
    2*robot_radius + (dt**2*vmax/(4*wheels_distance) * (2)), 
    rotation_point=(float(robot_pose[0]), float(robot_pose[1])), 
    angle=jnp.rad2deg(robot_pose[2]), 
    color='red', 
    fill=False, 
    zorder=3, 
    linewidth=2
))
plt.gca().add_artist(plt.Rectangle(
    (robot_pose[0] - robot_radius, robot_pose[1] - new_alpha*dt**2*new_gamma*vmax/(4*wheels_distance) - robot_radius), 
    new_alpha*vmax*dt + 2 * robot_radius, 
    2*robot_radius + (new_alpha*dt**2*vmax/(4*wheels_distance) * (new_beta + new_gamma)), 
    rotation_point=(float(robot_pose[0]), float(robot_pose[1])), 
    angle=jnp.rad2deg(robot_pose[2]), 
    color='green', 
    fill=False, 
    zorder=3, 
    linewidth=2
))
for o in obstacles: plt.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
# Set plot specs
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.xlim(-0.2, 2)
plt.ylim(-0.2, 2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
