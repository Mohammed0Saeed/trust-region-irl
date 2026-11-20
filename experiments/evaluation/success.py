"""Success criteria of a Push-T rollout, shared by the policy rollouts (env_policy.py) and MTP (mtp_runner.py).

SUCCESS_MODE (with SUCCESS_THRESHOLD = thr):
    AND       pos_err < thr and orient_err < thr     (m and rad; PushT's own rule)
    SUM       pos_err + orient_err < thr

Coverage (Diffusion Policy, Chi et al. 2023, arXiv:2303.04137) is not a success rule here: it is only recorded
for every rollout and averaged in the overall evaluation.
coverage = area(T block intersect goal T) / area(goal T), in the table plane. Both T's are the two boxes of
scene_mjx_free.xml (bottom: centre (0, 0), half sizes (0.05, 0.025); top: centre (0.075, 0), half sizes
(0.025, 0.06), in the body frame). The boxes of one T do not overlap, so the intersection area is the sum of the
pairwise rectangle intersections, computed exactly by convex polygon clipping.
"""
import numpy as np

SUCCESS_MODES = ("AND", "SUM")

T_BOX_CENTRES = np.array([[0.0, 0.0], [0.075, 0.0]])
T_BOX_HALF_SIZES = np.array([[0.05, 0.025], [0.025, 0.06]])
T_AREA = float(np.sum(4.0 * T_BOX_HALF_SIZES[:, 0] * T_BOX_HALF_SIZES[:, 1]))
_UNIT_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])  # counter-clockwise


def _t_rectangles(x, y, yaw):
    """The two boxes of a T at planar pose (x, y, yaw), each as 4 counter-clockwise corners."""
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return [(centre + _UNIT_CORNERS * half) @ rotation.T + np.array([x, y])
            for centre, half in zip(T_BOX_CENTRES, T_BOX_HALF_SIZES)]


def _clip(subject, clipper):
    """Sutherland-Hodgman: the part of convex polygon subject inside convex counter-clockwise polygon clipper."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def intersection(p, q, a, b):
        d1, d2 = q - p, b - a
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / denom
        return p + t * d1

    output = list(subject)
    for k in range(len(clipper)):
        a, b = clipper[k], clipper[(k + 1) % len(clipper)]
        polygon, output = output, []
        for j in range(len(polygon)):
            p, q = polygon[j - 1], polygon[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    output.append(intersection(p, q, a, b))
                output.append(q)
            elif inside(p, a, b):
                output.append(intersection(p, q, a, b))
        if not output:
            break
    return output


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    p = np.asarray(polygon)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def planar_yaw(quat):
    """Rotation about the vertical axis of a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def coverage(rel_xy, rel_quat):
    """Fraction of the goal T covered by the block, from the block pose relative to the goal
    (position rel_xy and quaternion rel_quat = conj(goal) * block, as in the PushT observation dims 0..6)."""
    if not np.all(np.isfinite(rel_xy)) or not np.all(np.isfinite(rel_quat)):
        return 0.0
    goal = _t_rectangles(0.0, 0.0, 0.0)
    block = _t_rectangles(float(rel_xy[0]), float(rel_xy[1]), planar_yaw(rel_quat))
    overlap = sum(_area(_clip(b, g)) for b in block for g in goal)
    return float(min(overlap / T_AREA, 1.0))


def is_success(mode, threshold, pos_err, orient_err):
    if mode == "AND":
        return pos_err < threshold and orient_err < threshold
    if mode == "SUM":
        return pos_err + orient_err < threshold
    raise ValueError(f"Unknown success mode {mode}, expected one of {SUCCESS_MODES}")


def describe(mode, threshold):
    return {"AND": f"pos_err < {threshold} m and orient_err < {threshold} rad",
            "SUM": f"pos_err + orient_err < {threshold}"}[mode]
