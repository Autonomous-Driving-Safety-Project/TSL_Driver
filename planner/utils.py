import numpy as np
import numba as nb
from numba import float64, njit
from numba.experimental import jitclass

from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road
from typing import List, Tuple

def get_state(ego: Vehicle):
    position = ego.position
    speed = ego.speed
    hdg = ego.heading
    acc = ego.action['acceleration']
    steer = ego.action['steering']
    beta = np.arctan(1 / 2 * np.tan(steer))
    kappa = np.sin(beta) / (ego.LENGTH / 2)
    return position[0], position[1], speed, hdg, acc, kappa

def get_surround_vehicles(road: Road, ego: Vehicle):
    vehicles = set()
    for lane_idx in road.network.side_lanes(ego.lane_index) + [ego.lane_index]:
        preceder, follower = road.neighbour_vehicles(ego, lane_idx)
        if preceder is not None:
            vehicles.add(preceder)
        if follower is not None:
            vehicles.add(follower)
    return vehicles

def estimate_vehicle_s_bound(veh: Vehicle, refline: np.ndarray, t:float):
    s = estimate_vehicle_s(veh, refline, t)
    return s - veh.LENGTH / 2, s + veh.LENGTH / 2

def estimate_vehicle_s(veh: Vehicle, refline: np.ndarray, t:float):
    curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(refline_project(refline, veh.position), *(get_state(veh)[:4]))
    s = curr_s + curr_dot_s * t
    # print(f"S profile: {veh.id}: {curr_s}, {curr_dot_s}")
    return s

def range_clip(range:Tuple[float, float], min_v, max_v) -> Tuple[float, float]:
    if max(range[0],range[1]) < min_v:
        return (min_v, min_v)
    if min(range[0],range[1]) > max_v:
        return (max_v, max_v)
    else:
        return (max(min_v, min(range[0],range[1])), min(max_v, max(range[0],range[1])))
    

class Polynome(object): # abstract for typing
    def __init__(self) -> None:
        self.param_range = np.zeros(2)

    def value(self, x):
        pass
    
    def velocity(self, x):
        pass

    def acceleration(self, x):
        pass
    
    def jerk(self, x):
        pass

    def fit(self, *args):
        pass

@jitclass([
    ('a', float64),
    ('b', float64),
    ('c', float64),
    ('d', float64),
    ('e', float64),
    ('param_range', float64[:]),
])
class PolynomeOrder4():
    def __init__(self, a=0, b=0, c=0, d=0, e=0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.param_range = np.zeros(2)

    def value(self, x):
        return self.a * x**4 + self.b * x**3 + self.c * x**2 + self.d * x + self.e
    
    def velocity(self, x):
        return 4 * self.a * x**3 + 3 * self.b * x**2 + 2 * self.c * x + self.d

    def acceleration(self, x):
        return 12 * self.a * x**2 + 6 * self.b * x + 2 * self.c
    
    def jerk(self, x):
        return 24 * self.a * x + 6 * self.b

    def fit(self, t_0, y_0, dot_y_0, ddot_y_0, t_1, dot_y_1, ddot_y_1):
        A = np.array(
            [
                [t_0**4, t_0**3, t_0**2, t_0, 1],
                [4 * t_0**3, 3 * t_0**2, 2 * t_0, 1, 0],
                [12 * t_0**2, 6 * t_0, 2, 0, 0],
                [4 * t_1**3, 3 * t_1**2, 2 * t_1, 1, 0],
                [12 * t_1**2, 6 * t_1, 2, 0, 0],
            ]
        )
        B = np.array([y_0, dot_y_0, ddot_y_0, dot_y_1, ddot_y_1])
        coeff = np.linalg.solve(A, B)
        self.a = coeff[0]
        self.b = coeff[1]
        self.c = coeff[2]
        self.d = coeff[3]
        self.e = coeff[4]
        self.param_range[0] = t_0
        self.param_range[1] = t_1

@jitclass([
    ('a', float64),
    ('b', float64),
    ('c', float64),
    ('d', float64),
    ('e', float64),
    ('f', float64),
    ('param_range', float64[:]),
])
class PolynomeOrder5():
    def __init__(self, a=0, b=0, c=0, d=0, e=0, f=0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.param_range = np.zeros(2)

    def value(self, x):
        return (
            self.a * x**5
            + self.b * x**4
            + self.c * x**3
            + self.d * x**2
            + self.e * x
            + self.f
        )
    
    def velocity(self, x):
        return (
            5 * self.a * x**4
            + 4 * self.b * x**3
            + 3 * self.c * x**2
            + 2 * self.d * x
            + self.e
        )
    
    def acceleration(self, x):
        return (
            20 * self.a * x**3
            + 12 * self.b * x**2
            + 6 * self.c * x
            + 2 * self.d
        )
    
    def jerk(self, x):
        return (
            60 * self.a * x**2
            + 24 * self.b * x
            + 6 * self.c
        )

    def fit(self, t_0, y_0, dot_y_0, ddot_y_0, t_1, y_1, dot_y_1, ddot_y_1):
        A = np.array(
            [
                [t_0**5, t_0**4, t_0**3, t_0**2, t_0, 1],
                [5 * t_0**4, 4 * t_0**3, 3 * t_0**2, 2 * t_0, 1, 0],
                [20 * t_0**3, 12 * t_0**2, 6 * t_0, 2, 0, 0],
                [t_1**5, t_1**4, t_1**3, t_1**2, t_1, 1],
                [5 * t_1**4, 4 * t_1**3, 3 * t_1**2, 2 * t_1, 1, 0],
                [20 * t_1**3, 12 * t_1**2, 6 * t_1, 2, 0, 0],
            ]
        )
        B = np.array([y_0, dot_y_0, ddot_y_0, y_1, dot_y_1, ddot_y_1])
        coeff = np.linalg.solve(A, B)
        self.a = coeff[0]
        self.b = coeff[1]
        self.c = coeff[2]
        self.d = coeff[3]
        self.e = coeff[4]
        self.f = coeff[5]
        self.param_range[0] = t_0
        self.param_range[1] = t_1

class TrajectoryPointCartesian(object):
    def __init__(self, t, x, y, v, theta, a, kappa):
        self.t = t
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta
        self.a = a
        self.kappa = kappa
    
    def to_frenet(self, refline):
        rp = refline_project(refline, np.array([self.x, self.y]))
        s, l, dot_s, dot_l, prime_l, ddot_s, ddot_l, pprime_l = cartesian_to_frenet_l2(rp, self.x, self.y, self.v, self.theta, self.a, self.kappa)
        return TrajectoryPointFrenet(self.t, s, l, dot_s, ddot_s, prime_l, pprime_l)
    
    def get_vertices(self, length, width):
        return TrajectoryPointCartesian._get_vertices(self.x, self.y, self.theta, length, width)
    
    @staticmethod
    @nb.njit
    def _get_vertices(x, y, theta, length, width):
        lx = (length / 2) * np.cos(theta)
        ly = (length / 2) * np.sin(theta)
        wx = (width / 2) * np.sin(theta)
        wy = (width / 2) * np.cos(theta)
        vertices = np.array([
            [x + lx + wx, y + ly + wy],
            [x + lx - wx, y + ly - wy],
            [x - lx - wx, y - ly - wy],
            [x - lx + wx, y - ly + wy]
        ])
        return vertices

class TrajectoryPointFrenet(object):
    def __init__(self, t, s, l, dot_s, ddot_s, prime_l, pprime_l):
        self.t = t
        self.s = s
        self.l = l
        self.dot_s = dot_s
        self.ddot_s = ddot_s
        self.prime_l = prime_l
        self.pprime_l = pprime_l
    
    def to_cartesian(self, refline):
        rp = refline_get(refline, self.s)
        return TrajectoryPointCartesian(self.t, *frenet_to_cartesian_l2_prime(rp, self.l, self.dot_s, self.ddot_s, self.prime_l, self.pprime_l))

def get_refline(ego: Vehicle, lane: AbstractLane, ds = 1.0):
    S_BEGIN = -30.0
    S_END = 150.0
    
    s, l = lane.local_coordinates(ego.position)
    refline = np.empty(np.arange(s+S_BEGIN, s+S_END, ds).shape + (7,)) # s, x, y, sin_hdg, cos_hdg, kappa, d_kappa
    i = 0
    for _s in np.arange(s+S_BEGIN, s+S_END, ds):
        refline[i, 0] = _s - s
        refline[i,1:3] = lane.position(_s, 0)
        # Calculate direction angle (heading)
        dx = lane.position(_s + ds, 0)[0] - refline[i, 1]
        dy = lane.position(_s + ds, 0)[1] - refline[i, 2]
        refline[i, 3] = dy / np.sqrt(dx**2 + dy**2)  # sin_hdg
        refline[i, 4] = dx / np.sqrt(dx**2 + dy**2)  # cos_hdg
        
        # Calculate curvature
        dx_prev = lane.position(_s - ds, 0)[0] - refline[i, 1]
        dy_prev = lane.position(_s - ds, 0)[1] - refline[i, 2]
        dx_next = lane.position(_s + ds, 0)[0] - refline[i, 1]
        dy_next = lane.position(_s + ds, 0)[1] - refline[i, 2]
        
        curvature = (dx_prev * dy_next - dy_prev * dx_next) / (np.sqrt((dx_prev**2 + dy_prev**2) * (dx_next**2 + dy_next**2)) + 1e-6)
        refline[i, 5] = curvature
        
        # Calculate curvature derivative
        d_kappa = (curvature - refline[i-1, 5]) / ds if i > 0 else 0.0
        refline[i, 6] = d_kappa
        i += 1
    
    return refline

@nb.njit
def refline_project(refline: np.ndarray, point: np.ndarray):
    # Find the closest point on the reference line
    dists = np.sum((refline[:, 1:3] - point)**2, axis=1)
    closest_idx = np.argmin(dists)
    
    # Get the closest point on the reference line
    closest_point = refline[closest_idx, 1:3]
    
    # Calculate the projection of the input point onto the line segment
    prev_point = refline[max(closest_idx - 1, 0), 1:3]
    next_point = refline[min(closest_idx + 1, len(refline) - 1), 1:3]
    
    line = next_point - prev_point
    line_norm = np.linalg.norm(line)
    line_dir = line / line_norm
    
    projection = prev_point + np.dot(point - prev_point, line) / line_norm**2 * line
    
    # Interpolate parameters for the projected point
    s = refline[closest_idx, 0] + np.linalg.norm(projection - closest_point)
    sin_hdg = np.interp(s, refline[:, 0], refline[:, 3])
    cos_hdg = np.interp(s, refline[:, 0], refline[:, 4])
    kappa = np.interp(s, refline[:, 0], refline[:, 5])
    d_kappa = np.interp(s, refline[:, 0], refline[:, 6])
    
    return np.array([s, projection[0], projection[1], sin_hdg, cos_hdg, kappa, d_kappa])

@nb.njit
def refline_get(refline: np.ndarray, s: float):
    refpoint = np.empty(refline.shape[1])
    refpoint[0] = s
    # refpoint[1:] = np.interp(s, refline[:, 0], refline[:, 1:])
    for i in range(1, refline.shape[1]):
        refpoint[i] = np.interp(s, refline[:, 0], refline[:, i])
    return refpoint
    

@njit
def cartesian_to_frenet_l2(r: np.ndarray, x: float, y: float, v: float, theta: float, a: float, kappa: float):
    """
    convert from Cartesian x, y to Frenet s, l
    r: reference point
    x: x position
    y: y position
    v: speed
    theta: speed direction
    a: tangential acceleration 
    kappa: vehicle curvature
    returns s, l, dot(s), dot(l), l', ddot(s), ddot(l), l"
    """
    tau_r = np.array([r[4], r[3]])
    n_r = np.array([-r[3], r[4]])
    r_r = np.array([r[1], r[2]])
    
    tau_h = np.array([np.cos(theta), np.sin(theta)])
    n_h = np.array([-np.sin(theta), np.cos(theta)])
    r_h = np.array([x, y])
    v_h = v * tau_h
    a_h = a * tau_h + v * v * kappa * n_h
    
    s = r[0]
    l = (r_h - r_r).dot(n_r)
    dot_s = v_h.dot(tau_r) / (1 - r[5] * l) 
    dot_l = v_h.dot(n_r)
    prime_l = dot_l / dot_s if dot_s != 0 else 0
    ddot_s = (a_h.dot(tau_r) + dot_s ** 2 * (r[6] * l + 2 * r[5] * prime_l)) / (1 - r[5] * l)
    ddot_l = a_h.dot(n_r) - r[5] * dot_s * v_h.dot(tau_r)
    pprime_l = (ddot_l - ddot_s * prime_l) / dot_s ** 2 if dot_s != 0 else 0
    
    return s, l, dot_s, dot_l, prime_l, ddot_s, ddot_l, pprime_l

@njit
def cartesian_to_frenet_l1(r: np.ndarray, x: float, y: float, v: float, theta: float):
    """
    convert from Cartesian x, y to Frenet s, l
    r: reference point
    x: x position
    y: y position
    v: speed
    theta: speed direction
    returns s, l, dot(s), dot(l), l'
    """
    tau_r = np.array([r[4], r[3]])
    n_r = np.array([-r[3], r[4]])
    r_r = np.array([r[1], r[2]])
    
    tau_h = np.array([np.cos(theta), np.sin(theta)])
    n_h = np.array([-np.sin(theta), np.cos(theta)])
    r_h = np.array([x, y])
    v_h = v * tau_h
    
    s = r[0]
    l = (r_h - r_r).dot(n_r)
    dot_s = v_h.dot(tau_r) / (1 - r[5] * l) 
    dot_l = v_h.dot(n_r)
    prime_l = dot_l / dot_s if dot_s != 0 else 0
    
    return s, l, dot_s, dot_l, prime_l

@njit
def frenet_to_cartesian_l2_prime(r: np.ndarray, l: float, dot_s: float, ddot_s: float, prime_l: float, pprime_l: float):
    """
    convert from Frenet s, l to Cartesian x, y
    r: reference point
    s: s position
    l: l position
    dot_s: ds/dt
    ddot_s: dds/ddt
    dot_l: dl/dt
    prime_l: dl/ds
    ddot_l: ddl/ddt
    pprime_l: ddl/dds
    returns x, y, v, theta, a, kappa
    """
    dot_l = dot_s * prime_l
    ddot_l = pprime_l * dot_s ** 2 + prime_l * ddot_s
    r_r = np.array([r[1], r[2]])
    n_r = np.array([-r[3], r[4]])
    tau_r = np.array([r[4], r[3]])
    
    r_h = r_r + l * n_r
    v_t = (1-r[5]*l)*dot_s
    v_n = dot_l
    v = np.sqrt(v_t**2 + v_n**2)
    theta = np.arctan2(v_n, v_t) + np.arctan2(r[3],r[4])
    a_t = ddot_s * (1-r[5]*l) - dot_s**2 * (2 * r[5] * prime_l + r[6] * l)
    a_n = ddot_l + r[5] * (1-r[5]*l) * dot_s**2
    a = (a_n*n_r + a_t*tau_r).dot(np.array([np.cos(theta), np.sin(theta)]))
    n_h = np.array([-np.sin(theta), np.cos(theta)])
    kappa = n_h.dot(a_n*n_r + a_t*tau_r)
    
    return r_h[0], r_h[1], v, theta, a, kappa

@njit
def frenet_to_cartesian_l0(r: np.ndarray, l: float):
    """
    convert from Frenet s, l to Cartesian x, y
    r: reference point
    s: s position
    l: l position
    returns x, y
    """
    r_r = np.array([r[1], r[2]])
    n_r = np.array([-r[3], r[4]])
    
    r_h = r_r + l * n_r

    return r_h[0], r_h[1]