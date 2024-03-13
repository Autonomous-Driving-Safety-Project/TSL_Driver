import numpy as np
import numba as nb
from numba import float64, njit
from numba.experimental import jitclass

from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road

from asp_decider.asp_utils import distance_asp_to_range

from typing import List, Tuple

SAMPLE_T_POINTS = [1.0, 2.0, 3.0, 4.0]
SAMPLE_S_POINTS = [20.0, 40.0, 80.0]
SAMPLE_L_POINTS = [-4.0, 0.0, 4.0]
SPEED_MAX = 30.0
ACC_MAX = 10.0
ACC_MIN = -10.0
T_PRECISION = 0.2
S_PRECISION = 0.2
BUFFER_YIELD = 5.0
BUFFER_OVERTAKE = 5.0
COLLISION_COST_VAR = 0.25

WEIGHT_LON_OBJECTIVE = 1.0
WEIGHT_LON_JERK = 0.0
WEIGHT_LON_COLLISION = 20.0
WEIGHT_LAT_OFFSET = 0.1
WEIGHT_LAT_COMFORT = 0.0
WEIGHT_CENTRIPETAL_ACCELERATION = 0.0

COLLISION_CHECK_BUFFER_LON = 2.0
COLLISION_CHECK_BUFFER_LAT = 1.0

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
    curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(refline_project(refline, veh.position), *(get_state(veh)[:4]))
    s = curr_s + curr_dot_s * t
    return s - veh.LENGTH / 2, s + veh.LENGTH / 2

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

class PolyTrajectory(object):
    def __init__(self, st_poly, sl_poly) -> None:
        self.st_poly: Polynome = st_poly
        self.sl_poly: Polynome = sl_poly

    def get(self, t):
        s = self.st_poly.value(t)
        l = self.sl_poly.value(s)
        return TrajectoryPointFrenet(
            t,
            s,
            l,
            self.st_poly.velocity(t),
            self.st_poly.acceleration(t),
            self.sl_poly.velocity(s),
            self.sl_poly.acceleration(s),
        )

    def to_discretized(self, refline):
        descretized_points: List[TrajectoryPointFrenet] = []
        for t in np.arange(
            self.st_poly.param_range[0],
            self.st_poly.param_range[1] + T_PRECISION,
            T_PRECISION,
        ):
            descretized_points.append(self.get(t))
            if descretized_points[-1].s > self.sl_poly.param_range[1]:
                break
        return [
            descretized_point.to_cartesian(refline)
            for descretized_point in descretized_points
        ]

    def cost(
        self, obstacles: List[Vehicle], refline: np.ndarray, target_speed=None
    ) -> float:
        t_range = self.st_poly.param_range
        s_range = np.array([self.st_poly.value(t) for t in t_range])
        
        # 1. 目标cost
        lon_objective_cost = PolyTrajectory.lon_objective_cost(self.st_poly, self.sl_poly, target_speed, t_range)

        # 2. 舒适性cost
        lon_comfort_cost = PolyTrajectory.lon_comfort_cost(self.st_poly, self.sl_poly, t_range)
        
        # 3. 碰撞cost
        lon_collision_cost = PolyTrajectory.lon_collision_cost(self.st_poly, self.sl_poly, t_range, obstacles, refline)

        # 4. 向心加速度cost
        centripetal_cost = PolyTrajectory.centripetal_cost(self.st_poly, self.sl_poly, t_range, refline)

        # 5. 横向偏移cost
        lat_offset_cost = PolyTrajectory.lat_offset_cost(self.sl_poly, s_range)

        # 6. 横向舒适性cost
        lat_comfort_cost = PolyTrajectory.lat_comfort_cost(self.st_poly, self.sl_poly, t_range)

        return (
            lon_objective_cost * WEIGHT_LON_OBJECTIVE
            + lon_comfort_cost * WEIGHT_LON_JERK
            + lon_collision_cost * WEIGHT_LON_COLLISION
            + lat_offset_cost * WEIGHT_LAT_OFFSET
            + lat_comfort_cost * WEIGHT_LAT_COMFORT
            + centripetal_cost * WEIGHT_CENTRIPETAL_ACCELERATION
        )

    @staticmethod
    @njit
    def lon_objective_cost(st_poly: Polynome, sl_poly: Polynome, target_speed, t_range):
        
        target_speed_cost = 0
        if target_speed is not None:
            target_speed_cost_nume = 0
            target_speed_cost_deno = 0
            for t in np.arange(t_range[0], t_range[1] + T_PRECISION, T_PRECISION):
                s = st_poly.value(t)
                if s > sl_poly.param_range[1]:
                    break
                target_speed_cost_nume += abs(
                    st_poly.velocity(t) - target_speed
                ) * (t**2)
                target_speed_cost_deno += t**2
            target_speed_cost = target_speed_cost_nume / (1e-6 + target_speed_cost_deno)
        dist_travaled_cost = 1 / (
            1 + st_poly.value(t_range[1]) - st_poly.value(t_range[0])
        )

        lon_objective_cost = (1 * target_speed_cost + 10 * dist_travaled_cost) / 11
        return lon_objective_cost
    
    @staticmethod
    @njit
    def lon_comfort_cost(st_poly: Polynome, sl_poly: Polynome, t_range):
        jerk_cost_nume = 0
        jerk_cost_deno = 0
        for t in np.arange(t_range[0], t_range[1] + T_PRECISION, T_PRECISION):
            s = st_poly.value(t)
            if s > sl_poly.param_range[1]:
                break
            jerk = st_poly.jerk(t)
            jerk_cost_nume += (jerk / 2) ** 2
            jerk_cost_deno += abs(jerk / 2)
        lon_comfort_cost = jerk_cost_nume / (1 + jerk_cost_deno)
        return lon_comfort_cost
    
    @staticmethod
    # @lineprofile
    # @jit
    def lon_collision_cost(st_poly: Polynome, sl_poly: Polynome, t_range, obstacles: List[Vehicle], refline: np.ndarray):
        cost_square_sum = 0
        cost_sum = 0
        for t in np.arange(t_range[0], t_range[1] + T_PRECISION, T_PRECISION):
            s = st_poly.value(t)
            if s > sl_poly.param_range[1]:
                break
            for obs in obstacles:
                s_min, s_max = estimate_vehicle_s_bound(obs, refline, t)
                if s < s_min - BUFFER_YIELD:
                    d = s_min - BUFFER_YIELD - s
                elif s > s_max + BUFFER_OVERTAKE:
                    d = s - s_max - BUFFER_OVERTAKE
                else:
                    d = 0
                cost = np.exp(-(d**2) / (2 * COLLISION_COST_VAR))
                cost_square_sum += cost**2
                cost_sum += cost
        lon_collision_cost = cost_square_sum / (1e-6 + cost_sum)
        return lon_collision_cost
    
    @staticmethod
    @njit
    def centripetal_cost(st_poly: Polynome, sl_poly: Polynome, t_range, refline:np.ndarray):
        centripetal_cost_nume = 0
        centripetal_cost_deno = 0
        for t in np.arange(t_range[0], t_range[1] + T_PRECISION, T_PRECISION):
            s = st_poly.value(t)
            if s > sl_poly.param_range[1]:
                break
            dot_s = st_poly.velocity(t)
            rp = refline_get(refline, s)
            a = dot_s**2 * rp[5]
            centripetal_cost_nume += a**2
            centripetal_cost_deno += abs(a)
        centripetal_cost = centripetal_cost_nume / (1e-6 + centripetal_cost_deno)
        return centripetal_cost
    
    @staticmethod
    @njit
    def lat_offset_cost(sl_poly: Polynome, s_range):
        lat_offset_cost_nume = 0
        lat_offset_cost_deno = 0
        l_start = sl_poly.value(s_range[0])
        for s in np.arange(s_range[0], s_range[1] + S_PRECISION, S_PRECISION):
            if s > sl_poly.param_range[1]:
                break
            l = sl_poly.value(s)
            if l * l_start < 0:
                w = 10
            else:
                w = 1
            lat_offset_cost_nume += w * (l / 3) ** 2
            lat_offset_cost_deno += w * abs(l / 3)
        lat_offset_cost = lat_offset_cost_nume / (1e-6 + lat_offset_cost_deno)
        return lat_offset_cost
    
    @staticmethod
    @njit
    def lat_comfort_cost(st_poly: Polynome, sl_poly: Polynome, t_range):
        lat_comfort_costs = []
        for t in np.arange(t_range[0], t_range[1] + T_PRECISION, T_PRECISION):
            s = st_poly.value(t)
            if s > sl_poly.param_range[1]:
                break
            lat_comfort_costs.append(
                st_poly.velocity(t) ** 2 * sl_poly.acceleration(s)
                + sl_poly.velocity(s) * st_poly.acceleration(t)
            )
        lat_comfort_cost = max(lat_comfort_costs)
        return lat_comfort_cost

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

@njit
def _min_max(poly: np.ndarray):
    # 手动计算多边形顶点的最小和最大值
    min_x, min_y = poly[0, 0], poly[0, 1]
    max_x, max_y = poly[0, 0], poly[0, 1]
    for i in range(poly.shape[0]):
        if poly[i, 0] < min_x:
            min_x = poly[i, 0]
        if poly[i, 1] < min_y:
            min_y = poly[i, 1]
        if poly[i, 0] > max_x:
            max_x = poly[i, 0]
        if poly[i, 1] > max_y:
            max_y = poly[i, 1]
    return np.array([min_x, min_y]), np.array([max_x, max_y])

@njit
def _aabb_check(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    # 计算两个多边形的AABB
    min1, max1 = _min_max(poly1)
    min2, max2 = _min_max(poly2)
    # 检查AABB是否相交
    for i in range(2):  # 对x和y轴分别检查
        if max1[i] < min2[i] or max2[i] < min1[i]:
            return False
    return True

@njit
def _edge_normals(poly: np.ndarray):
    # 手动实现多边形顶点的轮转来计算边的法线
    num_vertices = poly.shape[0]
    normals = np.empty((num_vertices, 2))
    for i in range(num_vertices):
        # 计算当前边的向量
        current_vertex = poly[i]
        next_vertex = poly[(i + 1) % num_vertices]  # 循环取下一个顶点
        edge_vector = next_vertex - current_vertex
        # 计算法线向量（顺时针旋转90度）
        normal = np.array([-edge_vector[1], edge_vector[0]])
        normals[i] = normal
    return normals

@njit
def _project_poly(normal: np.ndarray, poly: np.ndarray):
    # 计算多边形在法向量上的投影
    projection = np.dot(np.ascontiguousarray(poly), np.ascontiguousarray(normal))
    return np.min(projection), np.max(projection)

@njit
def _overlap(min1, max1, min2, max2):
    # 检查两个投影是否重叠
    return not (max1 < min2 or max2 < min1)

@njit
def _sat_check(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    # 分离轴定理检测
    for poly in (poly1, poly2):
        normals = _edge_normals(poly)
        for normal in normals:
            min1, max1 = _project_poly(normal, poly1)
            min2, max2 = _project_poly(normal, poly2)
            if not _overlap(min1, max1, min2, max2):
                return False
    return True

@njit
def collision_check(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    # 凸多边形碰撞检测（AABB快筛+分离轴定理）
    # False for no collision, True for collision
    if not _aabb_check(poly1, poly2):
        return False  # 快速排除
    return _sat_check(poly1, poly2)  # 精确检测

@njit
def batch_collision_check(polys1: np.ndarray, polys2: np.ndarray) -> bool:
    max_i = min(polys1.shape[0], polys2.shape[0])
    # print(polys1.shape, polys2.shape)
    for i in range(max_i):
        if collision_check(polys1[i], polys2[i]):
            return True
    return False

# @njit
def get_vertices_array(pos: np.ndarray, hdg: np.ndarray, width: float, length: float):
    # 计算车辆的顶点
    points = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, +width / 2],
            [+length / 2, +width / 2],
            [+length / 2, -width / 2],
        ]
    ).T
    c, s = np.cos(hdg), np.sin(hdg)
    # rotation = np.array([[c, -s], [s, c]])
    rotation = np.stack((np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=-2)
    points = (rotation @ points).swapaxes(1,2) + np.tile(np.expand_dims(pos, axis=1),(1,4,1))
    return points

def collision_free(
    traj_cartesian: List[TrajectoryPointCartesian], obstacles: List[Vehicle], host_width, host_length
):
    host_vertices = np.array([point.get_vertices(host_length, host_width) for point in traj_cartesian])
    for obs in obstacles:
        # print(traj_cartesian[0].t, traj_cartesian[-1].t, T_PRECISION)
        # obs_vertices = obs.get_vertices_array(traj_cartesian[0].t, traj_cartesian[-1].t, T_PRECISION)
        obs_pos, obs_hdg = obs.predict_trajectory_constant_speed(np.arange(traj_cartesian[0].t, traj_cartesian[-1].t, T_PRECISION))
        obs_vertices = get_vertices_array(obs_pos, obs_hdg, obs.WIDTH, obs.LENGTH)
        # assert obs_vertices.shape[0] == host_vertices.shape[0]
        # print(obs_vertices.shape, host_vertices.shape)
        if batch_collision_check(host_vertices, obs_vertices):
            # 如果碰了
            return False
    return True

def off_road(traj_cartesian: List[TrajectoryPointCartesian], road: Road):
    for point in traj_cartesian:
        on_lane = False
        for lane in road.network.lanes_list():
            if lane.on_lane(np.array([point.x, point.y])):
                on_lane = True
                break
            
        if on_lane == False:
            return True
    return False

def lattice_plan(
        road:Road, 
        ego:Vehicle, 
        target_speed: float, 
        sample_t_points: List[float]=None, 
        sample_s_points: List[float]=None, 
        sample_l_points: List[float]=None,
        obstacles: List[Vehicle]=[],
    ) -> List[TrajectoryPointCartesian]:
    if sample_s_points is None:
        sample_s_points = SAMPLE_S_POINTS
    if sample_t_points is None:
        sample_t_points = SAMPLE_T_POINTS
    if sample_l_points is None:
        sample_l_points = SAMPLE_L_POINTS
    
    refline = get_refline(ego, ego.lane)
    # 获取规划起始点：用当前点代替
    plan_init_point = cartesian_to_frenet_l2(
        refline_project(refline, ego.position), *get_state(ego)
    )  # s, l, dot(s), dot(l), l', ddot(s), ddot(l), l"
    plan_init_t = 0
    # S-T规划
    st_polys = []
    ## 巡航
    if len(obstacles) == 0:
        for t in sample_t_points:
            for v in np.linspace(0, SPEED_MAX, 7):
                poly = PolynomeOrder4()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(v),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    ## 跟/超车
    for veh in obstacles:
        curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(
            refline_project(refline, veh.position), *(get_state(veh)[:4])
        )
        for t in sample_t_points:
            s_min, s_max = estimate_vehicle_s_bound(veh, refline, t)
            sample_s = [s_min - BUFFER_YIELD - 5.0, s_max + BUFFER_OVERTAKE + 5.0]
            for s in sample_s:
                if s <= plan_init_point[0]:
                    s = plan_init_point[0] + 0.1
                poly = PolynomeOrder5()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(s),
                    np.float64(curr_dot_s),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    # S-L规划
    sl_polys = []
    for s in sample_s_points:
        for l in sample_l_points:
            poly = PolynomeOrder5()
            poly.fit(
                np.float64(plan_init_point[0]),
                np.float64(plan_init_point[1]),
                np.float64(plan_init_point[4]),
                np.float64(plan_init_point[7]),
                np.float64(plan_init_point[0] + s),
                np.float64(l),
                np.float64(0.0),
                np.float64(0.0),
            )
            sl_polys.append(poly)
    # 合并轨迹，筛选
    trajectories = []
    for st_poly in st_polys:
        for sl_poly in sl_polys:
            trajectory = PolyTrajectory(st_poly, sl_poly)
            trajectories.append(trajectory)
    # TODO: 筛选
    # 计算cost，排序
    trajectories.sort(key=lambda x: x.cost(list(obstacles), refline, target_speed=target_speed))
    # 转离散，执行碰撞检测
    for traj in trajectories:
        traj_discretized = traj.to_discretized(refline)
        if collision_free(traj_discretized, list(obstacles), ego.WIDTH+COLLISION_CHECK_BUFFER_LAT, ego.LENGTH+COLLISION_CHECK_BUFFER_LON) and not off_road(traj_discretized, road):
            return traj_discretized

def lattice_plan_modeled(
        road:Road, 
        ego:Vehicle, 
        target_speed: float, 
        sample_t_points: List[float]=None, 
        sample_s_points: List[float]=None, 
        sample_l_points: List[float]=None,
        obstacles: List[Vehicle]=[],
        front_obstacles: List[Tuple[Vehicle, int]]=[],
        rear_obstacles: List[Tuple[Vehicle, int]]=[],
    ) -> List[TrajectoryPointCartesian]:
    if sample_s_points is None:
        sample_s_points = SAMPLE_S_POINTS
    if sample_t_points is None:
        sample_t_points = SAMPLE_T_POINTS
    if sample_l_points is None:
        sample_l_points = SAMPLE_L_POINTS
    
    refline = get_refline(ego, ego.lane)
    # 获取规划起始点：用当前点代替
    plan_init_point = cartesian_to_frenet_l2(
        refline_project(refline, ego.position), *get_state(ego)
    )  # s, l, dot(s), dot(l), l', ddot(s), ddot(l), l"
    plan_init_t = 0
    # S-T规划
    st_polys = []
    ## 巡航
    if len(obstacles) == 0 and len(front_obstacles) == 0 and len(rear_obstacles) == 0:
        for t in sample_t_points:
            for v in np.linspace(0, SPEED_MAX, 7):
                poly = PolynomeOrder4()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(v),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    ## 跟/超车
    for veh in obstacles:
        curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(
            refline_project(refline, veh.position), *(get_state(veh)[:4])
        )
        for t in sample_t_points:
            s_min, s_max = estimate_vehicle_s_bound(veh, refline, t)
            sample_s = [s_min - BUFFER_YIELD - 5.0, s_max + BUFFER_OVERTAKE + 5.0]
            for s in sample_s:
                if s <= plan_init_point[0]:
                    s = plan_init_point[0] + 0.1
                poly = PolynomeOrder5()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(s),
                    np.float64(curr_dot_s),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    # 跟车
    for veh,dis in front_obstacles:
        curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(
            refline_project(refline, veh.position), *(get_state(veh)[:4])
        )
        for t in sample_t_points:
            s_min, s_max = estimate_vehicle_s_bound(veh, refline, t)
            dis_min, dis_max = distance_asp_to_range(dis, veh, ego, ego)
            sample_s = np.linspace(s_min-dis_max, s_min-dis_min, 3)
            for s in sample_s:
                if s <= plan_init_point[0]:
                    s = plan_init_point[0] + 0.1
                poly = PolynomeOrder5()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(s),
                    np.float64(curr_dot_s),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    # 超车
    for veh,dis in rear_obstacles:
        curr_s, _, curr_dot_s, _, _ = cartesian_to_frenet_l1(
            refline_project(refline, veh.position), *(get_state(veh)[:4])
        )
        for t in sample_t_points:
            s_min, s_max = estimate_vehicle_s_bound(veh, refline, t)
            dis_min, dis_max = distance_asp_to_range(dis, ego, veh, ego)
            sample_s = np.linspace(s_max+dis_min, s_max+dis_max, 3)
            for s in sample_s:
                if s <= plan_init_point[0]:
                    s = plan_init_point[0] + 0.1
                poly = PolynomeOrder5()
                poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(s),
                    np.float64(curr_dot_s),
                    np.float64(0.0),
                )
                st_polys.append(poly)
    # S-L规划
    sl_polys = []
    for s in sample_s_points:
        for l in sample_l_points:
            poly = PolynomeOrder5()
            poly.fit(
                np.float64(plan_init_point[0]),
                np.float64(plan_init_point[1]),
                np.float64(plan_init_point[4]),
                np.float64(plan_init_point[7]),
                np.float64(plan_init_point[0] + s),
                np.float64(l),
                np.float64(0.0),
                np.float64(0.0),
            )
            sl_polys.append(poly)
    # 合并轨迹，筛选
    trajectories = []
    for st_poly in st_polys:
        for sl_poly in sl_polys:
            trajectory = PolyTrajectory(st_poly, sl_poly)
            trajectories.append(trajectory)
    # TODO: 筛选
    # 计算cost，排序
    trajectories.sort(key=lambda x: x.cost(list(obstacles) + [veh for veh,_ in front_obstacles] + [veh for veh,_ in rear_obstacles], refline, target_speed=target_speed))
    # 转离散，执行碰撞检测
    for traj in trajectories:
        traj_discretized = traj.to_discretized(refline)
        if collision_free(traj_discretized, list(obstacles) + [veh for veh,_ in front_obstacles] + [veh for veh,_ in rear_obstacles], ego.WIDTH+COLLISION_CHECK_BUFFER_LAT, ego.LENGTH+COLLISION_CHECK_BUFFER_LON) and not off_road(traj_discretized, road):
            return traj_discretized