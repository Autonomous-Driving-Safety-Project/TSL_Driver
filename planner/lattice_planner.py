import numpy as np
import numba as nb
import math
from numba import float64, njit
from numba.experimental import jitclass

from planner.utils import *

from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road

from asp_decider.asp_utils import distance_asp_to_range
from asp_decider.asp_grounding import SLTVArea
from typing import List, Tuple

SAMPLE_T_POINTS = [1.0, 2.0, 3.0, 4.0]
SAMPLE_S_POINTS = [20.0, 40.0, 80.0]
SAMPLE_L_POINTS = [-4.0, 0.0, 4.0]
SAMPLE_V_POINTS = [15.0, 20.0, 25.0, 30.0]
SPEED_MAX = 40.0
ACC_MAX = 6.0
ACC_MIN = -10.0
T_PRECISION = 0.2
S_PRECISION = 0.1
BUFFER_YIELD = 5.0
BUFFER_OVERTAKE = 5.0
COLLISION_COST_VAR = 0.25

WEIGHT_LON_OBJECTIVE = 20.0
WEIGHT_LON_JERK = 0.0
WEIGHT_LON_COLLISION = 0.0
WEIGHT_LAT_OFFSET = 5.0
WEIGHT_LAT_COMFORT = 0.0
WEIGHT_CENTRIPETAL_ACCELERATION = 0.0

COLLISION_CHECK_BUFFER_LON = 2.0
COLLISION_CHECK_BUFFER_LAT = 2.0

DISTANCE_TOLERANCE = 2.0

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
    # @njit
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
        if len(lat_comfort_costs) == 0:
            print(sl_poly.param_range)
        lat_comfort_cost = max(lat_comfort_costs)
        return lat_comfort_cost

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
        obs_vertices = get_vertices_array(np.array(obs_pos), np.array(obs_hdg), obs.WIDTH, obs.LENGTH)
        # assert obs_vertices.shape[0] == host_vertices.shape[0]
        # print(obs_vertices.shape, host_vertices.shape)
        if batch_collision_check(host_vertices, obs_vertices):
            # 如果碰了
            return False
    return True

def off_road(traj_cartesian: List[TrajectoryPointCartesian], road: Road):
    # return False
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
        area: SLTVArea,
        sample_t_points: List[float]=None, 
        sample_v_points: List[float]=None, 
        obstacles: List[Vehicle]=[],
    ) -> List[TrajectoryPointCartesian]:

    if sample_t_points is None:
        sample_t_points = SAMPLE_T_POINTS
    if sample_v_points is None:
        sample_v_points = SAMPLE_V_POINTS
    
    refline = get_refline(ego, ego.lane)
    # 获取规划起始点：用当前点代替
    plan_init_point = cartesian_to_frenet_l2(
        refline_project(refline, ego.position), *get_state(ego)
    )  # s, l, dot(s), dot(l), l', ddot(s), ddot(l), l"
    plan_init_t = 0

    # Sample
    trajectories = []
    
    # st_polys = []
    for t in sample_t_points:
        for v in sample_v_points:
            s_range, l_range = area.get_s_l(t, v, refline)
            if s_range is None or l_range is None:
                continue
            # print(f"t: {t}, v: {v}, s_range: {s_range[0]} ~ {s_range[1]}")
            if s_range[1] - s_range[0] < DISTANCE_TOLERANCE:
                continue
            s_range = range_clip(s_range, plan_init_point[0]+1.0, 300)
            for s in np.linspace(s_range[0], s_range[1], 7):
                st_poly = PolynomeOrder5()
                st_poly.fit(
                    np.float64(plan_init_t),
                    np.float64(plan_init_point[0]),
                    np.float64(plan_init_point[2]),
                    np.float64(plan_init_point[5]),
                    np.float64(plan_init_t + t),
                    np.float64(s),
                    np.float64(v),
                    np.float64(0.0),
                )
                # st_polys.append(st_poly)
                # for l in np.linspace(l_range[0], l_range[1], 3):
                for l in [(l_range[0] + l_range[1]) / 2]:
                    sl_poly = PolynomeOrder5()
                    sl_poly.fit(
                        np.float64(plan_init_point[0]),
                        np.float64(plan_init_point[1]),
                        np.float64(plan_init_point[4]),
                        np.float64(plan_init_point[7]),
                        np.float64(plan_init_point[0] + s),
                        np.float64(l),
                        np.float64(0.0),
                        np.float64(0.0),
                    )
                    trajectory = PolyTrajectory(st_poly, sl_poly)
                    trajectories.append(trajectory)
    # S-L规划
    # sl_polys = []
    # for s in SAMPLE_S_POINTS:
    #     for l in [area.l_range[0] + area.l_range[1] / 2]:
    #         poly = PolynomeOrder5()
    #         poly.fit(
    #             np.float64(plan_init_point[0]),
    #             np.float64(plan_init_point[1]),
    #             np.float64(plan_init_point[4]),
    #             np.float64(plan_init_point[7]),
    #             np.float64(plan_init_point[0] + s),
    #             np.float64(l),
    #             np.float64(0.0),
    #             np.float64(0.0),
    #         )
    #         sl_polys.append(poly)
    # 合并轨迹，筛选
    # trajectories = []
    # for st_poly in st_polys:
    #     for sl_poly in sl_polys:
    #         trajectory = PolyTrajectory(st_poly, sl_poly)
    #         trajectories.append(trajectory)            
    
    print(f"Trajectories: {len(trajectories)}")
    if len(trajectories) == 0:
        return None
    # TODO: 筛选
    # 计算cost，排序
    trajectories.sort(key=lambda x: x.cost(list(obstacles), refline, target_speed=target_speed))
    # 转离散，执行碰撞检测
    for traj in trajectories:
        traj_discretized = traj.to_discretized(refline)
        if collision_free(traj_discretized, list(obstacles), ego.WIDTH+COLLISION_CHECK_BUFFER_LAT, ego.LENGTH+COLLISION_CHECK_BUFFER_LON) and not off_road(traj_discretized, road):
            return traj_discretized