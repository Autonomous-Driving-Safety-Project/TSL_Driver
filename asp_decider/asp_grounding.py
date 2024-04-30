# 将ASP解转化为S-L-T-V空间的区域
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from forecast_model.forecast import Model
from typing import List, Tuple
import math
import numpy as np
from planner.utils import estimate_vehicle_s

class RSSRange(object):
    RHO = 0.1
    A_MAX_BRAKE = 5.0
    A_MIN_BRAKE = 5.0
    A_MAX_ACCEL = 5.0

    def __init__(self, vf: Vehicle, vr: Vehicle, distance: int) -> None:
        self.vf = vf
        self.vr = vr
        self.distance = distance
         
    def distance_range_f(self, v) -> Tuple[float, float]:
        if self.distance == 0:
            return (0, self.vf.LENGTH)
        if self.d_vr(v) <= 0:
            # d_hat == 0
            if self.distance == 1:
                return (self.vf.LENGTH, self.vf.LENGTH)
            else:
                return (self.vf.LENGTH, math.inf)
        else:
            if self.distance == 1:
                return (self.vf.LENGTH, self.vf.LENGTH + self.d_vr(v))
            else:
                return (self.vf.LENGTH + self.d_vr(v), math.inf)
    
    def distance_range_r(self, v) -> Tuple[float, float]:
        if self.distance == 0:
            return (0, self.vf.LENGTH)
        if self.d_vf(v) <= 0:
            # d_hat == 0
            if self.distance == 1:
                return (self.vf.LENGTH, self.vf.LENGTH)
            else:
                return (self.vf.LENGTH, math.inf)
        else:
            if self.distance == 1:
                return (self.vf.LENGTH, self.vf.LENGTH + self.d_vf(v))
            else:
                return (self.vf.LENGTH + self.d_vf(v), math.inf)
    
    def est_s_range_f(self, t:float, v:float, refline):
        s = estimate_vehicle_s(self.vf, refline, t)
        dr = self.distance_range_f(v)
        # print(f"F: {self.vf.id}: s={s}, dr={dr}")
        return (s - dr[1], s - dr[0])
    
    def est_s_range_r(self, t:float, v:float, refline):
        s = estimate_vehicle_s(self.vr, refline, t)
        dr = self.distance_range_r(v)
        # print(f"R: {self.vr.id}: s={s}, dr={dr}")
        return (s + dr[0], s + dr[1])

    def d_vr(self, vr:float):
        return vr * self.RHO + (vr + self.RHO * self.A_MAX_ACCEL) ** 2 / (2 * self.A_MIN_BRAKE) + 0.5 * self.A_MAX_ACCEL * self.RHO ** 2 - self.vf.speed ** 2 / (2 * self.A_MAX_BRAKE)
    
    def d_vf(self, vf:float):
        return self.vr.speed * self.RHO + (self.vr.speed + self.RHO * self.A_MAX_ACCEL) ** 2 / (2 * self.A_MIN_BRAKE) + 0.5 * self.A_MAX_ACCEL * self.RHO ** 2 - vf ** 2 / (2 * self.A_MAX_BRAKE)


class SLTVArea(object):
    def __init__(self, model: Model, road:Road, ego:Vehicle) -> None:
        self.model = model
        self.road = road
        self.ego = ego
        self.front_vehs, self.rear_vehs, self.l_range = model.get_sample_area(1, road, ego)
        self.front = [RSSRange(veh, ego, dist) for veh,dist in self.front_vehs]
        self.rear = [RSSRange(ego, veh, dist) for veh,dist in self.rear_vehs]
    
    def get_s_l(self, t:float, v:float, refline):
        s_max = []
        s_min = []
        for r in self.front:
            s1, s2 = r.est_s_range_f(t, v, refline)
            s_max.append(s2)
            s_min.append(s1)
            # print(f"t={t}, v={v}: {r.vf.id}: {s1} ~ {s2}")
        for r in self.rear:
            s1, s2 = r.est_s_range_r(t, v, refline)
            s_max.append(s2)
            s_min.append(s1)
            # print(f"t={t}, v={v}: {r.vr.id}: {s1} ~ {s2}")
        
        if len(s_max) == 0:
            return (-math.inf, math.inf), self.l_range
        
        s_max_m = min(s_max)
        s_min_m = max(s_min)
        
        # print(f"t={t}, v={v}: {s_min_m} ~ {s_max_m}")
        
        if s_max_m < s_min_m:
            return None, self.l_range
        else:
            return (s_min_m, s_max_m), self.l_range