from planner.lattice_planner import TrajectoryPointCartesian
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from typing import List

LOOKAHEAD_DISTANCE = 3.0

def pure_persuit(trajectory: List[TrajectoryPointCartesian], ego: Vehicle):
    for traj_point in trajectory:
        point_coord = np.array([traj_point.x, traj_point.y])
        dist = np.linalg.norm(point_coord - ego.position)
        if dist >= LOOKAHEAD_DISTANCE:
            break
    kappa = 2 * np.array([-np.sin(ego.heading), np.cos(ego.heading)]).dot(point_coord - ego.position) / (dist ** 2) # FIXME
    velocity = dist / traj_point.t
    acceleration = (velocity - ego.speed) / traj_point.t
    steer = np.arctan(ego.LENGTH * kappa / 2)
    return acceleration, steer