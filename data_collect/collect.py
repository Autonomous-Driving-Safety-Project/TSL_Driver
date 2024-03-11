
import gymnasium as gym
import numpy as np
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road
import pandas as pd
import argparse
from tqdm import tqdm

def get_surround_vehicles(road:Road, ego:Vehicle): # 左前、左后、前、后、右前、右后
    vehicles = [None] * 6
    from_, to_, id_ = ego.lane_index
    for i in range(3):
        looking_id = id_ - 1 + i
        if looking_id < 0 or looking_id >= len(road.network.graph[from_][to_]):
            continue
        vehicles[2*i], vehicles[2*i+1] = road.neighbour_vehicles(ego, (from_, to_, looking_id))
    return vehicles

class Relation():
    AHEAD = 1
    COVER = 0
    BEHIND = 2
    LEFT = 1
    SAME_LANE = 0
    RIGHT = 2
    
    LON_NAME = {AHEAD: "AHEAD", COVER: "COVER", BEHIND: "BEHIND"}
    LAT_NAME = {LEFT: "LEFT", SAME_LANE: "SAME_LANE", RIGHT: "RIGHT"}
    def __init__(self, longitudinal, lateral):
        self.longitudinal = longitudinal
        self.lateral = lateral
    
    def __int__(self):
        return self.longitudinal * 3 + self.lateral
    
    def __eq__(self, __value: object) -> bool:
        return self.longitudinal == __value.longitudinal and self.lateral == __value.lateral
    
    def __ne__(self, __value: object) -> bool:
        return self.longitudinal != __value.longitudinal or self.lateral != __value.lateral
    
    def __repr__(self) -> str:
        return f"({Relation.LON_NAME[self.longitudinal]}, {Relation.LAT_NAME[self.lateral]})"

def _get_lon_relation(veh, ego) -> int:
    dist = veh.lane_distance_to(ego, veh.lane)
    half_length = (veh.LENGTH + ego.LENGTH) / 2
    if dist > half_length:
        return Relation.BEHIND
    elif dist < -half_length:
        return Relation.AHEAD
    else:
        return Relation.COVER

def _get_lat_relation(veh, ego) -> int:
    if veh.lane_index[2] < ego.lane_index[2]:
        return Relation.LEFT
    elif veh.lane_index[2] > ego.lane_index[2]:
        return Relation.RIGHT
    else:
        return Relation.SAME_LANE

def get_relationship(veh, ego): # veh R ego
    return Relation(_get_lon_relation(veh, ego), _get_lat_relation(veh, ego))

def collect(episodes=5, frames=10, render=False, file_name="data.csv"):
    print("Preparing environment...")
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.unwrapped.config["offscreen_rendering"] = not render
    env.unwrapped.config["policy_frequency"] = 1
    env.unwrapped.config["simulation_frequency"] = 5
    env.unwrapped.config["action"]["type"] = "DiscreteMetaAction"

    data_list = []

    for episode in tqdm(range(episodes)):
        env.reset()
        for frame in range(frames):
            road, ego = env.unwrapped.get_ground_truth()
            # action = env.action_type.actions_indexes["IDLE"]
            
            surrounds = get_surround_vehicles(road, ego)
            for veh in surrounds:
                if veh is None:
                    continue
                veh_surrounds = get_surround_vehicles(road, veh)
                relation_with_ego = get_relationship(veh, ego)
                data = {
                    "episode": episode, 
                    "frame": frame, 
                    "ego_id": id(ego) % 1000,
                    "ego_position_x": ego.position[0],
                    "ego_position_y": ego.position[1],
                    "ego_speed": ego.speed,
                    "ego_heading": ego.heading,
                    "vehicle_position_x": veh.position[0],
                    "vehicle_position_y": veh.position[1],
                    "vehicle_speed": veh.speed,
                    "vehicle_heading": veh.heading,
                    "vehicle_id": id(veh) % 1000,
                    "relate_position_x": veh.position[0] - ego.position[0],
                    "relate_position_y": veh.position[1] - ego.position[1],
                    "relate_speed": veh.speed - ego.speed,
                    "surrounds_LF_relate_position_x": veh_surrounds[0].position[0] - veh.position[0] if veh_surrounds[0] is not None else np.nan,
                    "surrounds_LF_relate_position_y": veh_surrounds[0].position[1] - veh.position[1] if veh_surrounds[0] is not None else np.nan,
                    "surrounds_LF_relate_speed": veh_surrounds[0].speed - veh.speed if veh_surrounds[0] is not None else np.nan,
                    "surrounds_LF_heading": veh_surrounds[0].heading if veh_surrounds[0] is not None else np.nan,
                    "surrounds_LB_relate_position_x": veh_surrounds[1].position[0] - veh.position[0] if veh_surrounds[1] is not None else np.nan,
                    "surrounds_LB_relate_position_y": veh_surrounds[1].position[1] - veh.position[1] if veh_surrounds[1] is not None else np.nan,
                    "surrounds_LB_relate_speed": veh_surrounds[1].speed - veh.speed if veh_surrounds[1] is not None else np.nan,
                    "surrounds_LB_heading": veh_surrounds[1].heading if veh_surrounds[1] is not None else np.nan,
                    "surrounds_F_relate_position_x": veh_surrounds[2].position[0] - veh.position[0] if veh_surrounds[2] is not None else np.nan,
                    "surrounds_F_relate_position_y": veh_surrounds[2].position[1] - veh.position[1] if veh_surrounds[2] is not None else np.nan,
                    "surrounds_F_relate_speed": veh_surrounds[2].speed - veh.speed if veh_surrounds[2] is not None else np.nan,
                    "surrounds_F_heading": veh_surrounds[2].heading if veh_surrounds[2] is not None else np.nan,
                    "surrounds_B_relate_position_x": veh_surrounds[3].position[0] - veh.position[0] if veh_surrounds[3] is not None else np.nan,
                    "surrounds_B_relate_position_y": veh_surrounds[3].position[1] - veh.position[1] if veh_surrounds[3] is not None else np.nan,
                    "surrounds_B_relate_speed": veh_surrounds[3].speed - veh.speed if veh_surrounds[3] is not None else np.nan,
                    "surrounds_B_heading": veh_surrounds[3].heading if veh_surrounds[3] is not None else np.nan,
                    "surrounds_RF_relate_position_x": veh_surrounds[4].position[0] - veh.position[0] if veh_surrounds[4] is not None else np.nan,
                    "surrounds_RF_relate_position_y": veh_surrounds[4].position[1] - veh.position[1] if veh_surrounds[4] is not None else np.nan,
                    "surrounds_RF_relate_speed": veh_surrounds[4].speed - veh.speed if veh_surrounds[4] is not None else np.nan,
                    "surrounds_RF_heading": veh_surrounds[4].heading if veh_surrounds[4] is not None else np.nan,
                    "surrounds_RB_relate_position_x": veh_surrounds[5].position[0] - veh.position[0] if veh_surrounds[5] is not None else np.nan,
                    "surrounds_RB_relate_position_y": veh_surrounds[5].position[1] - veh.position[1] if veh_surrounds[5] is not None else np.nan,
                    "surrounds_RB_relate_speed": veh_surrounds[5].speed - veh.speed if veh_surrounds[5] is not None else np.nan,
                    "surrounds_RB_heading": veh_surrounds[5].heading if veh_surrounds[5] is not None else np.nan,
                    "relation_with_ego": int(relation_with_ego),
                }
                data_list.append(data)
            action = np.random.choice(list(env.unwrapped.action_type.ACTIONS_ALL.keys()))
            obs, reward, done, truncated, info = env.step(action)
            if render:
                env.render()
            if done or truncated:
                break
    env.close()
    print(f"{len(data_list)} data collected.")
    print("Saving data...")
    df = pd.DataFrame(data_list)
    df.to_csv(file_name, index=False)

def main():
    argparse.ArgumentParser(description="Collect data from highway-env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--file_name", type=str, default="data.csv", help="File name to save")
    args = parser.parse_args()
    collect(args.episodes, args.frames, args.render, args.file_name)

if __name__ == "__main__":
    main()
