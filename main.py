import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from planner.lattice_planner import lattice_plan, lattice_plan_modeled, get_surround_vehicles
from planner.pure_persuit import pure_persuit
from forecast_model.forecast import Model
from asp_decider.asp_plan import asp_plan
from asp_decider.asp_utils import neighbour_vehicle_to_asp, distance_asp_to_range, road_network_to_asp, asp2str
from highway_env.road.road import Road
import pandas as pd
from datetime import datetime

env = gym.make('highway-v0', render_mode='rgb_array')
env.config["offscreen_rendering"] = False
env.config["policy_frequency"] = 15
env.config["action"]["type"] = "ContinuousAction"
env.config["action"]["acceleration_range"] = (-10.0, 10.0)
env.config["vehicles_density"] = 1

def model_check(road, ego, model):
    # 检查当前逻辑状态是否符合ASP模型
    # 当不符合时重新执行asp_plan
    # 由于只取1步，因此只要与第0步不一致即重新规划（无论与第一步一致与否均需要重新规划）
    veh_always, veh_init = neighbour_vehicle_to_asp(road, ego)
    for predicate in veh_init:
        if predicate not in model.model[0]:
            print("Model Check Failed!")
            # print(predicate)
            # print([str(p) for p in model.model[0]])
            return False
    print("Model Check Passed!")
    return True

def plan_with_model(road, ego, model):
    front_vehs, rear_vehs, l_bound = model.get_sample_area(1, road, ego)
    print("front: ",[(id(veh) % 1000, dis, distance_asp_to_range(dis, veh, ego, ego)) for veh,dis in front_vehs])
    print("rear: ",[(id(veh) % 1000, dis, distance_asp_to_range(dis, ego, veh, ego)) for veh,dis in rear_vehs])
                
    # if s_bound[1] - s_bound[0] < 1.0 or l_bound[1] - l_bound[0] < 0.5:
    if l_bound[1] - l_bound[0] < 0.5:
        print("Invalid sample area.")
        return None
    else:
        traj = lattice_plan_modeled(road, ego, 30.0, sample_l_points=np.linspace(l_bound[0], l_bound[1], 3), front_obstacles=front_vehs, rear_obstacles=rear_vehs)
        
        if traj is None:
            return None
        acc, steer = pure_persuit(traj, ego)
        action = np.array([acc / 10.0, steer / 0.7853981633974483])
        return action

def plan_without_model(road, ego):
    traj = lattice_plan(road, ego, 30.0, obstacles=get_surround_vehicles(road, ego))
    if traj is None:
        return None
    acc, steer = pure_persuit(traj, ego)
    action = np.array([acc / 10.0, steer / 0.7853981633974483])
    return action

for episode in range(50):
    env.reset()
    env.render()

    goal = """#program final.
    :- is_ego(E), is_vehicle(V), E != V, ahead(V,E).
    """

    metrics = pd.DataFrame(columns=["frame", "speed", "TTC_front", "TTC_rear", "done", "truncated"])
    exec_model = None
    frame_count = 0

    while True:
        print("----------------------------------------------")
        road:Road = None
        road, ego = env.unwrapped.get_ground_truth()
        action = None
        try:
            if exec_model is None or not model_check(road, ego, exec_model):
                
                    models = asp_plan(road, ego, goal)
                    if len(models) == 0:
                        raise RuntimeError("No plan found.")
                    models = [Model(model, road, ego) for model in models]
                    print(len(models))
                    # models.sort(key=lambda x: x.prob, reverse=True)
                    for model in models:
                        print(model.prob)
                        action = plan_with_model(road, ego, model)
                        if action is not None:
                            exec_model = model
                            break
                    if action is None:
                        print("All model is invalid. Downgrade to naive lattice planner.")
                        raise RuntimeWarning("No plan found.")
            
            else:
                action = plan_with_model(road, ego, exec_model)
                if action is None:
                    exec_model = None
                    print("Cannot use last model. Replan.")
                    continue
        except RuntimeWarning:
            dump_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            veh_always, veh_init = neighbour_vehicle_to_asp(road, ego)
            vehicle_asp_str = asp2str(veh_always, "#program always.") + '\n' + asp2str(veh_init, "#program initial.")
            models = asp_plan(road, ego, goal)
            
            with open(f"{dump_name}_warn.log", "w") as f:
                f.write(vehicle_asp_str)
                for v in road.vehicles:
                    f.write(f"\n{id(v) % 1000}: {v.position}, {v.speed}")
                
                f.write("\n")
                for i,model in enumerate(models):
                    f.write(f"Model {i}:\n")
                    m = Model(model, road, ego)
                    for j,state in enumerate(model):
                        f.write(f"State {j}:\n")
                        f.writelines([str(pred) for pred in state])
                        f.write("\n")
                    front_vehs, rear_vehs, l_bound = m.get_sample_area(1, road, ego)
                    f.write(f"front: {[(id(veh) % 1000, dis, distance_asp_to_range(dis, veh, ego, ego)) for veh,dis in front_vehs]}\n")
                    f.write(f"rear: {[(id(veh) % 1000, dis, distance_asp_to_range(dis, ego, veh, ego)) for veh,dis in rear_vehs]}\n")
                    from planner.lattice_planner import get_refline, estimate_vehicle_s
                    refline = get_refline(ego, ego.lane)
                    for t in [1.0, 2.0, 3.0, 4.0]:
                        for veh,dis in front_vehs:
                            s = estimate_vehicle_s(veh, refline, t)
                            dis_min, dis_max = distance_asp_to_range(dis, veh, ego, ego)
                            f.write(f"veh_{id(veh) % 1000}: {s-dis_max} ~ {s-dis_min}\n")
                        for veh,dis in rear_vehs:
                            s = estimate_vehicle_s(veh, refline, t)
                            dis_min, dis_max = distance_asp_to_range(dis, veh, ego, ego)
                            f.write(f"veh_{id(veh) % 1000}: {s+dis_min} ~ {s+dis_max}\n")
            plt.imsave(f"{dump_name}_warn.png", env.render())
            action = plan_without_model(road, ego)
            if action is None:
                print("Unable to plan. Use zero action.")
                action = np.array([0.0, 0.0])
        except RuntimeError:
            print("Unsatisfiable. Image dumped.")
            dump_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            veh_always, veh_init = neighbour_vehicle_to_asp(road, ego)
            vehicle_asp_str = asp2str(veh_always, "#program always.") + '\n' + asp2str(veh_init, "#program initial.")
            # print(vehicle_asp_str)
            # for v in road.vehicles:
            #     print(f"{id(v) % 1000}: {v.position}, {v.speed}")
            with open(f"{dump_name}_error.log", "w") as f:
                f.write(vehicle_asp_str)
                for v in road.vehicles:
                    f.write(f"\n{id(v) % 1000}: {v.position}, {v.speed}")
            plt.imsave(f"{dump_name}_error.png", env.render())
            break
        obs, reward, done, truncated, info = env.step(action)
        metric = env.unwrapped.get_metrics()
        print(metric)
        metrics = pd.concat([metrics,
            pd.DataFrame({
                "frame": frame_count,
                "speed": metric["speed"],
                "TTC_front": metric["TTC_front"],
                "TTC_rear": metric["TTC_rear"],
                "done": done,
                "truncated": truncated
            },index=[0])], ignore_index=True
        )
        frame_count += 1
        env.render()
        
        if done or truncated:
            break

    metrics.to_csv(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')}metrics_{episode}.csv")

input("Press Enter to continue...")
env.close()
