import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from planner.lattice_planner import lattice_plan, get_surround_vehicles
from planner.pure_persuit import pure_persuit
from forecast_model.forecast import Model
from asp_decider.asp_plan import asp_plan
from asp_decider.asp_utils import neighbour_vehicle_to_asp

env = gym.make('highway-v0', render_mode='rgb_array')
env.config["offscreen_rendering"] = False
env.config["policy_frequency"] = 15
env.config["action"]["type"] = "ContinuousAction"
env.config["action"]["acceleration_range"] = (-10.0, 10.0)
env.config["vehicles_density"] = 1
env.reset()
env.render()

goal = """#program final.
:- is_ego(E), is_vehicle(V), E != V, not ahead(E,V).
"""

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
    follow_vehs, overtake_vehs, cover_vehs, s_bound, l_bound = model.get_sample_area(1, road, ego)
                
    if s_bound[1] - s_bound[0] < 1.0 or l_bound[1] - l_bound[0] < 0.5:
        print("Invalid sample area.")
        return None
    else:
        traj = lattice_plan(road, ego, 30.0, sample_s_points=np.linspace(s_bound[0], s_bound[1], 7), sample_l_points=(np.linspace(l_bound[0], l_bound[1], 6) + [0.0] if l_bound[0]*l_bound[1]<0 else np.linspace(l_bound[0], l_bound[1], 7)), follow_obstacles=follow_vehs, overtake_obstacles=overtake_vehs, cover_obstacles=cover_vehs)
        
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

exec_model = None

for i in range(1000):
    print("----------------------------------------------")
    road, ego = env.unwrapped.get_ground_truth()
    action = None
    # models = []
    try:
        if exec_model is None or not model_check(road, ego, exec_model):
            
                models = asp_plan(road, ego, goal)
                if len(models) == 0:
                    raise RuntimeWarning("No plan found.")
                models = [Model(model, road, ego) for model in models]
                print(len(models))
                models.sort(key=lambda x: x.get_prob(), reverse=True)
                for model in models:
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
        action = plan_without_model(road, ego)
        if action is None:
            print("Unable to plan. Use zero action.")
            action = np.array([0.0, 0.0])
        
    #         planned = False
    #         for model in models:
    #             print(model.get_prob())
                
    #                 planned = True
    #                 break
    #         if not planned:
    #             print("All model is invalid. Downgrade to naive lattice planner.")
    #             traj = lattice_plan(road, ego, 30.0, obstacles=get_surround_vehicles(road, ego))
    #             if traj is None:
    #                 break
    #             acc, steer = pure_persuit(traj, ego)
    #             action = np.array([acc / 5.0, steer / 0.7853981633974483])
    #             obs, reward, done, truncated, info = env.step(action)
    #     else:
    #         print("No plan found. Downgrade to naive lattice planner.")
            
    #         print(action)
    #         obs, reward, done, truncated, info = env.step(action)
    # else:
    #     traj = lattice_plan(road, ego, 30.0, sample_s_points=np.linspace(s_bound[0], s_bound[1], 7), sample_l_points=np.linspace(l_bound[0], l_bound[1], 7) + [0.0], follow_obstacles=follow_vehs, overtake_obstacles=overtake_vehs, cover_obstacles=cover_vehs)  
    #     if traj is None:
    #         continue
    #     exec_model = model
    #     acc, steer = pure_persuit(traj, ego)
    #     action = np.array([acc / 5.0, steer / 0.7853981633974483])
    #     obs, reward, done, truncated, info = env.step(action)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    if done or truncated:
        break
    
    # traj_p = np.array([[p.x, p.y] for p in traj])
    # traj_p -= ego.position
    # traj_p *= env.unwrapped.config["scaling"]
    # traj_p += np.array([180, 73])
    # plt.plot(traj_p[:, 0], traj_p[:, 1], 'r-')
    # # for traj in trajs:
    # #     traj_p = np.array([[p.x, p.y] for p in traj])
    # #     traj_p -= ego.position
    # #     traj_p += np.array([180, 73])
    # #     plt.plot(traj_p[:, 0], traj_p[:, 1], 'b-')
    # plt.imshow(env.render())
    # plt.show()
input("Press Enter to continue...")

env.close()
