from asp_decider.asp_utils import road_network_to_asp, neighbour_vehicle_to_asp, asp2str
from asp_decider.TSL.main import solve
from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle

def asp_plan(road: Road, ego: Vehicle, goal):
    map_asp = road_network_to_asp(road.network)
    map_asp_str = asp2str(map_asp, "#program always.")
    veh_always, veh_init = neighbour_vehicle_to_asp(road, ego)
    vehicle_asp_str = asp2str(veh_always, "#program always.") + '\n' + asp2str(veh_init, "#program initial.")
    print(vehicle_asp_str)
    # print(map_asp_str)
    # _, ret, horizon = solve(map_asp_str, vehicle_asp_str, goal + "\n#program dynamic.\non_lane(C,L) :- is_vehicle(C), not is_ego(C), _on_lane(C,L).", models=1, imax=10)
    # if not ret.satisfiable:
    #     print("UNSAT!")
    #     print(vehicle_asp_str)
    #     return []
    # print(horizon)
    # models, _, _ = solve(map_asp_str, vehicle_asp_str, goal, imin=horizon, models=100, imax=horizon+1)
    models, ret, horizon = solve(map_asp_str, vehicle_asp_str, goal, models=100, imin=12, imax=20, branchcut=True,
                        #    options=["--heuristic=Domain"]
                           options=["--project=project"]
    )
    if not ret.satisfiable:
        # print(vehicle_asp_str)
        print("\033[31mUNSAT, plan without goal!\033[0m")
        models, ret, _ = solve(map_asp_str, vehicle_asp_str, models=10, imin=2, imax=2)
        if not ret.satisfiable:
            raise RuntimeError("UNSAT!")
    models = [model for model in models if len(model) > 1]
    # else:
    #     models, _, _ = solve(map_asp_str, vehicle_asp_str, goal, models=100, imin=horizon+1, imax=20, branchcut=True, options=["--project=project"])
    return models