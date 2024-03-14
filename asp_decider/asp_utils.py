from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from clingo import Function, Symbol, Number
from typing import List

def get_asp_lane_repr(from_: str, to_: str, id: int):
    return f"f{from_}t{to_}l{id}"

def get_asp_road_repr(from_: str, to_: str):
    return f"f{from_}t{to_}"

def asp2str(asp:List[Symbol], prefix:str="", filter:List[str]=[]):
    if filter:
        asp = [i for i in asp if i.name in filter]
    return prefix + "\n" +" \n".join([f"{i}." for i in asp])

def road_network_to_asp(network: RoadNetwork) -> List[Symbol]:
    """将HighwayEnv的路网转化成asp描述
       暂时只支持Highway环境

    Args:
        network (RoadNetwork): HighwayEnv的路网对象

    Returns:
        List[Symbol]: Clingo的Symbol列表
    """    
    result = []
    roads = set()
    for index, lane in network.lanes_dict().items():
        # result += f"is_lane({get_asp_lane_repr(*index)}).\n"
        result.append(Function("is_lane", [Function(get_asp_lane_repr(*index))]))
        # result += f"has_lane({get_asp_road_repr(index[0], index[1])},{get_asp_lane_repr(*index)}).\n"
        # result.append(Function("has_lane", [Function(get_asp_road_repr(index[0], index[1])), Function(get_asp_lane_repr(*index))]))
        roads.add((index[0], index[1]))
        id_ = index[2]
        if id_ > 0:
            # result += f"left({get_asp_lane_repr(index[0], index[1], id_-1)},{get_asp_lane_repr(*index)}).\n"
            result.append(Function("left", [Function(get_asp_lane_repr(index[0], index[1], id_-1)), Function(get_asp_lane_repr(*index))]))
    # for road in roads:
    #     # result += f"is_road({get_asp_road_repr(*road)}).\n"
    #     result.append(Function("is_road", [Function(get_asp_road_repr(*road))]))
    return result

def get_asp_vehicle_repr(vehicle):
    return f"veh_{id(vehicle) % 1000}"

def get_lon_relation(v1, v2) -> Symbol:
    dist = v1.lane_distance_to(v2, v1.lane)
    half_length = (v1.LENGTH + v2.LENGTH) / 2
    if dist > half_length:
        return Function("ahead", [Function(get_asp_vehicle_repr(v2)), Function(get_asp_vehicle_repr(v1))])
    elif dist < -half_length:
        return Function("ahead", [Function(get_asp_vehicle_repr(v1)), Function(get_asp_vehicle_repr(v2))])
    else:
        return Function("cover", [Function(get_asp_vehicle_repr(v1)), Function(get_asp_vehicle_repr(v2))])

def neighbour_vehicle_to_asp(road, ego):
    """将HighwayEnv的车辆转化成asp描述
       暂时只支持Highway环境

    Args:
        road (Road): HighwayEnv的Road对象
        ego (Vehicle): HighwayEnv的自车

    Returns:
        always (List[Symbol]): #program always程序段（包含is_vehicle和is_ego）
        initial (List[Symbol]): #program initial程序段（当前车辆状态）
    """    
    always = []
    initial = []
    always.append(Function("is_vehicle", [Function(get_asp_vehicle_repr(ego))]))
    always.append(Function("is_ego", [Function(get_asp_vehicle_repr(ego))]))
    vehicles = set()
    vehicles.add(ego)
    network:RoadNetwork = road.network
    for lane_idx in network.all_side_lanes(ego.lane_index):
        preceder, follower = road.neighbour_vehicles(ego, lane_idx)
        if preceder is not None:
            always.append(Function("is_vehicle", [Function(get_asp_vehicle_repr(preceder))]))
            
            vehicles.add(preceder)
        if follower is not None:
            always.append(Function("is_vehicle", [Function(get_asp_vehicle_repr(follower))]))
            vehicles.add(follower)
    vehicles = list(vehicles)
    for i in range(len(vehicles) - 1):
        for j in range(i+1, len(vehicles)):
            # initial.append(get_lon_relation(vehicles[i], vehicles[j]))
            initial.append(get_distance(vehicles[i], vehicles[j], ego))
    for vehicle in vehicles:
        initial.append(Function("on_lane", [Function(get_asp_vehicle_repr(vehicle)), Function(get_asp_lane_repr(*vehicle.lane_index))]))
    return always, initial

def rss_safe_distance(veh:Vehicle, veh_f:Vehicle):
    RHO = 0.1
    A_MAX_BRAKE = 5.0
    A_MIN_BRAKE = 5.0
    A_MAX_ACCEL = 5.0
    vr = veh.speed
    vf = veh_f.speed
    return max(0, vr * RHO + 0.5 * A_MAX_ACCEL * RHO ** 2 + (vr + RHO * A_MAX_ACCEL) ** 2 / (2 * A_MIN_BRAKE) - vf ** 2 / (2 * A_MAX_BRAKE))

def get_distance(v1:Vehicle, v2:Vehicle, ego:Vehicle):
    if v1.position[0] >= v2.position[0]:
        # v1在v2前面
        dis_asp = None
        dis = v1.position[0] - v2.position[0]
        vr_sd = rss_safe_distance(v2, ego)
        ego_sd = rss_safe_distance(ego, v1)
        if dis < v1.LENGTH:
            dis_asp = 0
        elif dis <  v1.LENGTH + ego.LENGTH:
            dis_asp = 1
        elif dis < v1.LENGTH + ego.LENGTH + vr_sd:
            dis_asp = 2
        elif dis < v1.LENGTH + ego.LENGTH + vr_sd + ego_sd:
            dis_asp = 3
        else:
            dis_asp = 4
        return Function("distance", [Function(get_asp_vehicle_repr(v1)), Function(get_asp_vehicle_repr(v2)), Number(dis_asp)])
    else:
        # v1在v2后面
        dis_asp = None
        dis = v2.position[0] - v1.position[0]
        vr_sd = rss_safe_distance(v1, ego)
        ego_sd = rss_safe_distance(ego, v2)
        if dis < v2.LENGTH:
            dis_asp = 0
        elif dis <  v2.LENGTH + ego.LENGTH:
            dis_asp = 1
        elif dis < v2.LENGTH + ego.LENGTH + vr_sd:
            dis_asp = 2
        elif dis < v2.LENGTH + ego.LENGTH + vr_sd + ego_sd:
            dis_asp = 3
        else:
            dis_asp = 4
        return Function("distance", [Function(get_asp_vehicle_repr(v2)), Function(get_asp_vehicle_repr(v1)), Number(dis_asp)])

def distance_asp_to_range(dis_asp:int, vf:Vehicle, vr:Vehicle, ego:Vehicle):
    vr_sd = rss_safe_distance(vr, ego)
    ego_sd = rss_safe_distance(ego, vf)
    distance_dict = {
        0: (0, vf.LENGTH),
        1: (vf.LENGTH, vf.LENGTH + ego.LENGTH),
        2: (vf.LENGTH + ego.LENGTH, vf.LENGTH + ego.LENGTH + vr_sd),
        3: (vf.LENGTH + ego.LENGTH + vr_sd, vf.LENGTH + ego.LENGTH + vr_sd + ego_sd),
        4: (vf.LENGTH + ego.LENGTH + vr_sd + ego_sd, vf.LENGTH + ego.LENGTH + vr_sd + ego_sd + 200)
    }
    return distance_dict[dis_asp]
