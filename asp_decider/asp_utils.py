from highway_env.road.road import RoadNetwork
from clingo import Function, Symbol
from typing import List

def get_asp_lane_repr(from_: str, to_: str, id: int):
    return f"f{from_}t{to_}l{id}"

def get_asp_road_repr(from_: str, to_: str):
    return f"f{from_}t{to_}"

def asp2str(asp:List[Symbol], prefix:str=""):
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
        result.append(Function("has_lane", [Function(get_asp_road_repr(index[0], index[1])), Function(get_asp_lane_repr(*index))]))
        roads.add((index[0], index[1]))
        id_ = index[2]
        if id_ > 0:
            # result += f"left({get_asp_lane_repr(index[0], index[1], id_-1)},{get_asp_lane_repr(*index)}).\n"
            result.append(Function("left", [Function(get_asp_lane_repr(index[0], index[1], id_-1)), Function(get_asp_lane_repr(*index))]))
    for road in roads:
        # result += f"is_road({get_asp_road_repr(*road)}).\n"
        result.append(Function("is_road", [Function(get_asp_road_repr(*road))]))
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
            initial.append(get_lon_relation(vehicles[i], vehicles[j]))
    for vehicle in vehicles:
        initial.append(Function("on_lane", [Function(get_asp_vehicle_repr(vehicle)), Function(get_asp_lane_repr(*vehicle.lane_index))]))
    return always, initial