import numpy as np
import sparse
import os
import re
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import AbstractLane
from highway_env.road.road import Road
from data_collect.collect import Relation, get_relationship, get_surround_vehicles as get_surround_6
from planner.lattice_planner import cartesian_to_frenet_l1, get_refline, refline_project, get_state, get_surround_vehicles
from asp_decider.asp_utils import neighbour_vehicle_to_asp, road_network_to_asp, asp2str, get_asp_vehicle_repr

forecast_model = sparse.load_npz(os.path.join(os.path.dirname(__file__), "stat3.npz"))

def tsl_lane_repr_to_index(lane_repr: str):
    matches = re.match(r'f(\d+)t(\d+)l(\d+)', lane_repr)
    return tuple(map(int, matches.groups()))

class Model():
    def __init__(self, model, road:Road, ego:Vehicle):
        self.model = model
        self.prob = 1.0
        self.veh_dict = {}
        self.lat_relations = [None]
        self.lon_relations = [None]
        self.target_lane_index = [ego.lane_index]
        self.front_distances = [None]
        self.rear_distances = [None]
        
        states = {}
        init_relations = {}
        vehs = list(get_surround_vehicles(road, ego))
        for veh in vehs:
            if veh is None:
                continue
            state, relation = get_index(road, ego, veh)
            states[get_asp_vehicle_repr(veh)] = state
            self.veh_dict[get_asp_vehicle_repr(veh)] = veh
            init_relations[get_asp_vehicle_repr(veh)] = relation
        
        for answer in self.model[1:2]:
            lat_relations = {}
            lon_relations = {}
            front_distance = {}
            rear_distance = {}

            for atom in answer:
                if len(atom.arguments) == 1:
                    if atom.name == "cover_relation":
                        lon_relations[atom.arguments[0].name] = Relation.COVER
                    elif atom.name == "ahead_relation":
                        lon_relations[atom.arguments[0].name] = Relation.AHEAD
                    elif atom.name == "behind_relation":
                        lon_relations[atom.arguments[0].name] = Relation.BEHIND
                    elif atom.name == "left_relation":
                        lat_relations[atom.arguments[0].name] = Relation.LEFT
                    elif atom.name == "right_relation":
                        lat_relations[atom.arguments[0].name] = Relation.RIGHT
                    elif atom.name == "same_lane_relation":
                        lat_relations[atom.arguments[0].name] = Relation.SAME_LANE
                    elif atom.name == "ego_on_lane":
                        target_lane_index = tsl_lane_repr_to_index(atom.arguments[0].name)
                elif len(atom.arguments) == 3:
                    if atom.name == "distance":
                        if atom.arguments[0].name == get_asp_vehicle_repr(ego):
                            rear_distance[atom.arguments[1].name] = float(atom.arguments[2].number)
                        elif atom.arguments[1].name == get_asp_vehicle_repr(ego):
                            front_distance[atom.arguments[0].name] = float(atom.arguments[2].number)
            
            self.lat_relations.append(lat_relations)
            self.lon_relations.append(lon_relations)
            self.target_lane_index.append(target_lane_index)
            self.front_distances.append(front_distance)
            self.rear_distances.append(rear_distance)
            
            probs = np.empty((len(states), forecast_model.shape[-3]))
            for i,veh in enumerate(states.keys()):
                state = states[veh]
                init_relation = init_relations[veh]
                final_lat_relation = lat_relations.get(veh, Relation.SAME_LANE)
                final_lon_relation = lon_relations.get(veh, Relation.COVER)
                final_relation = Relation(final_lon_relation, final_lat_relation)
                # print(*state, init_relation, int(final_relation))
                v = forecast_model[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11], :, init_relation, int(final_relation)].todense()
                sum_v = forecast_model[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11], :, init_relation, :].todense()
                sum_v = sum_v.sum(axis=1).astype(np.float64)
                sum_v += 1e-6
                probs[i,:] = v/sum_v
                # print(v)
            self.prob *= probs.prod(axis=0).sum()
    
    def get_prob(self):
        return self.prob
    
    def get_sample_area(self, i, road:Road, ego:Vehicle):
        if i < 1 or i > len(self.lat_relations):
            raise ValueError("logical state index out of bound.")
        follow_vehs = []
        overtake_vehs = []
        cover_vehs = []
        front_vehs = []
        rear_vehs = []
        
        # print(self.lat_relations[i])
        # print(self.lon_relations[i])
        print({k: Relation(self.lon_relations[i][k], self.lat_relations[i].get(k, None)) for k, v in self.lon_relations[i].items()})
        
        refline = get_refline(ego, ego.lane)
        
        for veh_key in self.veh_dict.keys():
            veh = self.veh_dict[veh_key]
            if self.front_distances[i].get(veh_key, None) is not None:
                front_vehs.append((veh, self.front_distances[i].get(veh_key, None)))
            elif self.rear_distances[i].get(veh_key, None) is not None:
                rear_vehs.append((veh, self.rear_distances[i].get(veh_key, None)))
            # s, l, _, _, _ = cartesian_to_frenet_l1(
            #     refline_project(refline, veh.position), *(get_state(veh)[:4])
            # )
            # print(veh_key, s, l)
            # if self.lat_relations[i].get(veh_key, None) == Relation.LEFT:
            #     min_l.append(l+veh.WIDTH)
            # elif self.lat_relations[i].get(veh_key, None) == Relation.RIGHT:
            #     max_l.append(l-veh.WIDTH)
            # elif self.lat_relations[i].get(veh_key, None) == Relation.SAME_LANE:
            #     min_l.append(l-veh.WIDTH)
            #     max_l.append(l+veh.WIDTH)
            # if self.lon_relations[i].get(veh_key, None) == Relation.BEHIND:
            #     min_s.append(s + veh.LENGTH + 5.0)
            #     overtake_vehs.append(veh)
            # elif self.lon_relations[i].get(veh_key, None) == Relation.AHEAD:
            #     max_s.append(s - veh.LENGTH - 5.0)
            #     follow_vehs.append(veh)
            # elif self.lon_relations[i].get(veh_key, None) == Relation.COVER:
            #     # min_s.append(s - veh.LENGTH)
            #     # max_s.append(s + veh.LENGTH)
            #     cover_vehs.append(veh)
        l_offset = (self.target_lane_index[i][2] - ego.lane_index[2]) * 4.0
        # print("l bounds: ", (max(min_l), min(max_l)))
        # print("s bounds: ", (max(min_s), min(max_s)))
        # print("follow: ", follow_vehs)
        # print("overtake: ", overtake_vehs)
        # print("cover: ", cover_vehs)
        return  front_vehs, rear_vehs, (l_offset-0.5, l_offset+0.5)

def get_index(road, ego, veh):
    veh_surrounds = get_surround_6(road, veh)
    relation_with_ego = get_relationship(veh, ego)
    
    relate_speed_bins = [-5, 0, 5, 10, np.nan]
    back_relate_position_x_bins = [-200, -100, np.nan]
    front_relate_position_x_bins = [100, 200, np.nan]

    bins = [relate_speed_bins] * 6 + [front_relate_position_x_bins, back_relate_position_x_bins] * 3
    
    state = [
        veh_surrounds[0].speed - veh.speed if veh_surrounds[0] is not None else np.nan,
        veh_surrounds[1].speed - veh.speed if veh_surrounds[1] is not None else np.nan,
        veh_surrounds[2].speed - veh.speed if veh_surrounds[2] is not None else np.nan,
        veh_surrounds[3].speed - veh.speed if veh_surrounds[3] is not None else np.nan,
        veh_surrounds[4].speed - veh.speed if veh_surrounds[4] is not None else np.nan,
        veh_surrounds[5].speed - veh.speed if veh_surrounds[5] is not None else np.nan,
        veh_surrounds[0].position[0] - veh.position[0] if veh_surrounds[0] is not None else np.nan,
        veh_surrounds[1].position[0] - veh.position[0] if veh_surrounds[1] is not None else np.nan,
        veh_surrounds[2].position[0] - veh.position[0] if veh_surrounds[2] is not None else np.nan,
        veh_surrounds[3].position[0] - veh.position[0] if veh_surrounds[3] is not None else np.nan,
        veh_surrounds[4].position[0] - veh.position[0] if veh_surrounds[4] is not None else np.nan,
        veh_surrounds[5].position[0] - veh.position[0] if veh_surrounds[5] is not None else np.nan,
    ]
    state_index = [
        int(np.digitize(x, bins=bins[i])) for i, x in enumerate(state)
    ]
    return state_index, int(relation_with_ego)