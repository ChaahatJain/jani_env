#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random

from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

random.seed(2020)


class OneWayLineModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--tank-capacity", type=int, default=-1, help="Initial/maximal tank.")
        self.optionParser.add_argument("--dropping-prob", type=float, default=0, help="Probability to drop a package.")
        self.optionParser.add_argument("--slipping-prob", type=float, default=0, help="Probability to slip on road when picking/dropping a pacage.")
        self.optionParser.add_argument("--icy-prob", type=float, default=0, help="Probability to slip on icy roads in the middle.")
        self.optionParser.add_argument("--add-parking", type=int, default=0, help="Add the parking action.")
        self.optionParser.add_argument("--fail-dec-on-ice", type=int, default=0, help="Fail decelerating on ice. Guarantees no safe policies for the benchmark.")
        # options for property generation (and thus not saved in model file):
        self.optionParser.add_argument("--single-safe-start-value", action="store_true", default=False, help="Fix safety variable to a single safe value at start.")
        self.optionParser.add_argument("--fix-truck-start", action="store_true", default=False, help="Trucks are at default position at start.")
        self.optionParser.add_argument("--fix-package-start", action="store_true", default=False, help="Packages are at default locations at start.")
        self.optionParser.add_argument("--zero-load-start", action="store_true", default=False, help="No packages loaded at start (which is -- at documentation time -- subsumed by --fix-package-start).")


class OneWayLineModelGenerator(JaniModelGeneratorPddlInJani):
    # customization for linetrack
    LEFT_TO_BRIDGE = True
    # constants:
    NUM_LOCATIONS = "num_locations"
    NUM_PACKAGES = "num_packages"
    TANK_CAPACITY_NAME = "tank_capacity"
    DROPPING_PROB_NAME = "dropping_prob"
    SLIPPING_PROB_NAME = "slipping_prob"
    ICY_PROB_NAME = "icy_prob"
    PARKING_FLAG = "add_parking"
    FAIL_DEC_ON_ICE_FLAG = "fail_dec_on_ice"


    def add_park_action(self):
        return self.add_parking > 0

    def use_fuel(self):
        return self.tank_capacity > 0

    def fail_dec_action_on_icy(self):
        return self.fail_dec_on_ice > 0

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.special_predicates = "_truck_pos_predicates_first" in self.property_type
        self.terminal_at_unsafe_jani = False

        if self.model_file is None:
            # model generation
            self.description_name = options.description
            self.tank_capacity = options.tank_capacity
            self.dropping_prob = options.dropping_prob
            self.slipping_prob = options.slipping_prob
            self.icy_prob = options.icy_prob
            self.add_parking = options.add_parking
            self.fail_dec_on_ice = options.fail_dec_on_ice
        else:
            # property generation
            constants = self.model["constants"]
            _, self.dropping_prob = self.read_constant(constants, self.DROPPING_PROB_NAME)
            _, self.slipping_prob = self.read_constant(constants, self.SLIPPING_PROB_NAME)
            _, self.icy_prob = self.read_constant(constants, self.ICY_PROB_NAME)
            _, self.tank_capacity = self.read_constant(constants, self.TANK_CAPACITY_NAME)
            self.description_name = self.model["name"]
            _, self.add_parking = self.read_constant(constants, self.PARKING_FLAG)
            _, self.fail_dec_on_ice = self.read_constant(constants, self.FAIL_DEC_ON_ICE_FLAG)
            #
        self.single_safe_start_value = options.single_safe_start_value
        self.fix_truck_start = options.fix_truck_start
        self.fix_package_start = options.fix_package_start
        self.zero_load_start = options.zero_load_start

        print(self.dropping_prob)
        self.model_type = JaniModelType.MDP if self.sample_in_model or self.dropping_prob > 0 or self.slipping_prob > 0 or self.icy_prob > 0 else JaniModelType.LTS
        self.model_name = self.description_name
        self.description = PythonUtils.load_json(self.description_name)
        self.num_locations = len(self.description["locations"])
        self.num_trucks = len(self.description["trucks"])
        self.num_packages = len(self.description["packages"])
        self.goal_location = self.description["goal"]
        self.icy_locations = [loc for loc in self.description["icy"]]
        self.max_speed = int(self.description["speed"])

        # variables:
        self.location_indexes = list(range(0, self.num_locations))
        self.location_values = list(self.location_indexes)  # per location index, the corresponding truck domain value
        self.truck_indexes = list(range(0, self.num_trucks))
        #
        self.location_load = dict([(loc, "location_load_" + str(loc)) for loc in self.location_indexes])
        self.truck_vars = dict([(truck, "truck_" + str(truck)) for truck in self.truck_indexes])
        self.truck_velocities = dict([(truck, "truck_vel_" + str(truck)) for truck in self.truck_indexes])
        self.truck_load = dict([(truck, "truck_load_" + str(truck)) for truck in self.truck_indexes])
        self.aux_vel = "aux_vel"
        self.variable_names = list(self.location_load.values()) + list(self.truck_vars.values()) + list(self.truck_load.values()) + list(self.truck_velocities.values()) + [self.aux_vel]
        if self.use_fuel():
            self.truck_tank = dict([(truck, "truck_tank_" + str(truck)) for truck in self.truck_indexes])
            self.variable_names += list(self.truck_tank.values())
        if self.add_park_action():
            self.park_vars = dict([(truck, "parked_" + str(truck)) for truck in self.truck_indexes])
            self.variable_names +=list(self.park_vars.values())
        # aux for vars:
        self.location_domain_size = self.num_packages + 1
        self.location_upper_bound = self.location_domain_size - 1
        self.truck_domain_size = self.num_locations
        self.truck_upper_bound = self.truck_domain_size - 1
        self.truck_capacities = dict([(truck, truck_item["capacity"]) for (truck, truck_item) in zip(self.truck_indexes, self.description["trucks"])])
        self.vel_lower_bound = 0
        self.vel_upper_bound = self.max_speed
        if self.use_fuel():
            self.tank_lower_bound = - self.num_packages  # for now we subtract one tank unit per package and road

        # actions
        self.action_labels_pick_up = dict([(truck, "pick_up_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_drop = dict([(truck, "drop_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_acc = dict([(truck, "acc_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_dec = dict([(truck, "dec_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_move = dict([(truck, "move_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels = list(self.action_labels_pick_up.values()) + list(self.action_labels_drop.values()) + list(self.action_labels_acc.values()) + list(self.action_labels_dec.values()) + list(self.action_labels_move.values()) 
        if self.add_park_action():
            self.action_labels_park = dict([(truck, "park_" + self.truck_vars[truck]) for truck in self.truck_indexes])
            self.action_labels += list(self.action_labels_park.values())
        self.compute_model_initial_and_goal_state()
        self.load_roads()

        if self.sample_in_model:
            JaniModelGeneratorPddlInJani.LOCATION_NAMES = [self.LOCATION_NAME]
            self.sample_truck_location = "sample_truck"
            JaniModelGeneratorPddlInJani.LOCATION_NAMES += [self.sample_truck_location]
            self.sample_load = "sample_load"
            JaniModelGeneratorPddlInJani.LOCATION_NAMES += [self.sample_load]
            JaniModelGeneratorPddlInJani.INITIAL_LOCATIONS = [self.sample_truck_location]

    # noinspection PyAttributeOutsideInit
    def compute_model_initial_and_goal_state(self):
        # initial
        self.initial_state = dict([(loc, 0) for loc in self.location_load.values()]
                                  + [(load, 0) for load in self.truck_load.values()]
                                  + [(vel, 0) for vel in self.truck_velocities.values()]
                                  + [(self.aux_vel, 1)]
                                  + ([(self.truck_tank[truck], self.tank_capacity) for truck in self.truck_indexes] if self.use_fuel() else [])
                                  + ([(self.park_vars[truck], 0) for truck in self.truck_indexes] if self.add_park_action() else []))
        # goal
        self.goal_state = dict([(loc, 0) for loc in self.location_load.values()])  # + [(self.unsafety_index, False)])

        # start-end trucks
        for (truck, truck_start_end) in zip(self.truck_indexes, self.description["trucks"]):
            self.initial_state[self.truck_vars[truck]] = truck_start_end["start"]
            if "end" in truck_start_end:  # trucks may have no goal position
                self.goal_state[self.truck_vars[truck]] = truck_start_end["end"]

        # start-end locations of packages
        for loc_start_end in self.description["packages"]:  # goal location loads specified by index ...
            self.initial_state[self.location_load[self.location_indexes[loc_start_end["start"]]]] += (0 if self.sample_in_model else 1)
            self.goal_state[self.location_load[self.location_indexes[loc_start_end["goal"]]]] += 1

    # noinspection PyAttributeOutsideInit
    def load_roads(self):
        roads = self.description["locations"]
        self.roads = list()
        self.road_bounds = dict()
        bounds = self.description["bounds"]
        for bound in bounds:
            src = self.location_indexes[bound["id"]]
            b = bound["bound"]
            self.road_bounds[src] = b
        for roads_item in roads:
            src_loc_index = self.location_indexes[roads_item["id"]]
            for road in roads_item["roads"]:
                target_loc_index = self.location_indexes[road["to"]]
                self.roads.append((src_loc_index, target_loc_index))

    def has_road(self, src_location_index: int, target_location_index: int) -> bool:
        return (src_location_index, target_location_index) in self.roads

    def has_road_bound(self, src_location_index: int) -> bool:
        return src_location_index in self.road_bounds
    
    def get_road_bound(self, src_location_index: int) -> int:
        if not (self.has_road_bound(src_location_index)):
            return self.vel_upper_bound + 1
        return self.road_bounds[src_location_index]
        
    # constraints #############################################################################################

    def truck_is_parked(self, truck: int) -> json:
        return Je.Ge(self.park_vars[truck], 1)

    def truck_is_not_parked(self, truck: int) -> json:
        return Je.Le(self.park_vars[truck], 0)

    def location_has_package(self, loc: int) -> json:
        return Je.Ge(self.location_load[loc], 1)

    def location_has_no_package(self, loc: int) -> json:
        return Je.Le(self.location_load[loc], 0)

    def truck_is_at(self, truck: int, loc: int) -> json:
        return Je.Eq(self.truck_vars[truck], self.location_values[loc])

    def truck_is_not_at(self, truck: int, loc: int) -> json:
        return Je.Ne(self.truck_vars[truck], self.location_values[loc])

    def truck_has_capacity(self, truck: int) -> json:
        return Je.Le(self.truck_load[truck], self.truck_capacities[truck] - 1)

    def truck_has_package(self, truck: int) -> json:
        return Je.Ge(self.truck_load[truck], 1)
    
    def truck_has_no_package(self, truck: int) -> json:
        return Je.Le(self.truck_load[truck], 0)

    def is_unsafe(self) -> json:
        return Je.Le(self.aux_vel, -1)

    def is_not_unsafe(self) -> json:
        return Je.Ge(self.aux_vel, 0)

    def has_tank(self, truck: int) -> json:
        return Je.Ge(self.truck_tank[truck], 0)  # self.truck_load[truck]

    def has_no_tank(self, truck: int) -> json:
        return Je.Le(self.truck_tank[truck], -1)  # Je.Add(self.truck_load[truck], -1)

    def can_accelerate(self, truck: int) -> json: 
        return Je.Le(self.truck_velocities[truck], self.vel_upper_bound - 1)
    
    def can_decelerate(self, truck: int) -> json:
        return Je.Ge(self.truck_velocities[truck], self.vel_lower_bound + 1)
    
    def has_no_velocity(self, truck: int) -> json:
        return Je.Le(self.truck_velocities[truck], 0)
    
    def has_some_velocity(self, truck: int) -> json:
        return Je.Ge(self.truck_velocities[truck], 1)
    
    def will_crash_right(self, truck: int, bound: int = 1) -> json:
        return Je.Ge(self.truck_velocities[truck], bound)
    
    def will_crash_left(self, truck: int, bound: int = -1) -> json: 
        return Je.Le(self.truck_velocities[truck], bound)
    
    def wont_crash_right(self, truck: int, bound: int) -> json:
        return Je.Le(self.truck_velocities[truck], bound - 1)
    
    def wont_crash_left(self, truck: int, bound: int) -> json:
        return Je.Ge(self.truck_velocities[truck], bound + 1)
    
    def cannot_accelerate(self, truck: int) -> json:
        return Je.Ge(self.truck_velocities[truck], self.vel_upper_bound)
    
    # assignments

    def park_truck(self, truck: int) -> json:
        return JaniStructureGenerator.generate_assignment(self.park_vars[truck], 1)

    def inc_location_load(self, loc: int, inc: int = 1) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.location_load[loc], inc)

    def inc_velocity(self, truck: int, inc: int = 1) -> json: 
        return JaniStructureGenerator.generate_self_assignment(self.truck_velocities[truck], inc)
    
    def inc_truck_load(self, truck: int, inc: int = 1) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.truck_load[truck], inc)

    def set_aux_vel(self, truck: int, target_loc: int, acc: int) -> json: 
        if target_loc == 0:
            return JaniStructureGenerator.generate_assignment(self.aux_vel, Je.Add(self.truck_velocities[truck], acc))
        if target_loc == self.num_locations - 1:
            return JaniStructureGenerator.generate_assignment(self.aux_vel, Je.Mult(-1, Je.Add(self.truck_velocities[truck], acc)))
        return JaniStructureGenerator.generate_assignment(self.aux_vel, 1)
    
    def update_drive_tank(self, truck: int) -> json:
        # for now we subtract one tank unit per package and path:
        return JaniStructureGenerator.generate_self_assignment(self.truck_tank[truck], Je.Mult(-1, self.truck_load[truck]))
    
    def update_package_tank(self, truck: int) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.truck_tank[truck], -1)

    def pick_up(self, truck: int, loc: int):
        return [self.inc_location_load(loc, -1), self.inc_truck_load(truck, 1)] + ([self.update_package_tank(truck)] if self.use_fuel() else [])

    def drop(self, truck: int, loc: int):
        return [self.inc_location_load(loc, 1), self.inc_truck_load(truck, -1)] + ([self.update_package_tank(truck)] if self.use_fuel() else [])
    
    def drop_fwd(self, truck: int, loc: int):
        if loc == len(self.location_load) -1:
            return self.drop(truck, loc)
        return [self.inc_location_load(loc + 1, 1), self.inc_truck_load(truck, -1)] + ([self.update_package_tank(truck)] if self.use_fuel() else [])
    
    
    def slip(self, truck: int):
        return [JaniStructureGenerator.generate_self_assignment(self.truck_vars[truck], 1)]
    
    def unsafe_move(self) -> json:
        return [JaniStructureGenerator.generate_assignment(self.aux_vel, -1)]

    def move(self, truck: int):
       return [JaniStructureGenerator.generate_self_assignment(self.truck_vars[truck], self.truck_velocities[truck])]
    
    def accelerate(self, truck: int):
        return [JaniStructureGenerator.generate_self_assignment(self.truck_vars[truck], self.truck_velocities[truck]), self.inc_velocity(truck)]

    def decelerate(self, truck: int):
        return [JaniStructureGenerator.generate_self_assignment(self.truck_vars[truck], self.truck_velocities[truck]), self.inc_velocity(truck, -1)]

    def park(self, truck: int):
        return [self.park_truck(truck)]

    # model generation #################################################################################################

    def generate_location_variable(self, loc: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.location_load[loc], 0, self.location_upper_bound, self.initial_state[self.location_load[loc]])

    def generate_truck_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_vars[truck], 0, self.truck_upper_bound, self.initial_state[self.truck_vars[truck]])

    def generate_truck_capacity_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_load[truck], 0, self.truck_capacities[truck], self.initial_state[self.truck_load[truck]])

    def generate_aux_velocity_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.aux_vel, -1, self.vel_upper_bound, self.initial_state[self.aux_vel])

    def generate_truck_velocity_variable(self, truck: int): 
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_velocities[truck], self.vel_lower_bound, self.vel_upper_bound, self.initial_state[self.truck_velocities[truck]])

    def generate_truck_tank_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_tank[truck], self.tank_lower_bound, self.tank_capacity, self.initial_state[self.truck_tank[truck]])

    def generate_parked_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.park_vars[truck], 0, 1, 0)

    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.NUM_LOCATIONS, JaniStructureGenerator.generate_int_type(), self.num_locations)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.NUM_PACKAGES, JaniStructureGenerator.generate_int_type(), self.num_packages)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.DROPPING_PROB_NAME, JaniStructureGenerator.generate_real_type(), self.dropping_prob)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.SLIPPING_PROB_NAME, JaniStructureGenerator.generate_real_type(), self.slipping_prob)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.ICY_PROB_NAME, JaniStructureGenerator.generate_real_type(), self.icy_prob)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.TANK_CAPACITY_NAME, JaniStructureGenerator.generate_int_type(), self.tank_capacity)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.PARKING_FLAG, JaniStructureGenerator.generate_int_type(), self.add_parking)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.FAIL_DEC_ON_ICE_FLAG, JaniStructureGenerator.generate_int_type(), self.fail_dec_on_ice)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.TERMINAL_AT_UNSAFE, JaniStructureGenerator.generate_bool_type(), self.terminal_at_unsafe_jani)]

    def generate_variables(self):
        variables = [self.generate_location_variable(loc) for loc in self.location_indexes]
        variables += [self.generate_truck_variable(truck) for truck in self.truck_indexes]
        variables += [self.generate_truck_capacity_variable(truck) for truck in self.truck_indexes]
        variables += [self.generate_truck_velocity_variable(truck) for truck in self.truck_indexes]
        variables += [self.generate_truck_tank_variable(truck) for truck in self.truck_indexes if self.use_fuel()]
        variables += [self.generate_parked_variable(truck) for truck in self.truck_indexes if self.add_park_action()]
        variables += [self.generate_aux_velocity_variable()]
        return variables

    #

    def generate_edge_aux(self, destinations: list, action: str, guard) -> json:
        if self.terminal_at_unsafe_jani:
            guard = Je.And(self.is_not_unsafe(), guard)
        return JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, guard)
    
    def generate_edges(self):
        def generate_crash_edge(truck: int, src_loc: int, action): 
            bound = self.get_road_bound(src_loc)
            guard_crash = Je.And(self.truck_is_at(truck, src_loc), self.will_crash_right(truck, bound))
            if self.add_park_action():
                guard_crash = Je.And(self.truck_is_not_parked(truck), guard_crash)
            destination_crash = [self.generate_destination_aux(self.unsafe_move(), None)]
            edge_crash = self.generate_edge_aux(destination_crash, action, guard_crash)
            return edge_crash
        
        def generate_move_icy_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.accelerate(truck), self.icy_prob)
            destination_drop = self.generate_destination_aux(self.move(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.move(truck), 1 - self.dropping_prob - self.icy_prob)
            destinations = [destination_icy, destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_icy_cant_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_no_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.accelerate(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.move(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]  
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_drop_cant_icy_edge(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.truck_has_package(truck), self.cannot_accelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_drop = self.generate_destination_aux(self.move(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.move(truck), 1 - self.dropping_prob)
            destinations = [destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_move_cant_drop_cant_icy_edge(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.truck_has_no_package(truck), self.cannot_accelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.move(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_icy_only_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.can_accelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.accelerate(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.move(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]  
            return self.generate_edge_aux(destinations, action, guard)

        def generate_move_cant_icy_only_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.cannot_accelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.move(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_drop = self.generate_destination_aux(self.move(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.move(truck), 1 - self.dropping_prob)
            destinations = [destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_cant_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.truck_has_no_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.move(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_move_det_edge(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_some_velocity(truck), self.wont_crash_right(truck, bound), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.move(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)


        ########### Parking

        def generate_park_truck_edge(truck: int, src_loc: int):
            action = self.action_labels_park[truck]
            assert(src_loc == self.num_locations - 1)
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.has_no_velocity(truck))
            guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination = [self.generate_destination_aux(self.park(truck), None)]
            return [self.generate_edge_aux(destination, action, guard)]

        
        ###################################################### ACCELERATE ###############################
        def generate_acc_icy_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_package(truck), self.has_some_velocity(truck))
            guard2 = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_package(truck), self.has_no_velocity(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
                guard2 = Je.And(self.truck_is_not_parked(truck), guard2)

            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob) # acceleration has a chance to fail on icy regions
            destination_drop = self.generate_destination_aux(self.accelerate(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), 1 - self.dropping_prob - self.icy_prob)
            destinations = [destination_icy, destination_drop, destination_normal]
            d_drop = self.generate_destination_aux(self.accelerate(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            d_normal = self.generate_destination_aux(self.accelerate(truck), 1 - self.dropping_prob)
            d = [d_drop, d_normal]
            return [self.generate_edge_aux(destinations, action, guard), self.generate_edge_aux(d, action, guard2)]
        
        def generate_acc_icy_cant_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_no_package(truck), self.has_some_velocity(truck))
            guard2 = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_no_package(truck), self.has_no_velocity(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
                guard2 = Je.And(self.truck_is_not_parked(truck), guard2)
            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]
            d_normal = self.generate_destination_aux(self.accelerate(truck), 1)
            return [self.generate_edge_aux(destinations, action, guard), self.generate_edge_aux([d_normal], action, guard2)]
        
        def generate_acc_icy_only_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.has_some_velocity(truck))
            guard2 = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck),  self.has_no_velocity(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
                guard2 = Je.And(self.truck_is_not_parked(truck), guard2)
            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]
            d_normal = self.generate_destination_aux(self.accelerate(truck), 1)
            return [self.generate_edge_aux(destinations, action, guard), self.generate_edge_aux(d_normal, action, guard2)]
        
        def generate_acc_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_drop = self.generate_destination_aux(self.accelerate(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), 1 - self.dropping_prob)
            destinations = [destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_acc_cant_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck), self.truck_has_no_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_acc_det_edge(truck: int, src_loc: int):
            action = self.action_labels_acc[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_accelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.accelerate(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        ################################################## DECELERATE ##############################################
        def generate_dec_icy_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob)
            destination_drop = self.generate_destination_aux(self.decelerate(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), 1 - self.dropping_prob - self.icy_prob)
            destinations = [destination_icy, destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_dec_icy_cant_drop_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck), self.truck_has_no_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]  
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_dec_icy_only_edge(truck: int, src_loc: int):
            assert(src_loc in self.icy_locations)
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_icy = self.generate_destination_aux(self.move(truck), self.icy_prob)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), 1 - self.icy_prob)
            destinations = [destination_icy, destination_normal]  
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_dec_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck), self.truck_has_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_drop = self.generate_destination_aux(self.decelerate(truck) + self.drop_fwd(truck, src_loc), self.dropping_prob)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), 1 - self.dropping_prob)
            destinations = [destination_drop, destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_dec_cant_drop_only_edge(truck: int, src_loc: int):
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck), self.truck_has_no_package(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        
        def generate_dec_det_edge(truck: int, src_loc: int):
            action = self.action_labels_dec[truck]
            bound = self.get_road_bound(src_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.wont_crash_right(truck, bound), self.can_decelerate(truck))
            if self.add_park_action():
                guard = Je.And(self.truck_is_not_parked(truck), guard)
            destination_normal = self.generate_destination_aux(self.decelerate(truck), None)
            destinations = [destination_normal]
            return self.generate_edge_aux(destinations, action, guard)
        ############################################### PICK AND DROP ######################################
        
        def generate_pick_up_edge(truck: int, loc: int):
            action = self.action_labels_pick_up[truck]
            guard = Je.And(self.truck_is_at(truck, loc), self.truck_has_capacity(truck), self.location_has_package(loc), self.has_no_velocity(truck))
            destinations = [self.generate_destination_aux(self.pick_up(truck, loc), 1 - self.slipping_prob), self.generate_destination_aux(self.slip(truck), self.slipping_prob)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_drop_edge(truck: int, loc: int):
            action = self.action_labels_drop[truck]
            guard = Je.And(self.truck_is_at(truck, loc), self.truck_has_package(truck), self.has_no_velocity(truck))
            destinations = [self.generate_destination_aux(self.drop(truck, loc), 1 - self.slipping_prob), self.generate_destination_aux(self.slip(truck), self.slipping_prob)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_move(truck: int, src_loc: int):
            action = self.action_labels_move[truck]
            edges = []
            if self.has_road_bound(src_loc):
                edge_crash = generate_crash_edge(truck, src_loc, action) # if crash possible, then one edge should be for crashing
                edges.append(edge_crash)
            if self.icy_prob == 0 and self.dropping_prob == 0:
                edge_det = generate_move_det_edge(truck, src_loc) # moving safely edge
                edges.append(edge_det)
                return edges
            elif (self.icy_prob > 0 and self.dropping_prob == 0):
                if src_loc in self.icy_locations: 
                    edge_icy = generate_move_icy_only_edge(truck, src_loc) # if on ice, and cannot drop, then non-deterministically accelerate
                    edge_cant_icy = generate_move_cant_icy_only_edge(truck, src_loc) # if on ice, and cannot drop, but truck too fast to accelerate.
                    edges += [edge_icy, edge_cant_icy]
                    return edges
                else:
                    edge_det = generate_move_det_edge(truck, src_loc) # if not on ice, then just move normally.
                    edges.append(edge_det)
                    return edges
            elif (self.dropping_prob > 0 and self.icy_prob == 0):
                edge_cant_drop = generate_move_cant_drop_only_edge(truck, src_loc) # if truck has no package, just move
                edge_drop = generate_move_drop_only_edge(truck, src_loc) # if truck has package, then non-deterministically drop it
                edges += [edge_drop, edge_cant_drop]
                return edges
            else: # both icy and drop
                if src_loc in self.icy_locations:
                    edge_icy_drop = generate_move_icy_drop_edge(truck, src_loc) # possible to drop and ice
                    edge_icy_no_drop = generate_move_icy_cant_drop_edge(truck, src_loc) # cannot drop because truck has no package
                    edge_drop_no_icy = generate_move_drop_cant_icy_edge(truck, src_loc) # cannot ice because truck moving at max speed
                    edge_no_icy_no_drop = generate_move_cant_drop_cant_icy_edge(truck, src_loc) # cannot ice because truck at max speed and cannot drop because truck has no package
                    edges = edges + [edge_icy_drop, edge_icy_no_drop, edge_drop_no_icy, edge_no_icy_no_drop]
                    return edges
                else: # no ice so can only drop
                    edge_cant_drop = generate_move_cant_drop_only_edge(truck, src_loc)
                    edge_drop = generate_move_drop_only_edge(truck, src_loc)
                    edges += [edge_drop, edge_cant_drop]
                    return edges 
            
        def generate_accelerate(truck: int, src_loc: int):
            action = self.action_labels_acc[truck]
            edges = []
            if self.has_road_bound(src_loc):
                edge_crash = generate_crash_edge(truck, src_loc, action) # if crash possible, then add it as an edge
                edges.append(edge_crash)
            if self.icy_prob == 0 and self.dropping_prob == 0:
                edge_det = generate_acc_det_edge(truck, src_loc) # accelerate normally
                edges.append(edge_det)
                return edges
            elif (self.icy_prob > 0 and self.dropping_prob == 0):
                if src_loc in self.icy_locations: 
                    edge_icy = generate_acc_icy_only_edge(truck, src_loc) # if on ice and cannot drop, then non-deterministically move. Truck can accelerate.
                    for edge in edge_icy:
                        edges.append(edge)
                    return edges
                else:
                    edge_det = generate_acc_det_edge(truck, src_loc) # accelerate normally when not on ice
                    edges.append(edge_det)
                    return edges
            elif (self.dropping_prob > 0 and self.icy_prob == 0):
                edge_cant_drop = generate_acc_cant_drop_only_edge(truck, src_loc) # if can drop, but truck has no package
                edge_drop = generate_acc_drop_only_edge(truck, src_loc) # if can drop and truck has package
                edges += [edge_drop, edge_cant_drop]
                return edges
            else: # both icy and drop
                if src_loc in self.icy_locations:
                    edge_icy_drop = generate_acc_icy_drop_edge(truck, src_loc) # if on ice, and can drop
                    edge_icy_no_drop = generate_acc_icy_cant_drop_edge(truck, src_loc) # if on ice, but cannot drop because truck has no package
                    for edge in edge_icy_drop:
                        edges.append(edge)
                    for edge in edge_icy_no_drop:
                        edges.append(edge)
                    return edges
                else: # no ice so can only drop
                    edge_drop = generate_acc_drop_only_edge(truck, src_loc) # can drop because truck has package
                    edge_cant_drop = generate_acc_cant_drop_only_edge(truck, src_loc) # truck does not have package
                    edges += [edge_drop, edge_cant_drop]
                    return edges
            
        def generate_decelerate(truck: int, src_loc: int):
            action = self.action_labels_dec[truck]
            edges = []
            if self.has_road_bound(src_loc):
                edge_crash = generate_crash_edge(truck, src_loc, action) # add edge if crashing possible
                edges.append(edge_crash)
            if self.icy_prob == 0 and self.dropping_prob == 0:
                edge_det = generate_dec_det_edge(truck, src_loc) # deterministically decelerate
                edges.append(edge_det)
                return edges
            elif (self.icy_prob > 0 and self.dropping_prob == 0):
                if self.add_park_action() or self.fail_dec_action_on_icy():
                    if src_loc in self.icy_locations:
                        edge_icy = generate_dec_icy_only_edge(truck, src_loc) # if on icy place, and can decelerate then do so.
                        edges.append(edge_icy)
                        return edges
                    else:
                        edge_det = generate_dec_det_edge(truck, src_loc) # if not on icy region, then deterministically decelerate
                        edges.append(edge_det)
                        return edges
                else:
                    edge_det = generate_dec_det_edge(truck, src_loc)
                    edges.append(edge_det)
                    return edges
            elif (self.dropping_prob > 0 and self.icy_prob == 0):
                edge_drop = generate_dec_drop_only_edge(truck, src_loc) # if can drop, then truck non-determistically drop so
                edge_cant_drop = generate_dec_cant_drop_only_edge(truck, src_loc)
                edges += [edge_drop, edge_cant_drop]
                return edges
            else: # both icy and drop
                if self.add_park_action() or self.fail_dec_action_on_icy():
                    if src_loc in self.icy_locations: # can always ice
                        edge_icy_drop = generate_dec_icy_drop_edge(truck, src_loc) # possible to drop and ice
                        edge_icy_no_drop = generate_dec_icy_cant_drop_edge(truck, src_loc) # cannot drop because truck has no package
                        edges = edges + [edge_icy_drop, edge_icy_no_drop]
                        return edges
                    else: # no ice so can only drop
                        edge_drop = generate_dec_drop_only_edge(truck, src_loc)
                        edge_cant_drop = generate_dec_cant_drop_only_edge(truck, src_loc)
                        edges += [edge_drop, edge_cant_drop]
                        return edges
                else:
                    edge_drop = generate_dec_drop_only_edge(truck, src_loc)
                    edge_cant_drop = generate_dec_cant_drop_only_edge(truck, src_loc)
                    edges += [edge_drop, edge_cant_drop]
                    return edges

            

        edges = [generate_pick_up_edge(truck, loc) for loc in self.location_indexes  for truck in self.truck_indexes if loc != self.goal_location ]
        edges += [generate_drop_edge(truck, loc) for loc in self.location_indexes for truck in self.truck_indexes if loc == self.goal_location ]
        for truck in self.truck_indexes:
            for src in self.location_indexes:
                edges += generate_accelerate(truck, src)
                edges += generate_decelerate(truck, src)
                edges += generate_move(truck, src)
                if (src == self.num_locations - 1) and self.add_park_action():
                   edges += generate_park_truck_edge(truck, src)
        return edges

    # property generation ##############################################################################################

    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], Je.And(self.generate_goal_expression(), self.is_not_unsafe()))
        return JaniStructureGenerator.generate_objective_expression(goal=goal, goal_potential=self.generate_goal_potential())

    def generate_goal_potential(self) -> json:
        if not self.use_goal_potential:
            return None
        additive_list = list()
        # rewards for packages at goal
        for loc in self.location_indexes:
            if self.goal_state[self.location_load[loc]] > 0:
                additive_list.append(Je.Mult(10, Je.min(self.goal_state[self.location_load[loc]], self.location_load[loc])))
        # rewards depending on load (if #packages at current position is satisfied)
        for truck in self.truck_indexes:
            for loc in self.location_indexes:
                max_cap_for_move = self.num_packages
                # negative reward if more packages are load than max cap
                if max_cap_for_move < self.num_packages:  # exclude special case where there is no capacity limitation
                    additive_list.append(Je.Ite(self.truck_is_at(truck, loc), Je.Mult(-10, Je.min(0, Je.Sub(self.truck_load[truck], max_cap_for_move))), 0))
                # loading at least one package is required towards the goal, if goal-number of packages is delivered at current loc ...
                if self.goal_state[self.location_load[loc]] < self.num_packages:  # exclude special case where all packages are required at the location
                    reward_con = [self.truck_is_at(truck, loc), Je.Ge(self.truck_load[truck], 1)] + ([Je.Ge(self.location_load[loc], self.goal_state[self.location_load[loc]])] if self.goal_state[self.location_load[loc]] > 0 else [])
                    additive_list.append(Je.Ite(Je.And(reward_con), 10, 0))
        goal_potential = Je.Add(additive_list)
        goal_potential = Je.Ite(self.generate_start(), 0, goal_potential) if self.ground_start_potential else goal_potential
        goal_potential = Je.Ite(Je.Or(Je.And(self.generate_goal_expression(), self.is_not_unsafe()), self.generate_reach()), 0, goal_potential) if self.ground_terminal_potential else goal_potential
        return goal_potential

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.is_unsafe())

    # starts ###########################################################################################################

    def generate_start(self):
        constraints = list()

        # safe at start!!!
        constraints.append(self.is_not_unsafe())
        constraints.append(Je.And([self.has_no_velocity(truck) for truck in self.truck_indexes]))
        # full tank at start
        constraints += [Je.Eq(tank, self.tank_capacity) for tank in self.truck_tank.values()] if self.use_fuel() else []
        constraints += [Je.Eq(park, 0) for park in self.park_vars.values()] if self.add_park_action() else []
        # truck can be at any position
        # truck load may not exceed capacity: holds by variable bounds
        # packet can be at any position
        # load sum over locations and trucks must equal total number of packages
        constraints.append(Je.Eq(self.num_packages, Je.Add([self.location_load[loc] for loc in self.location_indexes] + [self.truck_load[truck] for truck in self.truck_indexes])))

        # OPTIONALS
        if self.fix_truck_start:
            constraints += [Je.Eq(self.truck_vars[truck], self.initial_state[self.truck_vars[truck]]) for truck in self.truck_indexes]
        if self.fix_package_start:
            constraints += [Je.Eq(self.location_load[loc], self.initial_state[self.location_load[loc]]) for loc in self.location_indexes]
        elif self.zero_load_start:
            constraints += [Je.Eq(self.truck_load[truck], 0) for truck in self.truck_indexes]

        if self.LEFT_TO_BRIDGE:  # customized for line track, #starts = (#packages + #locations + 1 - 1| #packages) * (#locations)
            offset = self.num_locations - 2
            constraints.append(Je.Le(self.truck_vars[self.truck_indexes[0]], offset))  # "left" of bridge
            constraints += [Je.Eq(self.location_load[loc], self.initial_state[self.location_load[loc]]) for loc in self.location_indexes if loc >= offset + 1]
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, number_starts: int) -> list:

        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=number_starts, default_state=self.initial_state)

        random_loads = list(self.truck_load.values()) + list(self.location_load.values())

        # 25 %: 25% of packages on truck
        while states_values.size() < int(number_starts * 0.25):
            candidate = dict([(self.truck_vars[truck], 0) for truck in self.truck_indexes] + [(self.truck_velocities[truck], 0) for truck in self.truck_indexes] + [(self.aux_vel, 0)] + ([(self.park_vars[truck], 0) for truck in self.truck_indexes] if self.add_park_action() else []))
            # packages on truck
            rem_packages_on_truck = int(self.num_packages * 0.25)
            packages_on_truck = 0
            start_packages_mapping = dict([(var, 0) for var in list(self.truck_load.values()) + list(self.location_load.values())])
            truck_indexes_perm = list(self.truck_indexes)
            random.shuffle(truck_indexes_perm)
            while rem_packages_on_truck > 0 and sum(start_packages_mapping.values()) < sum(self.truck_capacities.values()):
                for truck in truck_indexes_perm:
                    truck_load = self.truck_load[truck]
                    inc = random.randint(0, min(rem_packages_on_truck, self.truck_capacities[truck] - start_packages_mapping[truck_load]))
                    start_packages_mapping[truck_load] += inc
                    rem_packages_on_truck -= inc
                    packages_on_truck += inc
            # remaining packages
            num_rem_packages = self.num_packages - packages_on_truck
            random.shuffle(random_loads)
            for var in random_loads:
                inc = random.randint(0, min(num_rem_packages, self.num_packages - start_packages_mapping[var]))
                start_packages_mapping[var] += inc
                num_rem_packages -= inc
            if num_rem_packages > 0:
                start_packages_mapping[self.location_load[self.location_indexes[0]]] += num_rem_packages
            # add to candidate:
            PythonUtils.update_dict(candidate, start_packages_mapping, True)
            #
            assert sum(start_packages_mapping.values()) == self.num_packages
            #
            rlt = states_values.add(candidate)
            if rlt is not None:
                return rlt

        # 25 %: 25% of packages at goal
        while states_values.size() < int(number_starts * 0.5):
            candidate = dict([(self.truck_vars[truck], 0) for truck in self.truck_indexes] + [(self.truck_velocities[truck], 0) for truck in self.truck_indexes] + [(self.aux_vel, 0)] + ([(self.park_vars[truck], 0) for truck in self.truck_indexes] if self.add_park_action() else []))
            #
            rem_packages_at_goal = int(self.num_packages * 0.25)
            packages_at_goal = 0
            start_packages_mapping = dict([(var, 0) for var in list(self.truck_load.values()) + list(self.location_load.values())])
            random.shuffle(random_loads)
            while rem_packages_at_goal > 0:
                for var in random_loads:
                    if var in self.goal_state:
                        inc = random.randint(0, min(rem_packages_at_goal, self.goal_state[var] - start_packages_mapping[var]))
                        start_packages_mapping[var] += inc
                        rem_packages_at_goal -= inc
                        packages_at_goal += inc
            num_rem_packages = self.num_packages - packages_at_goal
            random.shuffle(random_loads)
            for var in random_loads:
                inc = random.randint(0, min(num_rem_packages, self.num_packages - start_packages_mapping[var]))
                start_packages_mapping[var] += inc
                num_rem_packages -= inc
            if num_rem_packages > 0:
                start_packages_mapping[self.location_load[self.location_indexes[0]]] += num_rem_packages
            # add to candidate:
            PythonUtils.update_dict(candidate, start_packages_mapping, True)
            #
            assert sum(start_packages_mapping.values()) == self.num_packages
            #
            rlt = states_values.add(candidate)
            if rlt is not None:
                return rlt

        # 50 %: totally random:
        while states_values.size() < number_starts:
            candidate = dict([(self.truck_vars[truck], 0) for truck in self.truck_indexes] + [(self.truck_velocities[truck], 0) for truck in self.truck_indexes]  + [(self.aux_vel, 0)] + ([(self.park_vars[truck], 0) for truck in self.truck_indexes] if self.add_park_action() else []))
            #
            num_rem_packages = self.num_packages
            random.shuffle(random_loads)
            start_packages_mapping = dict([(var, 0) for var in list(self.truck_load.values()) + list(self.location_load.values())])  # initialize to have same-ordered state tuples at output
            for var in random_loads:
                start_packages_mapping[var] += random.randint(0, min(num_rem_packages, self.num_packages))
                num_rem_packages -= start_packages_mapping[var]
            if num_rem_packages > 0:
                start_packages_mapping[self.location_load[self.location_indexes[0]]] += num_rem_packages
            # add to candidate:
            PythonUtils.update_dict(candidate, start_packages_mapping, True)
            #
            assert sum(start_packages_mapping.values()) == self.num_packages
            #
            rlt = states_values.add(candidate)
            if rlt is not None:
                return rlt

        return states_values.generate_states_values()

    # predicate generation #############################################################################################

    def generate_stepwise_splits(self) -> list:
        if self.special_predicates:
            location_splits = [0 for _ in self.location_indexes]
            truck_load_splits = [0 for _ in self.truck_indexes]
            position_steps = [[0, i, 0] for i in range(0, self.truck_upper_bound + 1)]
            intermediate_steps = list()
            for i in range(0, len(truck_load_splits)):
                truck_load_splits[i] = 1
                intermediate_steps.append(location_splits + [self.truck_upper_bound] + truck_load_splits)
            for i in range(0, len(location_splits)):
                location_splits[i] = 1
                intermediate_steps.append(location_splits + [self.truck_upper_bound] + truck_load_splits)
            final_steps = [[i, self.truck_upper_bound, i] for i in range(2, max(self.location_upper_bound, max(self.truck_capacities.values())) + 1)]
            return position_steps + intermediate_steps + final_steps
        else:
            return JaniModelGenerator.generate_stepwise_splits([self.location_upper_bound, self.truck_upper_bound, max(self.truck_capacities.values())])

    def generate_splits_mapping(self, splits: list):
        split_specs = list()
        if self.special_predicates and len(splits) == self.num_locations + 1 + self.num_trucks:
            splits_index = 0
            # location load variables:
            for loc in self.location_indexes:
                split_specs.append(VarSplitSpec(BoundedVariable(self.location_load[loc], 0, self.location_upper_bound), splits[splits_index]))
                split_specs[-1].add_required_split(1)  # customized preds
                splits_index += 1
            # truck variables:
            for truck in self.truck_indexes:
                split_specs.append(VarSplitSpec(BoundedVariable(self.truck_vars[truck], 0, self.truck_upper_bound), splits[splits_index]))
                splits_index += 1
            # truck load variables:
            for truck in self.truck_indexes:
                split_specs.append(VarSplitSpec(BoundedVariable(self.truck_load[truck], 0, self.truck_capacities[truck]), splits[splits_index]))
                split_specs[-1].add_required_split(1)  # customized preds
                split_specs[-1].add_required_split(2)  # customized preds
                splits_index += 1
            return split_specs
        # else:
        JaniModelGenerator.unused_splits_check(splits, 3)  # location load, truck position, truck capacity
        # location load variables:
        for loc in self.location_indexes:
            split_specs.append(VarSplitSpec(BoundedVariable(self.location_load[loc], 0, self.location_upper_bound), splits[0]))
            if self.special_predicates:
                assert len(splits) != self.num_locations + 1 + self.num_trucks
                split_specs[-1].add_required_split(1)  # customized preds
        # truck variables:
        for truck in self.truck_indexes:
            split_specs.append(VarSplitSpec(BoundedVariable(self.truck_vars[truck], 0, self.truck_upper_bound), splits[1]))
        # truck load variables:
        for truck in self.truck_indexes:
            split_specs.append(VarSplitSpec(BoundedVariable(self.truck_load[truck], 0, self.truck_capacities[truck]), splits[2]))
            if self.special_predicates:
                assert len(splits) != self.num_locations + 1 + self.num_trucks
                split_specs[-1].add_required_split(1)  # customized preds
                split_specs[-1].add_required_split(2)  # customized preds
        return split_specs

    def generate_predicates(self, splits: list):
        assert not self.use_fuel(), "tank variables not support by predicate scaling"
        predicates_flag = [self.is_not_unsafe()]
        predicates_splitting = self.generate_splitting_predicates(self.generate_splits_mapping(splits))
        return predicates_flag + predicates_splitting

    # nn input #########################################################################################################

    def get_nn_inputs(self) -> list:
        nn_vars = list(self.variable_names)
        nn_vars.remove(self.aux_vel)
        return nn_vars

    ####################################################################################################################


if __name__ == "__main__":
    args = OneWayLineModelGenerationOptionParser().arg_parse()
    generator = OneWayLineModelGenerator(args)
    generator.generate()
