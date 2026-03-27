#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random

from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

random.seed(2020)


class TransportModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--dropping-prob", type=float, default=0, help="Probability to drop a package.")
        self.optionParser.add_argument("--tank-capacity", type=int, default=-1, help="Initial/maximal tank.")
        self.optionParser.add_argument("--fuel-unsafety", type=bool, default=False, help="Unsafe if not at goal within fuel limits.")
        # options for property generation (and thus not saved in model file):
        self.optionParser.add_argument("--single-safe-start-value", action="store_true", default=False, help="Fix safety variable to a single safe value at start.")
        self.optionParser.add_argument("--fix-truck-start", action="store_true", default=False, help="Trucks are at default position at start.")
        self.optionParser.add_argument("--fix-package-start", action="store_true", default=False, help="Packages are at default locations at start.")
        self.optionParser.add_argument("--zero-load-start", action="store_true", default=False, help="No packages loaded at start (which is -- at documentation time -- subsumed by --fix-package-start).")


class TransportModelGenerator(JaniModelGeneratorPddlInJani):
    # customization for linetrack
    LEFT_TO_BRIDGE = True
    # constants:
    NUM_LOCATIONS = "num_locations"
    NUM_PACKAGES = "num_packages"
    DROPPING_PROB_NAME = "dropping_prob"
    TANK_CAPACITY_NAME = "tank_capacity"
    FUEL_UNSAFE = "fuel_unsafe"

    def use_fuel(self):
        return False
        return self.tank_capacity > 0

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.special_predicates = "_truck_pos_predicates_first" in self.property_type
        self.terminal_at_unsafe_jani = False

        if self.model_file is None:
            # model generation
            self.description_name = options.description
            self.dropping_prob = options.dropping_prob
            self.tank_capacity = options.tank_capacity
            self.fuel_unsafe = options.fuel_unsafety
        else:
            # property generation
            constants = self.model["constants"]
            _, self.dropping_prob = self.read_constant(constants, self.DROPPING_PROB_NAME)
            _, self.tank_capacity = self.read_constant(constants, self.TANK_CAPACITY_NAME)
            _, self.fuel_unsafe = self.read_constant(constants, self.FUEL_UNSAFE)
            self.description_name = self.model["name"]
            #
        self.single_safe_start_value = options.single_safe_start_value
        self.fix_truck_start = options.fix_truck_start
        self.fix_package_start = options.fix_package_start
        self.zero_load_start = options.zero_load_start

        try:
            self.model_type = JaniModelType.MDP if self.sample_in_model or self.dropping_prob > 0 else JaniModelType.LTS
        except:
            self.model_type = JaniModelType.LTS
        self.model_name = self.description_name
        self.description = PythonUtils.load_json(self.description_name)
        self.num_locations = len(self.description["locations"])
        self.num_trucks = len(self.description["trucks"])
        self.num_packages = len(self.description["packages"])

        # variables:
        self.location_indexes = list(range(0, self.num_locations))
        self.location_values = list(self.location_indexes)  # per location index, the corresponding truck domain value
        self.truck_indexes = list(range(0, self.num_trucks))
        #
        self.location_load = dict([(loc, "location_load_" + str(loc)) for loc in self.location_indexes])
        self.truck_vars = dict([(truck, "truck_" + str(truck)) for truck in self.truck_indexes])
        self.truck_load = dict([(truck, "truck_load_" + str(truck)) for truck in self.truck_indexes])
        self.last_capacity_diff = "last_capacity_diff"
        self.variable_names = list(self.location_load.values()) + list(self.truck_vars.values()) + list(self.truck_load.values()) + ([self.last_capacity_diff] if not self.fuel_unsafe else [])
        if self.use_fuel():
            self.truck_tank = dict([(truck, "truck_tank_" + str(truck)) for truck in self.truck_indexes])
            self.variable_names += list(self.truck_tank.values())

        # aux for vars:
        self.location_domain_size = self.num_packages + 1
        self.location_upper_bound = self.location_domain_size - 1
        self.truck_domain_size = self.num_locations
        self.truck_upper_bound = self.truck_domain_size - 1
        self.truck_capacities = dict([(truck, truck_item["capacity"]) for (truck, truck_item) in zip(self.truck_indexes, self.description["trucks"])])
        self.truck_capacities_per_load = dict([(truck_load_var, self.truck_capacities[truck]) for (truck, truck_load_var) in self.truck_load.items()])
        if self.use_fuel():
            self.tank_lower_bound = - self.num_packages  # for now we subtract one tank unit per package and road

        # actions
        self.action_labels_pick_up = dict([(truck, "pick_up_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_drop = dict([(truck, "drop_" + self.truck_vars[truck]) for truck in self.truck_indexes])
        self.action_labels_drive = list(self.description["drive-actions"])
        self.action_labels = list(self.action_labels_pick_up.values()) + list(self.action_labels_drop.values()) + list(self.action_labels_drive)

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
                                  + ([(self.last_capacity_diff, 0)] if not self.fuel_unsafe else [])
                                  + ([(self.truck_tank[truck], self.tank_capacity) for truck in self.truck_indexes] if self.use_fuel() else []))
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
            self.goal_state[self.location_load[self.location_indexes[loc_start_end["end"]]]] += 1

    # noinspection PyAttributeOutsideInit
    def load_roads(self):
        roads = self.description["locations"]
        self.roads = dict()
        self.road_labels = dict()
        for roads_item in roads:
            src_loc_index = self.location_indexes[roads_item["id"]]
            for road in roads_item["roads"]:
                target_loc_index = self.location_indexes[road["to"]]
                self.roads[(src_loc_index, target_loc_index)] = (road["capacity"] if not self.fuel_unsafe else self.get_max_capacity(0))
                self.road_labels[(src_loc_index, target_loc_index)] = road["label"]

    def has_road(self, src_location_index: int, target_location_index: int) -> bool:
        return (src_location_index, target_location_index) in self.roads

    def get_capacity(self, src_location_index: int, target_location_index: int) -> int:
        assert self.has_road(src_location_index, target_location_index)
        return self.roads[(src_location_index, target_location_index)]

    def get_max_capacity(self, src_location_index: int) -> int:
        max_cap = 0
        for target_loc_index in self.location_indexes:
            if self.has_road(src_location_index, target_loc_index):
                max_cap = max(max_cap, self.get_capacity(src_location_index, target_loc_index))
        return max_cap

    def get_road_label(self, src_location_index: int, target_location_index: int) -> str:
        assert self.has_road(src_location_index, target_location_index)
        return self.road_labels[(src_location_index, target_location_index)]

    # constraints #############################################################################################

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

    def is_unsafe(self) -> json:
        return Je.Or(([self.exceeded_capacity()] if not self.fuel_unsafe else []) + ([self.has_no_tank(truck) for truck in self.truck_indexes] if self.use_fuel() else []))

    def is_not_unsafe(self) -> json:
        return Je.And(([self.did_not_exceed_capacity()] if not self.fuel_unsafe else []) + ([self.has_tank(truck) for truck in self.truck_indexes] if self.use_fuel() else []))

    def load_exceeds_capacity(self, truck: int, src_loc: int, target_loc: int) -> json:
        return Je.Ge(self.truck_load[truck], self.get_capacity(src_loc, target_loc) + 1)

    def load_not_exceeds_capacity(self, truck: int, src_loc: int, target_loc: int) -> json:
        return Je.Le(self.truck_load[truck], self.get_capacity(src_loc, target_loc))

    def exceeded_capacity(self):
        return Je.Le(self.last_capacity_diff, -1)

    def did_not_exceed_capacity(self):
        return Je.Ge(self.last_capacity_diff, 0)

    def has_tank(self, truck: int) -> json:
        return Je.Ge(self.truck_tank[truck], 0)  # self.truck_load[truck]

    def has_no_tank(self, truck: int) -> json:
        return Je.Le(self.truck_tank[truck], -1)  # Je.Add(self.truck_load[truck], -1)

    # assignments

    def inc_location_load(self, loc: int, inc: int = 1) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.location_load[loc], inc)

    def move_truck(self, truck: int, loc: int) -> json:
        return JaniStructureGenerator.generate_assignment(self.truck_vars[truck], self.location_values[loc])

    def inc_truck_load(self, truck: int, inc: int = 1) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.truck_load[truck], inc)

    def set_last_capacity_diff(self, truck: int, src_loc: int, target_loc: int) -> json:
        return JaniStructureGenerator.generate_assignment(self.last_capacity_diff, Je.Sub(self.get_capacity(src_loc, target_loc), self.truck_load[truck]))

    def update_drive_tank(self, truck: int) -> json:
        # for now we subtract one tank unit per package and path:
        return JaniStructureGenerator.generate_self_assignment(self.truck_tank[truck], Je.Mult(-1, self.truck_load[truck]))
    
    def update_package_tank(self, truck: int) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.truck_tank[truck], -1)

    def pick_up(self, truck: int, loc: int):
        return [self.inc_location_load(loc, -1), self.inc_truck_load(truck, 1)] + ([self.update_package_tank(truck)] if self.use_fuel() else [])

    def drop(self, truck: int, loc: int):
        return [self.inc_location_load(loc, 1), self.inc_truck_load(truck, -1)] + ([self.update_package_tank(truck)] if self.use_fuel() else [])

    # model generation #################################################################################################

    def generate_location_variable(self, loc: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.location_load[loc], 0, self.location_upper_bound, self.initial_state[self.location_load[loc]])

    def generate_truck_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_vars[truck], 0, self.truck_upper_bound, self.initial_state[self.truck_vars[truck]])

    def generate_truck_capacity_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_load[truck], 0, self.truck_capacities[truck], self.initial_state[self.truck_load[truck]])

    def generate_last_capacity_diff_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.last_capacity_diff, min(self.roads.values()) - max(self.truck_capacities.values()), max(self.truck_capacities.values()), self.initial_state[self.last_capacity_diff])

    def generate_truck_tank_variable(self, truck: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck_tank[truck], self.tank_lower_bound, self.tank_capacity, self.initial_state[self.truck_tank[truck]])

    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.NUM_LOCATIONS, JaniStructureGenerator.generate_int_type(), self.num_locations)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.NUM_PACKAGES, JaniStructureGenerator.generate_int_type(), self.num_packages)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.DROPPING_PROB_NAME, JaniStructureGenerator.generate_real_type(), self.dropping_prob)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.TANK_CAPACITY_NAME, JaniStructureGenerator.generate_int_type(), self.tank_capacity)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.TERMINAL_AT_UNSAFE, JaniStructureGenerator.generate_bool_type(), self.terminal_at_unsafe_jani)] + \
            [JaniStructureGenerator.generate_constant_declaration(self.FUEL_UNSAFE, JaniStructureGenerator.generate_bool_type(), self.fuel_unsafe)]

    def generate_variables(self):
        variables = [self.generate_location_variable(loc) for loc in self.location_indexes]
        variables += [self.generate_truck_variable(truck) for truck in self.truck_indexes]
        variables += [self.generate_truck_capacity_variable(truck) for truck in self.truck_indexes]
        variables += [self.generate_last_capacity_diff_variable()] if not self.fuel_unsafe else []
        variables += [self.generate_truck_tank_variable(truck) for truck in self.truck_indexes if self.use_fuel()]
        return variables

    #

    def generate_edge_aux(self, destinations: list, action: str, guard) -> json:
        if self.terminal_at_unsafe_jani:
            guard = Je.And(self.is_not_unsafe(), guard)
        return JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, guard)

    def generate_edges(self):

        def generate_pick_up_edge(truck: int, loc: int):
            action = self.action_labels_pick_up[truck]
            guard = Je.And(self.truck_is_at(truck, loc), self.truck_has_capacity(truck), self.location_has_package(loc))
            destinations = [self.generate_destination_aux(self.pick_up(truck, loc), None)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_drop_edge(truck: int, loc: int):
            action = self.action_labels_drop[truck]
            guard = Je.And(self.truck_is_at(truck, loc), self.truck_has_package(truck))
            destinations = [self.generate_destination_aux(self.drop(truck, loc), None)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_drive_success(truck: int, src_loc: int, target_loc: int):
            assert self.has_road(src_loc, target_loc)
            action = self.get_road_label(src_loc, target_loc)
            # guard = Je.And(self.truck_is_at(truck, src_loc), self.load_not_exceeds_capacity(truck, src_loc, target_loc))
            guard = self.truck_is_at(truck, src_loc)
            guard = Je.And(self.has_tank(truck), guard) if self.use_fuel() else guard
            assignments = [self.move_truck(truck, target_loc)]
            assignments += [self.set_last_capacity_diff(truck, src_loc, target_loc)] if not self.fuel_unsafe else []
            assignments += [self.update_package_tank(truck)] if self.use_fuel() else []
            destinations = [self.generate_destination_aux(assignments, None)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_drive_pick_up_edge(truck: int, src_loc: int, target_loc: int):
            assert self.has_road(src_loc, target_loc)
            action = self.get_road_label(src_loc, target_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.truck_has_capacity(truck), self.location_has_package(src_loc))
            destinations = [self.generate_destination_aux([self.inc_truck_load(truck, 1), self.inc_location_load(src_loc, -1)], None)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_drive_drop_edge(truck: int, src_loc: int, target_loc: int):
            assert self.has_road(src_loc, target_loc)
            action = self.get_road_label(src_loc, target_loc)
            guard = Je.And(self.truck_is_at(truck, src_loc), self.truck_has_package(truck))
            destinations = [self.generate_destination_aux([self.inc_truck_load(truck, -1), self.inc_location_load(src_loc, 1)], None)]
            return self.generate_edge_aux(destinations, action, guard)

        def generate_sample_trucks() -> list:
            assignments = list()
            for truck in self.truck_indexes:
                assignments += [JaniStructureGenerator.generate_non_det_assignment(self.truck_vars[truck], 0, self.truck_upper_bound)]
                if self.use_fuel():
                    assignments += [JaniStructureGenerator.generate_non_det_assignment(self.truck_tank[truck], 0, self.tank_capacity)]
            return [JaniStructureGenerator.generate_edge(location=self.sample_truck_location, destinations=[JaniStructureGenerator.generate_destination(location=self.sample_load, assignments=assignments)])]

        def generate_sample_load() -> list:
            guard = Je.Lt(Je.Add([self.location_load[index] for index in self.location_indexes] + [self.truck_load[index] for index in self.truck_indexes]), self.NUM_PACKAGES)

            destinations = list()

            # Number of possible packages positions:
            possible_positions = Je.Add([self.NUM_LOCATIONS] + [Je.to_int(self.truck_has_capacity(truck)) for truck in self.truck_indexes])

            # For each locations and truck we have a destinations
            for loc in self.location_indexes:
                destinations.append(JaniStructureGenerator.generate_destination(self.sample_load, assignments=[self.inc_location_load(loc, inc=1)], probability=Je.Div(1, possible_positions)))
            for truck in self.truck_indexes:
                destinations.append(JaniStructureGenerator.generate_destination(self.sample_load, assignments=[self.inc_truck_load(truck, inc=1)], probability=Je.Div(Je.to_int(self.truck_has_capacity(truck)), possible_positions)))

            return [JaniStructureGenerator.generate_edge(location=self.sample_load, destinations=destinations, guard=guard)]

        def generate_end_sampling_edge() -> list:
            guard = Je.Eq(Je.Add([self.location_load[index] for index in self.location_indexes] + [self.truck_load[index] for index in self.truck_indexes]), self.NUM_PACKAGES)
            return [JaniStructureGenerator.generate_edge(location=self.sample_load, destinations=[JaniStructureGenerator.generate_destination(location=self.LOCATION_NAME)], guard=guard)]

        edges = [generate_pick_up_edge(truck, loc) for loc in self.location_indexes for truck in self.truck_indexes]
        edges += [generate_drop_edge(truck, loc) for loc in self.location_indexes for truck in self.truck_indexes]
        edges += [generate_drive_success(truck, src_loc, target_loc) for target_loc in self.location_indexes for src_loc in self.location_indexes for truck in self.truck_indexes if self.has_road(src_loc, target_loc)]
        if self.dropping_prob > 0:
            edges += [generate_drive_pick_up_edge(truck, src_loc, target_loc) for target_loc in self.location_indexes for src_loc in self.location_indexes for truck in self.truck_indexes if self.has_road(src_loc, target_loc)]
            edges += [generate_drive_drop_edge(truck, src_loc, target_loc) for target_loc in self.location_indexes for src_loc in self.location_indexes for truck in self.truck_indexes if self.has_road(src_loc, target_loc)]
        if self.sample_in_model:
            edges += generate_sample_trucks()
            edges += generate_sample_load()
            edges += generate_end_sampling_edge()

        return edges

    # property generation ##############################################################################################

    def generate_objective(self) -> json:
        if not self.fuel_unsafe:
            goal = JaniStructureGenerator.generate_state_condition_expression([], Je.And(self.generate_goal_expression(), self.did_not_exceed_capacity()))
        else:
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
                max_cap_for_move = self.get_max_capacity(loc)
                # negative reward if more packages are load than max cap
                if max_cap_for_move < self.num_packages:  # exclude special case where there is no capacity limitation
                    additive_list.append(Je.Ite(self.truck_is_at(truck, loc), Je.Mult(-10, Je.min(0, Je.Sub(self.truck_load[truck], max_cap_for_move))), 0))
                # loading at least one package is required towards the goal, if goal-number of packages is delivered at current loc ...
                if self.goal_state[self.location_load[loc]] < self.num_packages:  # exclude special case where all packages are required at the location
                    reward_con = [self.truck_is_at(truck, loc), Je.Ge(self.truck_load[truck], 1)] + ([Je.Ge(self.location_load[loc], self.goal_state[self.location_load[loc]])] if self.goal_state[self.location_load[loc]] > 0 else [])
                    additive_list.append(Je.Ite(Je.And(reward_con), 10, 0))
        goal_potential = Je.Add(additive_list)
        goal_potential = Je.Ite(self.generate_start(), 0, goal_potential) if self.ground_start_potential else goal_potential
        goal_potential = Je.Ite(Je.Or(Je.And(self.generate_goal_expression(), self.did_not_exceed_capacity()), self.generate_reach()), 0, goal_potential) if self.ground_terminal_potential else goal_potential
        return goal_potential

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.is_unsafe())

    # starts ###########################################################################################################

    def generate_start(self):
        constraints = list()

        # safe at start!!!
        if not self.fuel_unsafe:
            constraints.append(Je.Eq(self.last_capacity_diff, 0) if self.single_safe_start_value else self.did_not_exceed_capacity())
        else:
            constraints.append(Je.And([self.has_tank(truck) for truck in self.truck_indexes]))
        # full tank at start
        constraints += [Je.Eq(tank, self.tank_capacity) for tank in self.truck_tank.values()] if self.use_fuel() else []

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

        # 25 %: 75% of packages on truck
        while states_values.size() < int(number_starts * 0.25):
            candidate = dict([(self.truck_vars[truck], random.randint(0, self.truck_upper_bound)) for truck in self.truck_indexes])  # random position
            # packages on truck
            rem_packages_on_truck = int(self.num_packages * 0.75)
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
                inc = random.randint(0, min(num_rem_packages, self.truck_capacities_per_load[var] - start_packages_mapping[var]) if var in self.truck_capacities_per_load else num_rem_packages)
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

        # 25 %: 75% of packages at goal
        while states_values.size() < int(number_starts * 0.5):
            candidate = dict([(self.truck_vars[truck], random.randint(0, self.truck_upper_bound)) for truck in self.truck_indexes])  # random position
            #
            rem_packages_at_goal = int(self.num_packages * 0.75)
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
                inc = random.randint(0, min(num_rem_packages, self.truck_capacities_per_load[var] - start_packages_mapping[var]) if var in self.truck_capacities_per_load else num_rem_packages)
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
            candidate = dict([(self.truck_vars[truck], random.randint(0, self.truck_upper_bound)) for truck in self.truck_indexes])  # random position
            #
            num_rem_packages = self.num_packages
            random.shuffle(random_loads)
            start_packages_mapping = dict([(var, 0) for var in list(self.truck_load.values()) + list(self.location_load.values())])  # initialize to have same-ordered state tuples at output
            for var in random_loads:
                start_packages_mapping[var] += random.randint(0, min(num_rem_packages, self.truck_capacities_per_load[var]) if var in self.truck_capacities_per_load else num_rem_packages)
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
        predicates_flag = [self.did_not_exceed_capacity()]
        predicates_splitting = self.generate_splitting_predicates(self.generate_splits_mapping(splits))
        return predicates_flag + predicates_splitting

    # nn input #########################################################################################################

    def get_nn_inputs(self) -> list:
        nn_vars = list(self.variable_names)
        if not self.fuel_unsafe:
            nn_vars.remove(self.last_capacity_diff)
        return nn_vars

    ####################################################################################################################


if __name__ == "__main__":
    args = TransportModelGenerationOptionParser().arg_parse()
    generator = TransportModelGenerator(args)
    generator.generate()
