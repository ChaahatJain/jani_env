#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random

from jani_generation.jani_model_generator import JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

from rovers_description import RoversDescription

random.seed(2020)


class RoversModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--enable-battery-overload", action="store_true", default=False, help="Battery may overload.")
        self.optionParser.add_argument("--enable-oom-moves", action="store_true", default=False, help="Rover may drive off map and crash.")
        self.optionParser.add_argument("--fail-prob-charge", type=float, default=0, help="Probability for charging to fail.")
        self.optionParser.add_argument("--fail-prob-sample", type=float, default=0, help="Probability for sampling to fail.")
        self.optionParser.add_argument("--fail-prob-image", type=float, default=0, help="Probability for image taking to fail.")
        self.optionParser.add_argument("--use-policy", action="store_true", default=False, help="Control rover(s) via policy.")
        self.optionParser.add_argument("--use-multi-loc-policy", action="store_true", default=False, help="Policy via multiple locations.")
        # options for policy/property generation (and thus not saved in model file):
        self.optionParser.add_argument("--rover-at-lander", action="store_true", default=False, help="Rover is at lander at start.")


class RoversModelGenerator(JaniModelGeneratorPddlInJani):
    # constants:
    ENABLE_BATTERY_OVERLOAD_NAME = "enable-battery-overload"
    ENABLE_OOM_MOVES_NAME = "enable-oom-moves"
    FAIL_PROB_CHARGE_NAME = "fail-prob-charge"
    FAIL_PROB_SAMPLE_NAME = "fail-prob-sample"
    FAIL_PROB_IMAGE_NAME = "fail-prob-image"
    USE_POLICY = "use-policy"
    USE_MULTI_LOC_POLICY = "use-policy-multi-loc-policy"
    # locations:
    CHOICE_LOC = "choice_point"
    ERROR_LOC = "error"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)

        if self.model_file is None:
            # model generation
            self.enable_battery_overload = options.enable_battery_overload
            self.enable_oom_moves = options.enable_oom_moves
            self.description_name = options.description
            self.fail_prob_charge = options.fail_prob_charge
            self.fail_prob_sample = options.fail_prob_sample
            self.fail_prob_image = options.fail_prob_image
            self.use_policy = options.use_policy
            self.use_multi_loc_policy = options.use_multi_loc_policy
        else:
            # property generation
            model = PythonUtils.load_json(self.model_file)
            constants = model["constants"]
            self.enable_battery_overload = self.read_constant(constants, self.ENABLE_BATTERY_OVERLOAD_NAME)
            self.enable_oom_moves = self.read_constant(constants, self.ENABLE_OOM_MOVES_NAME)
            self.fail_prob_charge = self.read_constant(constants, self.FAIL_PROB_CHARGE_NAME)
            self.fail_prob_sample = self.read_constant(constants, self.FAIL_PROB_SAMPLE_NAME)
            self.fail_prob_image = self.read_constant(constants, self.FAIL_PROB_IMAGE_NAME)
            self.use_policy = self.read_constant(constants, self.USE_POLICY)
            self.use_multi_loc_policy = self.read_constant(constants, self.USE_MULTI_LOC_POLICY)
            #
            self.rover_at_lander = options.rover_at_lander
            #
            self.description_name = model["name"]
        #
        self.model_type = JaniModelType.MDP if self.fail_prob_sample + self.fail_prob_image > 0 else JaniModelType.LTS

        self.model_name = self.description_name
        self.description = RoversDescription(self.description_name)

        assert self.enable_battery_overload

        # locations:
        if self.description.has_special_cap():
            self.check_move_locations = dict([((rover, direction), "check_move_" + direction + "_" + rover.to_str()) for direction in self.description.DIRECTION_NAMES for rover in self.description.rovers])

        # variables:
        self.rover_vars_x = dict([(rover, rover.to_str() + "_x") for rover in self.description.range_rovers()])
        self.rover_vars_y = dict([(rover, rover.to_str() + "_y") for rover in self.description.range_rovers()])
        self.rover_battery_vars = dict([(rover, "battery_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.rover_rock_vars = dict([(rover, "rocks_on_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.rover_soil_vars = dict([(rover, "soil_on_" + rover.to_str()) for rover in self.description.range_rovers()])
        # self.rover_image_vars = dict([(rover, "image_on_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.cell_rock_vars = dict([(cell, "rocks_on_cell_" + cell.to_str()) for cell in self.description.range_cells()])
        self.cell_soil_vars = dict([(cell, "soil_on_cell_" + cell.to_str()) for cell in self.description.range_cells()])
        self.objective_vars = dict([(objective, "image_of_objective_" + objective.to_str()) for objective in self.description.objectives])  # has objective been imaged
        # we do not use last capacity diff here, but add an extract location to check the direction
        # otherwise we need one operator per map path (as in transport)
        # alternatively: we could support disjunctions (over the special case capacities) directly in marabou and safe-vs-unsafe move operators

        # intermediate vars to check crash:
        if self.description.has_special_cap():
            self.rover_vars_x_next = dict([(rover, rover.to_str() + "_x_n") for rover in self.description.range_rovers()])
            self.rover_vars_y_next = dict([(rover, rover.to_str() + "_y_n") for rover in self.description.range_rovers()])
        elif self.description.has_default_cap():
            self.last_cap_diff = "last_cap_diff"

        # TODO intermediate location to check whether soil/rock at current position

        # actions
        self.action_labels_charge = dict([(rover, "charge_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.action_labels_sample = dict([(rover, "sample_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.action_labels_sample_drop = dict([(rover, "drop_samples_" + rover.to_str()) for rover in self.description.range_rovers()])
        self.action_labels_move = dict([((rover, direction), direction + "_" + rover.to_str()) for rover in self.description.range_rovers() for direction in self.description.DIRECTION_NAMES])
        if self.description.has_objectives():
            self.action_labels_take_image = dict([(rover, "take_image_" + rover.to_str()) for rover in self.description.range_rovers()])
            # self.action_labels_share_image = dict([(rover, "share_image_" + rover.to_str()) for rover in self.description.range_rovers()])
            action_labels_image = list(self.action_labels_take_image.values())  # + list(self.action_labels_share_image.values())
        else:
            action_labels_image = list()
        self.action_labels = list(self.action_labels_charge.values()) + list(self.action_labels_sample.values()) + list(self.action_labels_sample_drop.values()) + action_labels_image + list(self.action_labels_move.values())

        self.compute_model_initial_and_goal_state()

        # use policy:
        if self.use_policy:
            self.policy_automata = dict([(rover, "policy_" + rover.to_str()) for rover in self.description.rovers])
            if not self.use_multi_loc_policy:
                self.policy_init_loc = self.ITE_POLICY_LOC
                self.policy_locations = [self.policy_init_loc]
            else:
                self.policy_init_loc = self.CHARGE_LOC
                self.policy_locations = [self.CHARGE_LOC, self.DROP_LOC, self.RETURN_LOC]
                self.policy_locations += [self.TAKE_IMAGE_LOC] if self.description.has_objectives() else []
                self.policy_locations += [self.SAMPLE_LOC, self.SEARCH_LOC]
                self.aux_actions = dict([(rover, ["wait_for_policy_" + rover.to_str(), "in_process_" + rover.to_str()]) for rover in self.description.rovers])
                for aux_actions in self.aux_actions.values():
                    self.action_labels += aux_actions

    # noinspection PyAttributeOutsideInit
    def compute_model_initial_and_goal_state(self):
        # initial
        self.initial_state = dict([(rover_x, self.description.lander_x()) for rover_x in self.rover_vars_x.values()]
                                  + [(rover_y, self.description.lander_y()) for rover_y in self.rover_vars_y.values()]
                                  + [(var, rover.battery) for var, rover in zip(self.rover_battery_vars.values(), self.description.rovers)]
                                  + [(var, 0) for var in self.rover_rock_vars.values()]
                                  + [(var, 0) for var in self.rover_soil_vars.values()]
                                  # + [(var, -1) for var in self.rover_image_vars.values()]
                                  + [(var, self.description.rocks_on_cell(cell)) for cell, var in self.cell_rock_vars.items()]
                                  + [(var, self.description.soil_on_cell(cell)) for cell, var in self.cell_soil_vars.items()]
                                  + [(var, 0) for var in self.objective_vars.values()]
                                  )

        if self.description.has_special_cap():
            for rover_x, rover_y in zip(self.rover_vars_x_next.values(), self.rover_vars_y_next.values()):
                self.initial_state[rover_x] = self.description.lander_x()
                self.initial_state[rover_y] = self.description.lander_y()
        elif self.description.has_default_cap():
            self.initial_state[self.last_cap_diff] = 0

        # goal
        self.goal_state = None  # there is no distinguished singleton goal state

    def generate_goal_expression(self) -> json:
        goal_con = list()
        # rover at lander
        goal_con += [self.is_on_cell(rover, self.description.lander) for rover in self.description.rovers]
        # one rock & soil sample
        goal_con.append(Je.Ge(self.cell_rock_vars[self.description.lander], 1))
        goal_con.append(Je.Ge(self.cell_soil_vars[self.description.lander], 1))
        # images taken
        goal_con += [Je.Eq(var, 1) for var in self.objective_vars.values()]
        # rover is safe (including battery not exhausted):
        goal_con += [self.is_safe(rover) for rover in self.description.rovers]
        return Je.And(goal_con)

    # constraints ######################################################################################################

    # moves

    def is_on_cell(self, rover: RoversDescription.Rover, cell: RoversDescription.Cell, next_vars: bool = False, truth_value: bool = True) -> json:
        x = (self.rover_vars_x_next if next_vars else self.rover_vars_x)[rover]
        y = (self.rover_vars_y_next if next_vars else self.rover_vars_y)[rover]
        return Je.And(Je.Eq(x, cell.x), Je.Eq(y, cell.y)) if truth_value else Je.Or(Je.Ne(x, cell.x), Je.Ne(y, cell.y))

    def manhatten_distance(self, rover: RoversDescription.Rover, cell: RoversDescription.Cell) -> json:
        d_x = self.rover_vars_x[rover]if cell.x == 0 else Je.Abs(Je.AddNeg(self.rover_vars_x[rover], self.description.lander_x()))  # special case
        d_y = self.rover_vars_y[rover] if cell.y == 0 else Je.Abs(Je.AddNeg(self.rover_vars_y[rover], self.description.lander_y()))  # special case
        return Je.Add(d_x, d_y)

    @staticmethod
    def manhatten_distance_val(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def is_in_bounds(self, rover: RoversDescription.Rover):
        x_var, y_var = self.rover_vars_x[rover], self.rover_vars_y[rover]
        return Je.And(Je.Le(0, x_var), Je.Le(x_var, self.description.x_dim - 1), Je.Le(0, y_var), Je.Le(y_var, self.description.y_dim - 1))

    def move_in_bounds(self, rover: RoversDescription.Rover, direction: str) -> json:
        x_var, y_var = self.rover_vars_x[rover], self.rover_vars_y[rover]
        x_d, y_d = self.description.DIRECTION_DIFFS[direction]
        bounds = list()
        if x_d > 0:
            bounds.append(Je.Le(Je.Add(x_var, x_d), self.description.x_dim - 1))
        elif x_d < 0:
            bounds.append(Je.Le(0, Je.Add(x_var, x_d)))
        if y_d > 0:
            bounds.append(Je.Le(Je.Add(y_var, y_d), self.description.y_dim - 1))
        elif y_d < 0:
            bounds.append(Je.Le(0, Je.Add(y_var, y_d)))
        return Je.And(bounds)

    def exceeded_path_capacity(self, rover: RoversDescription.Rover, direction: str, next_vars: bool = True) -> json:
        assert self.description.has_special_cap()
        cap_map = self.description.extract_cells(direction)
        cap_expr = Je.Add(self.rover_rock_vars[rover], self.rover_soil_vars[rover])
        default_cap = self.description.extract_default_cap()
        ex_disjunction = list()
        # special caps:
        for cap, src_cells in cap_map.items():
            if cap == default_cap:  # can be handled via default
                continue
            ex_disjunction.append(Je.And(Je.Ge(cap_expr, cap + 1), Je.Or([self.is_on_cell(rover, cell, next_vars) for cell in src_cells])))
        # default:
        if len(cap_map) == 0 or default_cap >= max(cap_map.keys()):
            ex_disjunction.append(Je.Ge(cap_expr, default_cap + 1))
        else:
            not_special_conjunction = list()
            for cap, src_cells in cap_map.items():
                if cap <= default_cap:
                    continue
                not_special_conjunction += [self.is_on_cell(rover, cell, next_vars, truth_value=False) for cell in src_cells]
            ex_disjunction.append(Je.And(Je.Ge(cap_expr, default_cap + 1), Je.And(not_special_conjunction)))
        #
        return Je.Or(ex_disjunction)

    def respect_path_capacity(self, rover: RoversDescription.Rover, direction: str, next_vars: bool = True) -> json:
        return Je.Not(self.exceeded_path_capacity(rover, direction, next_vars))

    def safe_per_last_cap_diff(self, truth_value: bool = True) -> json:
        assert not self.description.has_only_default_cap()
        return Je.Ge(self.last_cap_diff, 0) if truth_value else Je.Le(self.last_cap_diff, -1)

    # rocks & soil

    def rock_on_rover(self, rover: RoversDescription.Rover, truth_value: bool = True) -> json:
        return Je.Ge(self.rover_rock_vars[rover], 1) if truth_value else Je.Le(self.rover_rock_vars[rover], 0)

    def soil_on_rover(self, rover: RoversDescription.Rover, truth_value: bool = True) -> json:
        return Je.Ge(self.rover_soil_vars[rover], 1) if truth_value else Je.Le(self.rover_soil_vars[rover], 0)

    def load_on_rover(self, rover: RoversDescription.Rover) -> json:
        return Je.Add(self.rover_rock_vars[rover], self.rover_soil_vars[rover])

    def samples_on_rover(self, rover: RoversDescription.Rover, truth_value: bool = True) -> json:
        return Je.Ge(self.load_on_rover(rover), 1) if truth_value else Je.Le(self.load_on_rover(rover), 0)

    def rock_on_cell(self, cell: RoversDescription.Cell, rover: RoversDescription.Rover = None) -> json:
        return Je.And([Je.Ge(self.cell_rock_vars[cell], 1)] + ([] if rover is None else [self.is_on_cell(rover, cell)]))

    def soil_on_cell(self, cell: RoversDescription.Cell, rover: RoversDescription.Rover = None) -> json:
        return Je.And([Je.Ge(self.cell_soil_vars[cell], 1)] + ([] if rover is None else [self.is_on_cell(rover, cell)]))

    def samples_on_cell(self, cell: RoversDescription.Cell, rover: RoversDescription.Rover = None) -> json:
        return Je.And([Je.Ge(Je.Add(self.cell_rock_vars[cell], self.cell_soil_vars[cell]), 1)] + ([] if rover is None else [self.is_on_cell(rover, cell)]))

    # images

    def image_taken(self, objective: RoversDescription.Cell, truth_value: bool = True) -> json:
        return Je.Ge(self.objective_vars[objective], 1) if truth_value else Je.Le(self.objective_vars[objective], 0)

    # energy

    def has_battery(self, rover: RoversDescription.Rover, min_b: int = 1, max_b: int = None) -> json:
        battery_var = self.rover_battery_vars[rover]
        return Je.And(Je.Ge(battery_var, min_b), Je.Le(battery_var, rover.battery if max_b is None else max_b))

    #

    def is_safe(self, rover: RoversDescription.Rover):
        safe_con = [self.has_battery(rover, min_b=0, max_b=rover.battery)] if self.enable_battery_overload else []
        safe_con += [self.is_in_bounds(rover)] if self.enable_oom_moves else []
        safe_con += [self.safe_per_last_cap_diff()] if self.description.has_only_default_cap() else []
        return Je.And(safe_con)  # just set battery negative for other unsafety criteria (e.g., capacity)

    # assignments ######################################################################################################

    # moves

    def update_position(self, rover: RoversDescription.Rover, direction: str, next_vars: bool = True) -> list:
        assert self.description.has_special_cap() or not next_vars
        x_d, y_d = self.description.DIRECTION_DIFFS[direction]
        assignments = list()
        if x_d != 0:
            if next_vars:
                assignments.append(JaniStructureGenerator.generate_assignment(self.rover_vars_x_next[rover], Je.Add(self.rover_vars_x[rover], x_d)))
            else:
                assignments.append(JaniStructureGenerator.generate_self_assignment(self.rover_vars_x[rover], x_d))
        if y_d != 0:
            if next_vars:
                assignments.append(JaniStructureGenerator.generate_assignment(self.rover_vars_y_next[rover], Je.Add(self.rover_vars_y[rover], y_d)))
            else:
                assignments.append(JaniStructureGenerator.generate_self_assignment(self.rover_vars_y[rover], y_d))
        assert len(assignments) > 0
        return assignments

    def update_position_next(self, rover: RoversDescription.Rover, direction: str) -> list:
        assert self.description.has_special_cap()
        x_d, y_d = self.description.DIRECTION_DIFFS[direction]
        assignments = list()
        if x_d != 0:
            assignments.append(JaniStructureGenerator.generate_assignment(self.rover_vars_x[rover], self.rover_vars_x_next[rover]))
        if y_d != 0:
            assignments.append(JaniStructureGenerator.generate_assignment(self.rover_vars_y[rover], self.rover_vars_y_next[rover]))
        assert len(assignments) > 0
        return assignments

    def set_last_cap_diff(self, rover: RoversDescription.Rover) -> list:
        assert self.description.has_only_default_cap()
        return [JaniStructureGenerator.generate_assignment(self.last_cap_diff, Je.AddNeg(self.description.default_path_capacity, self.load_on_rover(rover)))]

    # rock & soil

    def upd_rock_on_rover(self, rover: RoversDescription.Rover, diff: json) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.rover_rock_vars[rover], diff)

    def upd_soil_on_rover(self, rover: RoversDescription.Rover, diff: json) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.rover_soil_vars[rover], diff)

    def upd_rock_on_cell(self, cell: RoversDescription.Cell, diff: json) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.cell_rock_vars[cell], diff)

    def upd_soil_on_cell(self, cell: RoversDescription.Cell, diff: json) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.cell_soil_vars[cell], diff)

    def drop_rock(self, rover: RoversDescription.Rover, cell: RoversDescription.Cell) -> list:
        rover_var = self.rover_rock_vars[rover]
        return [JaniStructureGenerator.generate_assignment(rover_var, 0), JaniStructureGenerator.generate_self_assignment(self.cell_rock_vars[cell], rover_var)]

    def drop_soil(self, rover: RoversDescription.Rover, cell: RoversDescription.Cell) -> list:
        rover_var = self.rover_soil_vars[rover]
        return [JaniStructureGenerator.generate_assignment(rover_var, 0), JaniStructureGenerator.generate_self_assignment(self.cell_soil_vars[cell], rover_var)]

    # images

    def take_image(self, objective: RoversDescription.Cell) -> json:
        return JaniStructureGenerator.generate_assignment(self.objective_vars[objective], 1)

    # energy

    def update_battery(self, rover: RoversDescription.Rover, energy: json) -> json:
        return JaniStructureGenerator.generate_self_assignment(self.rover_battery_vars[rover], energy)

    def update_battery_move(self, rover: RoversDescription.Rover) -> json:
        energy_for_move = Je.Add(-self.description.move_energy(), Je.Mult(self.rover_rock_vars[rover], -self.description.move_energy_per_rock()), Je.Mult(self.rover_soil_vars[rover], -self.description.move_energy_per_soil()))
        return self.update_battery(rover, energy_for_move)

    def upd_battery_sample(self, rover: RoversDescription.Rover) -> list:
        return [self.update_battery(rover, -self.description.sample_energy())]

    def upd_battery_drop(self, rover: RoversDescription.Rover) -> list:
        return [self.update_battery(rover, -self.description.drop_energy())] if self.description.drop_energy() > 0 else []

    def upd_battery_take_image(self, rover: RoversDescription.Rover) -> list:
        return [self.update_battery(rover, -self.description.take_image_energy())]

    def set_battery(self, rover: RoversDescription.Rover, value: int):
        return JaniStructureGenerator.generate_assignment(self.rover_battery_vars[rover], value)

    # model generation #################################################################################################

    def gen_bounded_int_var(self, var: str, upper_bound: int, lower_bound: int = 0):
        return JaniStructureGenerator.generate_bounded_int_variable(var, lower_bound, upper_bound, self.initial_state[var])

    def gen_rover_vars(self, rover: RoversDescription.Rover) -> list:
        rover_vars = [self.gen_bounded_int_var(self.rover_vars_x[rover], lower_bound=(-1 if self.enable_oom_moves else 0), upper_bound=self.description.x_dim - (0 if self.enable_oom_moves else 1))]
        rover_vars += [self.gen_bounded_int_var(self.rover_vars_y[rover], lower_bound=(-1 if self.enable_oom_moves else 0), upper_bound=self.description.y_dim - (0 if self.enable_oom_moves else 1))]
        rover_vars += [self.gen_bounded_int_var(self.rover_battery_vars[rover], rover.battery + (1 if self.enable_battery_overload else 0), 1 - self.description.max_energy_per_step())]
        rover_vars += [self.gen_bounded_int_var(self.rover_rock_vars[rover], min(self.description.rocks_in_total(), rover.capacity)), self.gen_bounded_int_var(self.rover_soil_vars[rover], min(self.description.soil_in_total(), rover.capacity))]
        # rover_vars += [self.gen_bounded_int_var(self.rover_image_vars[rover], self.description.num_cells(), -1)]
        if self.description.has_special_cap():
            rover_vars += [self.gen_bounded_int_var(self.rover_vars_x_next[rover], lower_bound=(-1 if self.enable_oom_moves else 0), upper_bound=self.description.x_dim - (0 if self.enable_oom_moves else 1)),
                           self.gen_bounded_int_var(self.rover_vars_y_next[rover], lower_bound=(-1 if self.enable_oom_moves else 0), upper_bound=self.description.y_dim - (0 if self.enable_oom_moves else 1))]
        elif self.description.has_default_cap():
            rover_vars += [self.gen_bounded_int_var(self.last_cap_diff, lower_bound=self.description.default_path_capacity - self.description.samples_in_total(), upper_bound=self.description.samples_in_total())]
        #
        return rover_vars

    def gen_cell_vars(self, cell: RoversDescription.Cell) -> list:
        return [self.gen_bounded_int_var(self.cell_rock_vars[cell], self.description.rocks_in_total()),
                self.gen_bounded_int_var(self.cell_soil_vars[cell], self.description.soil_in_total())]

    def gen_objective_vars(self) -> list:
        return [self.gen_bounded_int_var(var, 1) for var in self.objective_vars.values()]

    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.ENABLE_BATTERY_OVERLOAD_NAME, JaniStructureGenerator.generate_bool_type(), self.enable_battery_overload),
                JaniStructureGenerator.generate_constant_declaration(self.ENABLE_OOM_MOVES_NAME, JaniStructureGenerator.generate_bool_type(), self.enable_oom_moves),
                JaniStructureGenerator.generate_constant_declaration(self.FAIL_PROB_CHARGE_NAME, JaniStructureGenerator.generate_real_type(), self.fail_prob_charge),
                JaniStructureGenerator.generate_constant_declaration(self.FAIL_PROB_SAMPLE_NAME, JaniStructureGenerator.generate_real_type(), self.fail_prob_sample),
                JaniStructureGenerator.generate_constant_declaration(self.FAIL_PROB_IMAGE_NAME, JaniStructureGenerator.generate_real_type(), self.fail_prob_image),
                JaniStructureGenerator.generate_constant_declaration(self.USE_POLICY, JaniStructureGenerator.generate_bool_type(), self.use_policy),
                JaniStructureGenerator.generate_constant_declaration(self.USE_MULTI_LOC_POLICY, JaniStructureGenerator.generate_bool_type(), self.use_multi_loc_policy)
                ]

    def generate_variables(self):
        variables = list()
        for rover in self.description.range_rovers():
            variables += self.gen_rover_vars(rover)
        for cell in self.description.range_cells():
            variables += self.gen_cell_vars(cell)
        variables += self.gen_objective_vars()
        return variables

    def generate_locations(self) -> list:
        locations = [RoversModelGenerator.CHOICE_LOC]
        if self.description.has_special_cap():
            locations += [RoversModelGenerator.ERROR_LOC] + list(self.check_move_locations.values())
        return JaniStructureGenerator.generate_locations(locations)

    #

    def generate_edges(self):

        def annotate_edge(edge: json, rover: RoversDescription.Rover, annotate_battery: bool = True, annotate_in_bounds: bool = True) -> json:
            guard_annotation = list()
            guard_annotation += [self.has_battery(rover)] if annotate_battery else []  # "guard" in edge and "battery" not in str(edge["guard"]["exp"])
            guard_annotation += [self.is_in_bounds(rover)] if self.enable_oom_moves and annotate_in_bounds else []
            guard_annotation += [self.safe_per_last_cap_diff()] if self.description.has_only_default_cap() else []
            return JaniStructureGenerator.annotate_edge_guard(edge, Je.And(guard_annotation)) if len(guard_annotation) > 0 else edge

        def gen_charge_edge(rover: RoversDescription.Rover) -> json:
            action = self.action_labels_charge[rover]
            guard = Je.Or([self.is_on_cell(rover, cell) for cell in self.description.charger])
            guard = Je.And(guard, self.has_battery(rover, min_b=0, max_b=(rover.battery - (0 if self.enable_battery_overload else 1))))
            destinations = list()
            destinations.append(JaniStructureGenerator.generate_destination(self.CHOICE_LOC, [self.update_battery(rover, 1)], 1 - self.fail_prob_charge))
            destinations += [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, [], self.fail_prob_charge)] if self.fail_prob_charge > 0 else []
            return JaniStructureGenerator.generate_edge(self.CHOICE_LOC, destinations, action, guard)

        def gen_move_edge(rover: RoversDescription.Rover, direction: str) -> json:
            action = self.action_labels_move[(rover, direction)]
            guard = None if self.enable_oom_moves else self.move_in_bounds(rover, direction)
            assignments = self.update_position(rover, direction, next_vars=self.description.has_special_cap())
            assignments += [self.update_battery_move(rover)]
            assignments += self.set_last_cap_diff(rover) if self.description.has_only_default_cap() else []
            destinations = [JaniStructureGenerator.generate_destination(self.check_move_locations[(rover, direction)] if self.description.has_special_cap() else self.CHOICE_LOC, assignments=assignments)]
            return JaniStructureGenerator.generate_edge(self.CHOICE_LOC, destinations, action, guard)

        def gen_check_move(rover: RoversDescription.Rover, direction: str) -> list:
            assert self.description.has_special_cap()
            source_loc = self.check_move_locations[(rover, direction)]
            # error
            g_error = self.exceeded_path_capacity(rover, direction, next_vars=False)
            e_error = JaniStructureGenerator.generate_edge(location=source_loc, destinations=[JaniStructureGenerator.generate_destination(self.ERROR_LOC, [self.set_battery(rover, -1)])], guard=g_error)
            # no error
            g_safe = self.respect_path_capacity(rover, direction, next_vars=False)
            e_safe = JaniStructureGenerator.generate_edge(location=source_loc, destinations=[JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.update_position_next(rover, direction))], guard=g_safe)
            return [e_error, e_safe]

        def gen_sample_edges(rover: RoversDescription.Rover, cell: RoversDescription.Cell) -> list:
            # TODO may be optimized using ite assignments
            action = self.action_labels_sample[rover]
            g1 = self.rock_on_cell(cell, rover)
            a1 = [self.upd_rock_on_cell(cell, -1), self.upd_rock_on_rover(rover, 1)]
            d1 = [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, a1 + self.upd_battery_sample(rover), 1 - self.fail_prob_sample)]
            g2 = self.soil_on_cell(cell, rover)
            a2 = [self.upd_soil_on_cell(cell, -1), self.upd_soil_on_rover(rover, 1)]
            d2 = [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, a2 + self.upd_battery_sample(rover), 1 - self.fail_prob_sample)]
            g3 = Je.And(self.rock_on_cell(cell, rover), self.soil_on_cell(cell, rover))
            d3 = [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, a1 + a2 + self.upd_battery_sample(rover), 1 - self.fail_prob_sample)]
            if self.fail_prob_sample > 0:
                d1.append(JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.upd_battery_sample(rover), self.fail_prob_sample))
                d2.append(JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.upd_battery_sample(rover), self.fail_prob_sample))
                d3.append(JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.upd_battery_sample(rover), self.fail_prob_sample))
            return [JaniStructureGenerator.generate_edge(self.CHOICE_LOC, d, action, g) for d, g in zip([d1, d2, d3], [g1, g2, g3])]

        def gen_drop_edges(rover: RoversDescription.Rover, cell: RoversDescription.Cell) -> list:
            action = self.action_labels_sample_drop[rover]
            guard = Je.And(self.samples_on_rover(rover), self.is_on_cell(rover, cell))
            destination = [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.drop_rock(rover, cell) + self.drop_soil(rover, cell) + self.upd_battery_drop(rover))]
            return [JaniStructureGenerator.generate_edge(self.CHOICE_LOC, destination, action, guard)]

        def gen_take_image_edge(rover: RoversDescription.Rover, objective: RoversDescription.Cell) -> json:
            action = self.action_labels_take_image[rover]
            guard = self.is_on_cell(rover, objective)
            destinations = [JaniStructureGenerator.generate_destination(self.CHOICE_LOC, [self.take_image(objective)] + self.upd_battery_take_image(rover), 1 - self.fail_prob_image)]
            if self.fail_prob_image > 0:
                destinations.append(JaniStructureGenerator.generate_destination(self.CHOICE_LOC, self.upd_battery_take_image(rover), self.fail_prob_sample))
            return JaniStructureGenerator.generate_edge(self.CHOICE_LOC, destinations, action, guard)

        edges = list()
        for rover_ in self.description.rovers:
            edges += [annotate_edge(gen_charge_edge(rover_), rover_, annotate_battery=False, annotate_in_bounds=False)]
            for direction_ in self.description.DIRECTION_NAMES:
                edges += [annotate_edge(gen_move_edge(rover_, direction_), rover_)]
                edges += gen_check_move(rover_, direction_) if self.description.has_special_cap() else []
            for cell_ in self.description.range_cells():
                edges += [annotate_edge(e, rover_, annotate_in_bounds=False) for e in gen_sample_edges(rover_, cell_)]
                edges += [annotate_edge(e, rover_, annotate_in_bounds=False) for e in gen_drop_edges(rover_, cell_)]
            for objective_ in self.description.objectives:
                edges += [annotate_edge(gen_take_image_edge(rover_, objective_), rover_, annotate_in_bounds=False)]
            if self.use_policy and self.use_multi_loc_policy:
                edges.append(JaniStructureGenerator.generate_edge(self.CHOICE_LOC, [JaniStructureGenerator.generate_destination(self.CHOICE_LOC)], self.aux_actions[rover_][0]))

        return edges

    def generate_automaton(self) -> json:
        return JaniStructureGenerator.generate_automaton(self.AUTOMATON_NAME, self.generate_locations(), [self.CHOICE_LOC], self.generate_edges())

# policy automaton #####################################################################################################

    # policy locations
    CHARGE_LOC = "check_charge"
    DROP_LOC = "check_drop"
    RETURN_LOC = "check_return"
    TAKE_IMAGE_LOC = "check_take_image"
    SAMPLE_LOC = "check_sampling"
    SEARCH_LOC = "check_search"
    # ITE POLICY
    ITE_POLICY_LOC = "policy_loc"

    def generate_ite_policy(self, rover: RoversDescription.Rover):
        guards = list()
        actions = list()
        # if at charger, then charge until full
        g_charge = Je.And(Je.Or([self.is_on_cell(rover, cell) for cell in self.description.charger]), self.has_battery(rover, min_b=0, max_b=(rover.battery - 1)))
        guards.append(g_charge)
        actions.append(self.action_labels_charge[rover])

        # if have rock or soil and at lander,
        g_drop_lander_1 = self.is_on_cell(rover, self.description.lander)
        # if have rock/soil and some rock/soil at lander,
        g_rocks = Je.And(self.rock_on_rover(rover), Je.Not(self.rock_on_cell(self.description.lander)))  # have rock but no rocks at lander
        g_soil = Je.And(self.soil_on_rover(rover), Je.Not(self.soil_on_cell(self.description.lander)))  # have soil but no soil at lander
        g_drop_lander_2 = Je.Not(Je.Or(g_rocks, g_soil))  # negate disjunction of both cases
        g_drop_lander = Je.And(self.samples_on_rover(rover), Je.Or(g_drop_lander_1, g_drop_lander_2))
        g_non_lander_drop_dis = list()
        if self.description.has_special_cap() or self.description.has_default_cap():
            # if load exceeds next min capacity; for simplicity just drop if minimum cap is exceeded
            g_drop_cap = Je.Ge(self.load_on_rover(rover), self.description.extract_min_cap())
            g_non_lander_drop_dis.append(g_drop_cap)
        # if battery <= manhatten * energy_per_move + amount * energy_per_rock_and_soil and have load, then drop (i.e., try to move load closer to lander as long as remaining battery suffices to drive back after drop)
        move_energy = Je.Mult(self.manhatten_distance(rover, self.description.lander), self.description.move_energy())
        load_energy = Je.Add(Je.Mult(self.rover_rock_vars[rover], self.description.move_energy_per_rock()), Je.Mult(self.rover_soil_vars[rover], self.description.move_energy_per_soil()))
        g_drop_battery = Je.And(self.has_battery(rover, 0, Je.Add(move_energy, load_energy)), self.samples_on_rover(rover))
        g_non_lander_drop_dis.append(g_drop_battery)
        #
        assert self.description.charger[0] == self.description.lander and 1 == len(self.description.charger)  # for now no need to exclude g_charge, at charger/lander we may just drop
        g_drop = Je.Or([g_drop_lander] + g_non_lander_drop_dis)
        # then drop
        guards.append(g_drop)
        actions.append(self.action_labels_sample_drop[rover])

        # if battery <= manhatten * (energy_per_move) + 1,
        move_energy = Je.Add(Je.Mult(self.manhatten_distance(rover, self.description.lander), self.description.move_energy()), 1)
        g_move_battery = self.has_battery(rover, 0, move_energy)
        # if have rock/soil and not rock/soil at lander,
        g_move_sample = self.samples_on_rover(rover)  # suffices (wrt. g_drop) due to ite-structure (see below)
        # if rock & soil on lander and images taken,
        g_move_finished = Je.And([self.rock_on_cell(self.description.lander), self.soil_on_cell(self.description.lander)] + [self.image_taken(objective) for objective in self.description.objectives])
        # then return to charger/lander:
        g_ret = Je.Or(g_move_battery, g_move_sample, g_move_finished)
        assert self.description.lander_x() == 0 and self.description.lander_y() == 0  # for now must only exclude drops at non-lander places, as move is blocked at lander
        g_ret = Je.And(g_ret, Je.Not(Je.Or(g_non_lander_drop_dis))) if not self.use_multi_loc_policy else g_ret
        # if x > 0, move left
        guards.append(Je.And(Je.Ge(self.rover_vars_x[rover], 1), g_ret))
        actions.append(self.action_labels_move[(rover, "left")])
        # if y > 0, move down
        guards.append(Je.And(Je.Ge(self.rover_vars_y[rover], 1), g_ret))
        actions.append(self.action_labels_move[(rover, "down")])

        if self.description.has_objectives():
            # if sufficient battery and on objective cell, then take image
            g_take_image_ = Je.And(self.has_battery(rover, self.description.take_image_energy()), Je.Or([Je.And(self.is_on_cell(rover, objective), self.image_taken(objective, truth_value=False)) for objective in self.description.objectives]))
            # For now only exclude battery relevant reactions (however, may then have to unnecessarily drop loaded samples in subsequent steps):
            g_take_image = Je.And(g_take_image_, Je.Not(g_drop_battery), Je.Not(g_move_battery)) if self.USE_POLICY else g_take_image_
            guards.append(g_take_image)
            actions.append(self.action_labels_take_image[rover])
        else:
            g_take_image_ = None

        # if sufficient battery and not some rock/soil on lander and not rock/soil on rover and on cell with rock/soil, then sample
        on_cell_dis_rocks = Je.Or([self.rock_on_cell(cell, rover) for cell in self.description.range_cells()])
        con_rocks = Je.And(Je.Not(self.rock_on_cell(self.description.lander)), Je.Not(self.rock_on_rover(rover)), on_cell_dis_rocks)
        on_cell_dis_soil = Je.Or([self.soil_on_cell(cell, rover) for cell in self.description.range_cells()])
        con_soil = Je.And(Je.Not(self.soil_on_cell(self.description.lander)), Je.Not(self.soil_on_rover(rover)), on_cell_dis_soil)
        g_sample_ = Je.And(self.has_battery(rover, self.description.sample_energy()), Je.Or(con_rocks, con_soil))
        # Charge & drop at lander is implicitly excluded as we do not sample on lander;
        # Again: Only exclude battery relevant reactions. (However, may end up in drop-sample loop due to g_drop_cap, or again unnecessarily drop if loaded other sample type already).
        g_sample = Je.And(g_sample_, Je.Not(g_drop_battery), Je.Not(g_move_battery)) if not self.use_multi_loc_policy else g_sample_
        guards.append(g_sample)
        actions.append(self.action_labels_sample[rover])

        # search:
        # if x is even and y < max, then move up
        g_up = Je.And(Je.Le(self.rover_vars_y[rover], self.description.y_dim - 2), Je.Or([Je.Eq(self.rover_vars_x[rover], val) for val in range(0, self.description.x_dim) if val % 2 == 0]))
        # if x is even (and x < max) and y == max, then move right
        g_right = Je.And(Je.Ge(self.rover_vars_y[rover], self.description.y_dim - 1), Je.Le(self.rover_vars_x[rover], self.description.x_dim - 2), Je.Or([Je.Eq(self.rover_vars_x[rover], val) for val in range(0, self.description.x_dim) if val % 2 == 0]))
        # if x is odd and y > 0, then move down
        g_down = Je.And(Je.Ge(self.rover_vars_y[rover], 1), Je.Or([Je.Eq(self.rover_vars_x[rover], val) for val in range(0, self.description.x_dim) if val % 2 == 1]))
        # if x is odd (and x < max) and y == 0, then move right
        g_right = Je.Or(g_right, Je.And(Je.Le(self.rover_vars_y[rover], 0), Je.Le(self.rover_vars_x[rover], self.description.x_dim - 2), Je.Or([Je.Eq(self.rover_vars_x[rover], val) for val in range(0, self.description.x_dim) if val % 2 == 1])))
        # if x == max and not "up or down" (dummy)
        g_left = Je.And(Je.Not(Je.Or(g_up, g_down)), Je.Ge(self.rover_vars_x[rover], self.description.x_dim - 1))
        g_search = [g_down, g_up, g_right, g_left]
        # g_drop is excluded by negation of samples_on_rover (as well as g_move_sample):
        if not self.use_multi_loc_policy:
            g_search_single_loc = [Je.Not(g_charge), self.samples_on_rover(rover, truth_value=False), Je.Not(g_move_battery), Je.Not(g_move_finished)]
            g_search_single_loc += [Je.Not(g_take_image_)] if self.description.has_objectives() else []
            g_search_single_loc += [Je.Not(g_sample_)]
            g_search = [Je.And([g] + g_search_single_loc) for g in g_search]
        guards += g_search
        actions += [self.action_labels_move[(rover, direction)] for direction in ["down", "up", "right", "left"]]
        return guards, actions, g_ret

    def generate_policy_edges(self, rover: RoversDescription.Rover) -> json:

        def generate_edge(src_loc, guard, action, succ_loc, alt_loc) -> list:
            edges_ = [JaniStructureGenerator.generate_edge(src_loc, [JaniStructureGenerator.generate_destination(succ_loc)], action, guard)]
            if alt_loc is not None:
                edges_.append(JaniStructureGenerator.generate_edge(src_loc, [JaniStructureGenerator.generate_destination(alt_loc)], self.aux_actions[rover][1], Je.Not(guard)))
            return edges_

        src_locs = [self.CHARGE_LOC, self.DROP_LOC, self.RETURN_LOC, self.RETURN_LOC]
        src_locs += [self.TAKE_IMAGE_LOC] if self.description.has_objectives() else []
        src_locs += [self.SAMPLE_LOC, self.SEARCH_LOC, self.SEARCH_LOC, self.SEARCH_LOC, self.SEARCH_LOC]
        alt_locs = [self.DROP_LOC, self.RETURN_LOC]
        alt_locs += [None, None, self.SAMPLE_LOC] if self.description.has_objectives() else [self.SAMPLE_LOC, self.SAMPLE_LOC]
        alt_locs += [self.SEARCH_LOC, None, None, None, None]
        guards, actions, g_ret = self.generate_ite_policy(rover)
        edges = list()
        assert len(actions) == len(guards) == (10 if self.description.has_objectives() else 9)
        if not self.use_multi_loc_policy:
            for g, a in zip(guards, actions):
                edges += generate_edge(self.policy_init_loc, g, a, self.policy_init_loc, alt_loc=None)
            return edges
        # else:
        for i in range(0, len(guards)):
            edges += generate_edge(src_locs[i], guards[i], actions[i], self.policy_init_loc, alt_locs[i])
        # manually alt edge for return to base
        alt_loc = self.TAKE_IMAGE_LOC if self.description.has_objectives() else self.SAMPLE_LOC
        edges.append(JaniStructureGenerator.generate_edge(self.RETURN_LOC, [JaniStructureGenerator.generate_destination(alt_loc)], action=self.aux_actions[rover][1], guard=Je.Or(Je.Not(g_ret), self.is_on_cell(rover, self.description.lander))))
        return edges

    def generate_policy_automaton(self, rover: RoversDescription.Rover) -> json:
        return JaniStructureGenerator.generate_automaton(self.policy_automata[rover], JaniStructureGenerator.generate_locations(self.policy_locations), [self.policy_init_loc], self.generate_policy_edges(rover))

    def generate_automata(self) -> list:
        return [self.generate_automaton()] + ([self.generate_policy_automaton(rover) for rover in self.description.rovers] if self.use_policy else [])

    def generate_syncs(self) -> list:
        syncs = list()
        if self.use_policy:
            for rover, index in zip(self.description.rovers, range(1, self.description.num_rovers() + 1)):
                for action_label in self.action_labels:
                    sync = [action_label] + [None for _ in range(1, self.description.num_rovers() + 1)]
                    sync[index] = action_label
                    syncs.append(JaniStructureGenerator.generate_synchronization(sync, action_label))
                if self.use_multi_loc_policy:
                    sync = [None for _ in range(0, self.description.num_rovers() + 1)]
                    sync[0] = self.aux_actions[rover][0]
                    sync[index] = self.aux_actions[rover][1]
                    syncs.append(JaniStructureGenerator.generate_synchronization(sync))
            return syncs
        else:
            JaniModelGeneratorPddlInJani.generate_syncs(self)
        return [JaniStructureGenerator.generate_synchronization([action_label], action_label) for action_label in self.action_labels]

    def generate_composition(self) -> json:
        if self.use_policy:
            return JaniStructureGenerator.generate_composition(JaniStructureGenerator.generate_composition_elements([self.AUTOMATON_NAME] + list(self.policy_automata.values())), self.generate_syncs())
        else:
            return JaniModelGeneratorPddlInJani.generate_composition(self)

    # property generation ##############################################################################################

    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression(self.generate_automaton_location_value(self.CHOICE_LOC), self.generate_goal_expression())
        return JaniStructureGenerator.generate_objective_expression(goal=goal, goal_potential=self.generate_goal_potential())

    def generate_goal_potential(self) -> json:
        if not self.use_goal_potential:
            return None
        additive_list = list()
        for rover in self.description.rovers:
            # has battery
            additive_list.append(Je.Ite(self.has_battery(rover, 0), 10, 0))
            # rover at lander
            additive_list.append(Je.Ite(Je.And(self.is_safe(rover), self.is_on_cell(rover, self.description.lander)), 10, 0))
            # rock & soil sample
            additive_list.append(Je.Ite(Je.And(self.is_safe(rover), Je.Ge(self.cell_rock_vars[self.description.lander], 1)), 10, 0))  # first sample gives extra bonus
            additive_list.append(Je.Ite(Je.And(self.is_safe(rover), Je.Ge(self.cell_soil_vars[self.description.lander], 1)), 10, 0))
            additive_list.append(Je.Ite(self.is_safe(rover), Je.Mult(10, Je.Add(self.cell_rock_vars[self.description.lander], self.cell_soil_vars[self.description.lander])), 0))
            # image sent
            additive_list += [Je.Ite(self.is_safe(rover), Je.Mult(10, var), 0) for var in self.objective_vars.values()]
        goal_potential = Je.Add(additive_list)  # TODO for now unused
        goal_potential = Je.Ite(self.generate_start(), 0, goal_potential) if self.ground_start_potential else goal_potential
        goal = JaniStructureGenerator.generate_state_condition_expression(self.generate_automaton_location_value(self.CHOICE_LOC), self.generate_goal_expression())
        goal_potential = Je.Ite(Je.Or(goal, self.generate_reach()), 0, goal_potential) if self.ground_terminal_potential else goal_potential
        return goal_potential

    def generate_reach(self) -> json:
        state_cond = Je.Not(Je.And([self.is_safe(rover) for rover in self.description.rovers]))
        return JaniStructureGenerator.generate_state_condition_expression([], state_cond)

    # starts ###########################################################################################################

    def generate_start(self):
        constraints = list()
        constraints += [self.is_on_cell(rover, self.description.lander, next_vars=True) for rover in self.description.rovers] if self.description.has_special_cap() else []  # dummy
        # rovers at lander
        constraints += [self.is_on_cell(rover, self.description.lander) for rover in self.description.rovers] if self.rover_at_lander else []
        # rover not oom
        constraints += [self.is_in_bounds(rover) for rover in self.description.rovers] if self.enable_oom_moves and not self.rover_at_lander else []
        # rocks and soil
        constraints.append(Je.Eq(0, Je.Add(list(self.rover_rock_vars.values()))))  # no rocks loaded
        constraints.append(Je.Eq(0, Je.Add(list(self.rover_soil_vars.values()))))
        constraints.append(Je.Eq(self.description.rocks_in_total(), Je.Add(list(self.cell_rock_vars.values()))))  # rocks are distributed on the map
        constraints.append(Je.Eq(self.description.soil_in_total(), Je.Add(list(self.cell_soil_vars.values()))))
        # no images
        constraints += [Je.Eq(objective, 0) for objective in self.objective_vars.values()]
        # battery should suffice to recharge
        for rover in self.description.rovers:
            constraints.append(Je.And(Je.Ge(self.rover_battery_vars[rover], 1 if self.rover_at_lander else 1 * self.description.move_energy() * (self.description.x_dim + self.description.y_dim)), Je.Le(self.rover_battery_vars[rover], rover.battery)))
            constraints += [Je.Le(self.rover_battery_vars[rover], rover.battery)] if self.enable_battery_overload else []

        return Je.And(constraints)

    def generate_compact_starts(self):
        if self.use_policy:
            loc_values = JaniStructureGenerator.generate_location_values([self.AUTOMATON_NAME] + list(self.policy_automata.values()), [self.CHOICE_LOC] + [self.policy_init_loc for _ in range(0, self.description.num_rovers())])
        else:
            loc_values = JaniStructureGenerator.generate_location_values([self.AUTOMATON_NAME], [self.CHOICE_LOC])
        return JaniStructureGenerator.generate_state_condition_expression(loc_values, self.generate_start())

    def generate_random_states(self, number_starts: int) -> list:

        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=number_starts, default_state=self.initial_state)

        # 25 %: actual start states: i.e. randomly distribute rocks and soil
        rocks_sum = self.description.rocks_in_total()
        soil_sum = self.description.soil_in_total()
        while states_values.size() < int(number_starts * 0.25):
            rock_vector = PythonUtils.generate_random_vector(self.description.num_cells(), rocks_sum)
            state = JaniModelGenerator.generate_state(self.cell_rock_vars.values(), rock_vector, self.initial_state)
            soil_vector = PythonUtils.generate_random_vector(self.description.num_cells(), soil_sum)
            state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_soil_vars.values(), soil_vector, self.initial_state))
            rlt = states_values.add(state)
            if rlt is not None:
                return rlt

        # 10 %: close to goal
        # image sent, one rocks/soil on rover, enough battery
        if True:
            while states_values.size() < int(number_starts * 0.35):
                state = dict()
                # image sent
                for obj_var in self.objective_vars.values():
                    state[obj_var] = 1
                # one rocks/soil on rover
                rover = random.choice(self.description.rovers)
                mode = random.choice(["rock", "soil", "rock_soil"])
                state[self.rover_rock_vars[rover]] = min(1, rocks_sum) if "rock" in mode else 0
                state[self.rover_soil_vars[rover]] = min(1, soil_sum) if "soil" in mode else 0
                # random position
                for rover in self.description.rovers:
                    state[self.rover_vars_x[rover]] = random.randint(0, self.description.x_dim - 1)
                    state[self.rover_vars_y[rover]] = random.randint(0, self.description.y_dim - 1)
                    # enough battery
                    manhattan_distance = self.manhatten_distance_val(self.description.lander_x(), self.description.lander_y(), state[self.rover_vars_x[rover]], state[self.rover_vars_y[rover]])
                    energy_for_load = self.description.move_energy_per_rock() * state[self.rover_rock_vars[rover]] + self.description.move_energy_per_soil() * state[self.rover_soil_vars[rover]]
                    state[self.rover_battery_vars[rover]] = random.randint(manhattan_distance * (energy_for_load + self.description.move_energy()), rover.battery)
                # distribute rock & soil on cells
                rock_vector = PythonUtils.generate_random_vector(self.description.num_cells(), rocks_sum - PythonUtils.sum_dict(state, self.rover_rock_vars.values()))
                state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_rock_vars.values(), rock_vector, self.initial_state))
                soil_vector = PythonUtils.generate_random_vector(self.description.num_cells(), soil_sum - PythonUtils.sum_dict(state, self.rover_soil_vars.values()))
                state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_soil_vars.values(), soil_vector, self.initial_state))
                rlt = states_values.add(state)
                if rlt is not None:
                    return rlt

        # 15 % close to unsafe
        if True:
            special_cap_cells = self.description.extract_special_cap_cells()
            while states_values.size() < int(number_starts * 0.5):
                state = dict()
                for rover in self.description.rovers:
                    main_mode = random.choice(["none", "battery_load", "battery_distance", "oom_x", "oom_y"] + (["special_cap"] if self.description.has_special_cap() else []))  # causes of unsafety
                    if "special_cap" in main_mode:
                        # special cap positions:
                        special_cap_cell = random.choice(special_cap_cells)
                        state[self.rover_vars_x[rover]] = special_cap_cell.x
                        state[self.rover_vars_y[rover]] = special_cap_cell.y
                    elif "oom_x" in main_mode:
                        state[self.rover_vars_x[rover]] = random.choice([0, self.description.x_dim - 1])
                        state[self.rover_vars_y[rover]] = random.randint(0, self.description.y_dim - 1)
                    elif "oom_y" in main_mode:
                        state[self.rover_vars_x[rover]] = random.randint(0, self.description.x_dim - 1)
                        state[self.rover_vars_y[rover]] = random.choice([0, self.description.y_dim - 1])
                    else:
                        # random position
                        state[self.rover_vars_x[rover]] = random.randint(0, self.description.x_dim - 1)
                        state[self.rover_vars_y[rover]] = random.randint(0, self.description.y_dim - 1)
                    if "special_cap" in main_mode or "battery_load" in main_mode:
                        # some rocks/soil on rover
                        mode = random.choice(["rock", "soil", "rock_soil"])
                        state[self.rover_rock_vars[rover]] = min(1 if "rock" in mode else 0, rocks_sum)
                        state[self.rover_soil_vars[rover]] = min(1 if "soil" in mode else 0, soil_sum)
                    if "battery_load" in main_mode or "battery_distance" in main_mode:
                        # just sufficient battery
                        manhattan_distance = abs(self.description.lander_x() - state[self.rover_vars_x[rover]]) + abs(self.description.lander_y() - state[self.rover_vars_y[rover]])
                        if self.rover_rock_vars[rover] in state or self.rover_soil_vars[rover] in state:
                            energy_for_load = self.description.move_energy_per_rock() * state[self.rover_rock_vars[rover]] + self.description.move_energy_per_soil() * state[self.rover_soil_vars[rover]]
                        else:
                            energy_for_load = 0
                        minimal_energy = manhattan_distance * (energy_for_load + self.description.move_energy())
                        state[self.rover_battery_vars[rover]] = random.randint(minimal_energy, minimal_energy + 2)
                # distribute rock & soil on cells
                rocks_vars = list(self.rover_rock_vars.values()) + list(self.cell_rock_vars.values())
                soil_vars = list(self.rover_soil_vars.values()) + list(self.cell_soil_vars.values())
                rock_vector = PythonUtils.generate_random_vector(self.description.num_cells(), rocks_sum - PythonUtils.sum_dict(state, rocks_vars))
                state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_rock_vars.values(), rock_vector, self.initial_state))
                soil_vector = PythonUtils.generate_random_vector(self.description.num_cells(), soil_sum - PythonUtils.sum_dict(state, soil_vars))
                state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_soil_vars.values(), soil_vector, self.initial_state))
                rlt = states_values.add(state)
                if rlt is not None:
                    return rlt

        # rest: random states
        while states_values.size() < int(number_starts):
            state = dict()
            # random position
            for rover in self.description.rovers:
                state[self.rover_vars_x[rover]] = random.randint(0, self.description.x_dim - 1)
                state[self.rover_vars_y[rover]] = random.randint(0, self.description.y_dim - 1)
                # rocks/soil on rover
                state[self.rover_rock_vars[rover]] = random.randint(0, rocks_sum - PythonUtils.sum_dict(state, self.rover_rock_vars.values()))
                state[self.rover_soil_vars[rover]] = random.randint(0, soil_sum - PythonUtils.sum_dict(state, self.rover_soil_vars.values()))
                # (sufficient) battery
                manhattan_distance = abs(self.description.lander_x() - state[self.rover_vars_x[rover]]) + abs(self.description.lander_y() - state[self.rover_vars_y[rover]])
                energy_for_load = self.description.move_energy_per_rock() * state[self.rover_rock_vars[rover]] + self.description.move_energy_per_soil() * state[self.rover_soil_vars[rover]]
                state[self.rover_battery_vars[rover]] = random.randint(min(manhattan_distance * (energy_for_load + self.description.move_energy()), rover.battery), rover.battery)
            # distribute rock & soil on cells
            rocks_vars = list(self.rover_rock_vars.values()) + list(self.cell_rock_vars.values())
            soil_vars = list(self.rover_soil_vars.values()) + list(self.cell_soil_vars.values())
            rock_vector = PythonUtils.generate_random_vector(self.description.num_cells(), rocks_sum - PythonUtils.sum_dict(state, rocks_vars))
            state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_rock_vars.values(), rock_vector, self.initial_state))
            soil_vector = PythonUtils.generate_random_vector(self.description.num_cells(), soil_sum - PythonUtils.sum_dict(state, soil_vars))
            state = PythonUtils.update_dict(state, JaniModelGenerator.generate_state(self.cell_soil_vars.values(), soil_vector, self.initial_state))
            # image sent
            for obj_var in self.objective_vars.values():
                state[obj_var] = random.randint(0, 1)
            #
            rlt = states_values.add(state)
            if rlt is not None:
                return rlt

        return states_values.generate_states_values()

    # predicate generation #############################################################################################

    # nn input #########################################################################################################

    def get_nn_inputs(self) -> list:
        nn_vars = list()
        for rover in self.description.rovers:
            nn_vars.append(self.rover_vars_x[rover])
            nn_vars.append(self.rover_vars_y[rover])
            nn_vars.append(self.rover_battery_vars[rover])
            nn_vars.append(self.rover_rock_vars[rover])
            nn_vars.append(self.rover_soil_vars[rover])
        for objective in self.description.objectives:
            nn_vars.append(self.objective_vars[objective])
        # TODO some kind of interface to compress rock & soil information (e.g. only current position)
        for cell in self.description.range_cells():
            nn_vars.append(self.cell_rock_vars[cell])
            nn_vars.append(self.cell_soil_vars[cell])
        # rocks and soil at lander:
        # nn_vars.append(self.cell_rock_vars[self.description.lander])
        # nn_vars.append(self.cell_soil_vars[self.description.lander])
        #
        return nn_vars

    ####################################################################################################################


if __name__ == "__main__":
    args = RoversModelGenerationOptionParser().arg_parse()
    generator = RoversModelGenerator(args)
    generator.generate()
