import json
import os
import random
import math
from frozendict import frozendict
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JeEval, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils


random.seed(2020)


class FlightDiversionModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--oob-crash", type=bool, default=False, help="Crash if flight goes out of bounds")
        # self.optionParser.add_argument("--timestep", type=float, default=0.1, help="Time step for each episode.")
        # self.optionParser.add_argument("--velocity", type=float, default=28, help="Constant velocity of lead car.")


class FlightDiversionModelGenerator(JaniModelGeneratorPddlInJani):
    position_x = "pos_x"
    position_y = "pos_y"
    position_z = "pos_z"
    velocity_x = "vel_x"
    velocity_y = "vel_y"
    velocity_z = "vel_z"
    fuel = "fuel"
    crash = "crash"
    OOB_CRASH = "oob-crash"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.action_labels = ["accelerate_x", "accelerate_y", "accelerate_z", "decelerate_x", "decelerate_y", "decelerate_z"]
        if self.model_file is None:
            # model generation
            self.description_name = options.description
            self.oob_crash = options.oob_crash
        else:
            # property generation
            constants = self.model["constants"]
            self.description_name = self.model["name"]
            _, self.oob_crash = self.read_constant(constants, self.OOB_CRASH)

        self.model_name = self.description_name
        self.description = PythonUtils.load_json(self.description_name)
        print(self.description["bounds"])
        self.lb_x, self.ub_x = self.description["bounds"][0]["x"]
        self.lb_y, self.ub_y = self.description["bounds"][1]["y"]
        self.lb_z, self.ub_z = self.description["bounds"][2]["z"]
        self.fuel_capacity = self.description["fuel_capacity"]
        self.max_speed = self.description["max_speed"]
        self.lb_vel, self.ub_vel = -self.max_speed, self.max_speed
        
        self.unsafe_regions = self.description["unsafe_regions"]
        self.goal_region = self.description["goal_region"]
        # self.timestep_value = options.timestep
        # self.lead_velocity_value = options.velocity

    def compute_model_initial_and_goal_state(self):
        pass

    def generate_constants(self) -> list:
        return [JaniStructureGenerator.generate_constant_declaration(
            self.TERMINAL_AT_UNSAFE,
            JaniStructureGenerator.generate_bool_type(),
            False
        ),
        JaniStructureGenerator.generate_constant_declaration(
            self.OOB_CRASH,
            JaniStructureGenerator.generate_bool_type(),
            self.oob_crash
        )]

    def generate_variables(self) -> list:
        variables = [
                    JaniStructureGenerator.generate_bounded_real_variable(self.position_x, self.lb_x, self.ub_x, (self.lb_x + self.ub_x)/2),
                    JaniStructureGenerator.generate_bounded_real_variable(self.position_y, self.lb_y, self.ub_y, (self.lb_y + self.ub_y)/2),
                    JaniStructureGenerator.generate_bounded_real_variable(self.position_z, self.lb_z, self.ub_z, (self.lb_z + self.ub_z)/2),
                    JaniStructureGenerator.generate_bounded_real_variable(self.velocity_x, self.lb_vel, self.ub_vel, 0),
                    JaniStructureGenerator.generate_bounded_real_variable(self.velocity_y, self.lb_vel, self.ub_vel, 0),
                    JaniStructureGenerator.generate_bounded_real_variable(self.velocity_z, self.lb_vel, self.ub_vel, 0),
                    JaniStructureGenerator.generate_bounded_real_variable(self.fuel, 0, self.fuel_capacity, self.fuel_capacity),
                ]
        variables += [JaniStructureGenerator.generate_bounded_real_variable(self.crash, 0, 1, 0)] if self.oob_crash else []
        return variables


    def __crash(self):
        return [JaniStructureGenerator.generate_assignment(self.crash, 1)]
    
    def acc_x_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_x, 1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]

    def dec_x_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_x, -1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]
    
    def acc_y_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_y, 1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]

    def dec_y_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_y, -1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]
    
    def acc_z_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_z, 1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]

    def dec_z_only(self):
        return [JaniStructureGenerator.generate_self_assignment(self.velocity_z, -1), JaniStructureGenerator.generate_self_assignment(self.fuel, -1)]
    
    def __accelerate_x(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_x, Je.Add(self.velocity_x, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, self.velocity_x, 1)), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, self.velocity_y)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, self.velocity_z))]

    def __accelerate_y(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_y, Je.Add(self.velocity_y, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, self.velocity_y, 1)), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, self.velocity_x)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, self.velocity_z))]

    def __accelerate_z(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_z, Je.Add(self.velocity_z, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, self.velocity_z, 1)), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, self.velocity_y)), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, self.velocity_x))]

    def __decelerate_x(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_x, Je.Sub(self.velocity_x, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, Je.Sub(self.velocity_x, 1))), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, self.velocity_y)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, self.velocity_z))]

    def __decelerate_y(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_y, Je.Sub(self.velocity_y, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, Je.Sub(self.velocity_y, 1))), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, self.velocity_x)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, self.velocity_z))]

    def __decelerate_z(self):
        return [JaniStructureGenerator.generate_assignment(self.velocity_z, Je.Sub(self.velocity_z, 1)), JaniStructureGenerator.generate_assignment(self.fuel, Je.Sub(self.fuel, 1)), JaniStructureGenerator.generate_assignment(self.position_z, Je.Add(self.position_z, Je.Sub(self.velocity_z, 1))), JaniStructureGenerator.generate_assignment(self.position_y, Je.Add(self.position_y, self.velocity_y)), JaniStructureGenerator.generate_assignment(self.position_x, Je.Add(self.position_x, self.velocity_x))]

    def generate_edges(self) -> list:
        def accelerate_x_edge():
            return [
                JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[
                    JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.__accelerate_x())],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Ge(self.fuel, 1),
                    Je.Le(self.velocity_x, self.ub_vel - 1),
                    Je.Le(Je.Add(self.position_x, self.velocity_x, 1), self.ub_x),
                    Je.Le(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                    Je.Le(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                    Je.Ge(Je.Add(self.position_x, self.velocity_x, 1), self.lb_x),
                    Je.Ge(Je.Add(self.position_y, self.velocity_y), self.lb_y),
                    Je.Ge(Je.Add(self.position_z, self.velocity_z), self.lb_z)
                )
            ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0", assignments= self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.acc_x_only())],
                    action=self.action_labels[0],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Le(self.velocity_x, self.ub_vel - 1),
                        Je.Or(Je.Ge(Je.Add(self.position_x, self.velocity_x, 1), self.ub_x),
                        Je.Ge(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                        Je.Ge(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Le(Je.Add(self.position_x, self.velocity_x, 1), self.lb_x),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.lb_y),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.lb_z))
                    )
                ),
            ]

        def accelerate_y_edge():
            return [
                JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[
                    JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.__accelerate_y())],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Ge(self.fuel, 1),
                    Je.Le(self.velocity_y, self.ub_vel - 1),
                    Je.Le(Je.Add(self.position_y, self.velocity_y, 1), self.ub_y),
                    Je.Le(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                    Je.Le(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                    Je.Ge(Je.Add(self.position_y, self.velocity_y, 1), self.lb_y),
                    Je.Ge(Je.Add(self.position_x, self.velocity_x), self.lb_x),
                    Je.Ge(Je.Add(self.position_z, self.velocity_z), self.lb_z)
                )
            ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.acc_y_only())],
                    action=self.action_labels[1],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Le(self.velocity_y, self.ub_vel - 1),
                        Je.Or(Je.Ge(Je.Add(self.position_y, self.velocity_y, 1), self.ub_y),
                        Je.Ge(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                        Je.Ge(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Le(Je.Add(self.position_y, self.velocity_y, 1), self.lb_y),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.lb_x),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.lb_z))
                    )
                ),
            ]

        def accelerate_z_edge():
            return [
                JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[
                    JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.__accelerate_z())],
                action=self.action_labels[2],
                guard=Je.And(
                    Je.Ge(self.fuel, 1),
                    Je.Le(self.velocity_z, self.ub_vel - 1),
                    Je.Le(Je.Add(self.position_z, self.velocity_z, 1), self.ub_z),
                    Je.Le(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                    Je.Le(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                    Je.Ge(Je.Add(self.position_z, self.velocity_z, 1), self.lb_z),
                    Je.Ge(Je.Add(self.position_x, self.velocity_x), self.lb_x),
                    Je.Ge(Je.Add(self.position_y, self.velocity_y), self.lb_y)
                )
            ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.acc_z_only())],
                    action=self.action_labels[2],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Le(self.velocity_z, self.ub_vel - 1),
                        Je.Or(Je.Ge(Je.Add(self.position_z, self.velocity_z, 1), self.ub_z),
                        Je.Ge(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                        Je.Ge(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                        Je.Le(Je.Add(self.position_z, self.velocity_z, 1), self.lb_z),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.lb_x),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.lb_y))
                    )
                ),
            ]

        def decelerate_x_edge():
            return [
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",
                                                                    assignments=self.__decelerate_x())],
                    action=self.action_labels[3],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_x, self.lb_vel + 1),
                        Je.Le(Je.Add(self.position_x, Je.Sub(self.velocity_x, 1)), self.ub_x),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Ge(Je.Add(self.position_x, Je.Sub(self.velocity_x, 1)), self.lb_x),
                        Je.Ge(Je.Add(self.position_z, self.velocity_z), self.lb_z),
                        Je.Ge(Je.Add(self.position_y, self.velocity_y), self.lb_y)
                    )
                ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.dec_x_only())],
                    action=self.action_labels[3],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_x, self.lb_vel + 1),
                        Je.Or(Je.Ge(Je.Add(self.position_x, Je.Sub(self.velocity_x, 1)), self.ub_x),
                              Je.Ge(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                              Je.Ge(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Le(Je.Add(self.position_x, Je.Sub(self.velocity_x, 1)), self.lb_x),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.lb_z),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.lb_y))
                    )
                ),
            ]

        def decelerate_y_edge():
            return [
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__decelerate_y())],
                    action=self.action_labels[4],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_y, self.lb_vel + 1),
                        Je.Le(Je.Add(self.position_y, Je.Sub(self.velocity_y, 1)), self.ub_y),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Ge(Je.Add(self.position_y, Je.Sub(self.velocity_y, 1)), self.lb_y),
                        Je.Ge(Je.Add(self.position_z, self.velocity_z), self.lb_z),
                        Je.Ge(Je.Add(self.position_x, self.velocity_x), self.lb_x)
                    )
                ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.dec_y_only())],
                    action=self.action_labels[4],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_y, self.lb_vel + 1),
                        Je.Or(Je.Ge(Je.Add(self.position_y, Je.Sub(self.velocity_y, 1)), self.ub_y),
                              Je.Ge(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                              Je.Ge(Je.Add(self.position_z, self.velocity_z), self.ub_z),
                        Je.Le(Je.Add(self.position_y, Je.Sub(self.velocity_y, 1)), self.lb_y),
                        Je.Le(Je.Add(self.position_z, self.velocity_z), self.lb_z),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.lb_x))
                    )
                ),
            ]

        def decelerate_z_edge():
            return [
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__decelerate_z())],
                    action=self.action_labels[5],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_z, self.lb_vel + 1),
                        Je.Le(Je.Add(self.position_z, Je.Sub(self.velocity_z, 1)), self.ub_z),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                        Je.Ge(Je.Add(self.position_z, Je.Sub(self.velocity_z, 1)), self.lb_z),
                        Je.Ge(Je.Add(self.position_y, self.velocity_y), self.lb_y),
                        Je.Ge(Je.Add(self.position_x, self.velocity_x), self.lb_x)
                    )
                ),
                JaniStructureGenerator.generate_edge(
                    location="loc_0",
                    destinations=[
                        JaniStructureGenerator.generate_destination(location="loc_0",assignments=self.__crash())] if self.oob_crash else [JaniStructureGenerator.generate_destination(location="loc_0", assignments=self.dec_z_only())],
                    action=self.action_labels[5],
                    guard=Je.And(
                        Je.Ge(self.fuel, 1),
                        Je.Ge(self.velocity_z, self.lb_vel + 1),
                        Je.Or(Je.Ge(Je.Add(self.position_z, Je.Sub(self.velocity_z, 1)), self.ub_z),
                              Je.Ge(Je.Add(self.position_y, self.velocity_y), self.ub_y),
                              Je.Ge(Je.Add(self.position_x, self.velocity_x), self.ub_x),
                        Je.Le(Je.Add(self.position_z, Je.Sub(self.velocity_z, 1)), self.lb_z),
                        Je.Le(Je.Add(self.position_x, self.velocity_x), self.lb_x),
                        Je.Le(Je.Add(self.position_y, self.velocity_y), self.lb_y))
                    )
                ),
            ]

        return accelerate_x_edge() + accelerate_y_edge() + accelerate_z_edge() + decelerate_x_edge() + decelerate_y_edge() + decelerate_z_edge()

    def get_jani_expression_for_unsafe_regions(self):
        unsafety = []
        for region in self.unsafe_regions:
            x_min, x_max = region["x"]  # [5, 10]
            y_min, y_max = region["y"]  # [-10, -12.5]
            z_min, z_max = region["z"]  # [7.5, 9]
            unsafety.append(Je.And(Je.Le(self.position_x, x_max), Je.Ge(self.position_x, x_min),
                                   Je.Le(self.position_y, y_max), Je.Ge(self.position_y, y_min),
                                   Je.Le(self.position_z, z_max), Je.Ge(self.position_z, z_min),
                                   ))
        return JaniStructureGenerator.generate_large_disjunction(unsafety)
    
    def get_jani_expression_for_goal_regions(self):
        goal = []
        for region in self.goal_region:
            x_min, x_max = region["x"]  # [5, 10]
            y_min, y_max = region["y"]  # [-10, -12.5]
            z_min, z_max = region["z"]  # [7.5, 9]
            goal.append(Je.And(Je.Le(self.position_x, x_max), Je.Ge(self.position_x, x_min),
                                   Je.Le(self.position_y, y_max), Je.Ge(self.position_y, y_min),
                                   Je.Le(self.position_z, z_max), Je.Ge(self.position_z, z_min),
                                   ))
        return JaniStructureGenerator.generate_large_disjunction(goal)
    
    def __is_unsafe(self):
        if self.oob_crash:
            return Je.Or(Je.Gt(self.crash, 0), Je.Le(self.fuel, 0),
                    self.get_jani_expression_for_unsafe_regions()
                    )
        return Je.Or(Je.Le(self.fuel, 0),
                    self.get_jani_expression_for_unsafe_regions()
                    )
    def generate_goal_expression(self) -> json:
        return self.get_jani_expression_for_goal_regions()
    
    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], self.generate_goal_expression())
        return JaniStructureGenerator.generate_objective_expression(goal=goal)

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.__is_unsafe())

    def generate_start(self) -> list:
        mid_x = (self.lb_x + self.ub_x)/2
        mid_y = (self.lb_y + self.ub_y)/2
        mid_z = (self.lb_z + self.ub_z)/2
        mid_vel = (self.lb_vel + self.ub_vel)/2
        constraints = [
            Je.Ge(self.position_x, mid_x),
            Je.Le(self.position_x, mid_x + 3),
            Je.Ge(self.position_z, mid_z),
            Je.Le(self.position_z, mid_z + 5),
            Je.Ge(self.position_y, mid_y - 2),
            Je.Le(self.position_y, mid_y + 4),
            Je.Eq(self.fuel, self.fuel_capacity),
            Je.Ge(self.velocity_x, mid_vel),
            Je.Le(self.velocity_x, mid_vel + 0.5),
            Je.Ge(self.velocity_y, mid_vel),
            Je.Le(self.velocity_y, mid_vel + 0.7),
            Je.Ge(self.velocity_z, mid_vel - 0.3),
            Je.Le(self.velocity_z, mid_vel + 0.1),
        ]
        constraints += [Je.Le(self.crash, 0)] if self.oob_crash else []
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Generating " + str(max_number_states) + " random states ...")
        mid_x = (self.lb_x + self.ub_x)/2
        mid_y = (self.lb_y + self.ub_y)/2
        mid_z = (self.lb_z + self.ub_z)/2
        mid_vel = (self.lb_vel + self.ub_vel)/2
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state={})
        sample = {}
        while len(sample) < max_number_states:
            assignment = {
                self.position_x: round(random.uniform(mid_x, mid_x + 3), 4),
                self.position_y: round(random.uniform(mid_y - 2, mid_y + 4), 4),
                self.position_z: round(random.uniform(mid_z, mid_z + 5), 4),

                self.velocity_x: round(random.uniform(mid_vel, mid_vel + 0.5), 4),
                self.velocity_y: round(random.uniform(mid_vel, mid_vel + 0.7), 4),
                self.velocity_z: round(random.uniform(mid_vel - 0.3, mid_vel + 0.1), 4),
                self.fuel: self.fuel_capacity,
            }
            if self.oob_crash:
                assignment[self.crash] = 0
            sample[frozendict(assignment)] = assignment.copy()
        for _, state in sample.items():
            states_values.add(state)
        return states_values.generate_states_values()

    def get_nn_inputs(self) -> list:
        nn_inputs = [self.position_x, self.position_y, self.position_z, self.velocity_x, self.velocity_y, self.velocity_z, self.fuel]
        return nn_inputs


if __name__ == "__main__":
    args = FlightDiversionModelGenerationOptionParser().arg_parse()
    generator = FlightDiversionModelGenerator(args)
    generator.generate()
