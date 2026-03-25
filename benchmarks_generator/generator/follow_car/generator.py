import json
import os
import random
import math
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JeEval, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils


random.seed(2020)


class StoppingCarModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--timestep", type=float, default=0.1, help="Time step for each episode.")
        self.optionParser.add_argument("--velocity", type=float, default=28, help="Constant velocity of lead car.")


class StoppingCarModelGenerator(JaniModelGeneratorPddlInJani):
    lead_position = "lead_position"
    lead_velocity = "lead_velocity"
    ego_position = "ego_position"
    ego_velocity = "ego_velocity"
    timestep = "timestep"
    episode = "episode"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.action_labels = ["accelerate", "decelerate"]
        self.timestep_value = options.timestep
        self.lead_velocity_value = options.velocity
  
    def compute_model_initial_and_goal_state(self):
        self.initial_state = dict([(self.lead_position, 45), (self.ego_position, 23), (self.ego_velocity, 30), (self.episode, 0)])
        # goal
        self.goal_state = dict([self.episode, 1000])  # + [(self.unsafety_index, False)])

    def generate_constants(self) -> list:
        return [JaniStructureGenerator.generate_constant_declaration(
            self.TERMINAL_AT_UNSAFE,
            JaniStructureGenerator.generate_bool_type(),
            self.constant_values[self.TERMINAL_AT_UNSAFE]
        ), JaniStructureGenerator.generate_constant_declaration(
            self.timestep,
            JaniStructureGenerator.generate_real_type(),
            self.timestep_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.lead_velocity,
            JaniStructureGenerator.generate_real_type(),
            self.lead_velocity_value
        )]

    def generate_variables(self) -> list:
        return [
                JaniStructureGenerator.generate_bounded_int_variable(self.episode, 0, 4000, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.lead_position, 0, 300000, 50),
                JaniStructureGenerator.generate_bounded_real_variable(self.ego_position, 0, 300000, 30),
                JaniStructureGenerator.generate_bounded_real_variable(self.ego_velocity, 0, 200, 28)]

    def __accelerate(self):
        return [JaniStructureGenerator.generate_assignment(self.ego_velocity, Je.Add(self.ego_velocity, Je.Mult(1, self.timestep)))]

    def __decelerate(self):
        return [JaniStructureGenerator.generate_assignment(self.ego_velocity, Je.Sub(self.ego_velocity, Je.Mult(1, self.timestep)))]

    def __timestep(self):
        return [
            JaniStructureGenerator.generate_assignment(self.lead_position, Je.Add(self.lead_position, Je.Mult(self.lead_velocity, self.timestep))),
            JaniStructureGenerator.generate_assignment(self.ego_position, Je.Add(self.ego_position, Je.Mult(self.ego_velocity, self.timestep))),
            JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
        ]

    def __is_unsafe(self):
        return Je.Le(self.lead_position, self.ego_position)

    def generate_edges(self) -> list:
        def accelerate_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__accelerate() + self.__timestep())],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Le(Je.Add(self.lead_position, Je.Mult(self.lead_velocity, self.timestep)), 300000),
                    Je.Le(Je.Add(self.ego_position, Je.Mult(self.ego_velocity, self.timestep)), 300000),
                    Je.Le(Je.Add(self.ego_velocity, Je.Mult(1, self.timestep)), 200),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def decelerate_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__decelerate() + self.__timestep())],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Le(Je.Add(self.lead_position, Je.Mult(self.lead_velocity, self.timestep)), 300000),
                    Je.Le(Je.Add(self.ego_position, Je.Mult(self.ego_velocity, self.timestep)), 300000),
                    Je.Ge(Je.Sub(self.ego_velocity, Je.Mult(1, self.timestep)), 0),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        return [accelerate_edge(), decelerate_edge()]

    def generate_goal_expression(self) -> json:
        goal_expression = [Je.Ge(self.episode, 1000), Je.Ge(self.ego_position, Je.Mult(1000, Je.Mult(self.lead_velocity, self.timestep)))]
        return JaniStructureGenerator.generate_large_conjunction(goal_expression)

    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], self.generate_goal_expression())
        return JaniStructureGenerator.generate_objective_expression(goal=goal)

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.__is_unsafe())

    def generate_start(self) -> list:
        constraints = [
            Je.Ge(self.lead_position, 40),
            Je.Le(self.lead_position, 50),
            Je.Ge(self.ego_position, 20),
            Je.Le(self.ego_position, 29.9999),
            Je.Ge(self.ego_velocity, 26),
            Je.Le(self.ego_velocity, 32),
            Je.Eq(self.episode, 0)
        ]
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Generating " + str(max_number_states) + " random states ...")
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state={})
        sample = []
        while len(sample) < max_number_states:
            state = {
                self.lead_position: round(random.uniform(40, 50), 4),
                self.ego_position: round(random.uniform(20, 29.9999), 4),
                self.ego_velocity: round(random.uniform(26, 32), 4),
                self.episode: 0
            }
            sample.append(state)
        for state in sample:
            states_values.add(state)
        return states_values.generate_states_values()

    def get_nn_inputs(self) -> list:
        nn_inputs = [self.episode, self.lead_position, self.ego_position, self.ego_velocity]
        return nn_inputs


if __name__ == "__main__":
    args = StoppingCarModelGenerationOptionParser().arg_parse()
    generator = StoppingCarModelGenerator(args)
    generator.generate()
