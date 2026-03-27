import json
import os
import random
import math
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JeEval, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils


random.seed(2020)


class InvertedPendulumModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--timestep", type=float, default=0.05, help="Timestep for each episode.")
        self.optionParser.add_argument("--mass", type=float, default=1, help="Mass of the pendulum (with pole).")
        self.optionParser.add_argument("--length", type=float, default=1, help="Length of the pendulums pole.")
        self.optionParser.add_argument("--gravity", type=float, default=9.8067, help="Gravitational acceleration.")


class InvertedPendulumModelGenerator(JaniModelGeneratorPddlInJani):
    episode = "episode"
    angle = "angle"
    angular_velocity = "angular_velocity"
    mass = "mass"
    length = "length"
    timestep = "timestep"
    gravity = "gravity"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.action_labels = ["rotate_left", "rotate_right", "skip"]
        self.timestep_value = options.timestep
        self.mass_value = options.mass
        self.length_value = options.length
        self.gravity_value = options.gravity

    def compute_model_initial_and_goal_state(self):
        self.initial_state = dict([(self.angle, 0), (self.angular_velocity, 0), (self.episode, 0)])
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
            self.mass,
            JaniStructureGenerator.generate_real_type(),
            self.mass_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.length,
            JaniStructureGenerator.generate_real_type(),
            self.length_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.gravity,
            JaniStructureGenerator.generate_real_type(),
            self.gravity_value
        )]

    def generate_variables(self) -> list:
        return [
                JaniStructureGenerator.generate_bounded_int_variable(self.episode, 0, 4000, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.angle, -7.85, 7.85, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.angular_velocity, -10, 10, 0)]

    def __timestep(self):
        return Je.Mult(Je.Div(Je.Mult(-3, self.gravity), Je.Mult(2, self.length)), Je.Neg(self.angle))

    def __skip(self):
        return [
            JaniStructureGenerator.generate_assignment(self.angle, Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity, Je.Mult(self.__timestep(), self.timestep)), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.angular_velocity, Je.Add(self.angular_velocity, Je.Mult(self.__timestep(), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
        ]

    def __rotate(self, left):
        x = -2 if left else 2
        return [
            JaniStructureGenerator.generate_assignment(self.angle, Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity, Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length, self.length))), x)), self.timestep)), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.angular_velocity, Je.Add(self.angular_velocity, Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length, self.length))), x)), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
        ]

    def __is_unsafe(self):
        return Je.Or(Je.Ge(self.angle, 0.785), Je.Le(self.angle, -0.785))

    def generate_edges(self) -> list:
        def rotate_left():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__rotate(True))],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Ge(Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity,
                        Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length,
                            self.length))), -2)), self.timestep)), self.timestep)), -0.785),
                    Je.Le(Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity,
                        Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length,
                            self.length))), -2)), self.timestep)), self.timestep)), 0.785),
                    Je.Ge(Je.Add(self.angular_velocity, Je.Mult(Je.Add(self.__timestep(),
                        Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length, self.length))), -2)),
                            self.timestep)), -10),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def rotate_right():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__rotate(False))],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Ge(Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity,
                        Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length,
                            self.length))), 2)), self.timestep)), self.timestep)), -0.785),
                    Je.Le(Je.Add(self.angle, Je.Mult(Je.Add(self.angular_velocity,
                        Je.Mult(Je.Add(self.__timestep(), Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length,
                            self.length))), 2)), self.timestep)), self.timestep)), 0.785),
                    Je.Le(Je.Add(self.angular_velocity, Je.Mult(Je.Add(self.__timestep(),
                        Je.Mult(Je.Div(3, Je.Mult(self.mass, Je.Mult(self.length, self.length))), 2)),
                            self.timestep)), 10),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def skip():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__skip())],
                action=self.action_labels[2],
                guard=Je.Le(Je.Add(self.episode, 1), 4000)
            )

        return [rotate_left(), rotate_right(), skip()]

    def generate_goal_expression(self) -> json:
        goal_expression = [Je.Ge(self.episode, 1000)]
        return JaniStructureGenerator.generate_large_conjunction(goal_expression)

    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], self.generate_goal_expression())
        return JaniStructureGenerator.generate_objective_expression(goal=goal)

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.__is_unsafe())

    def generate_start(self) -> list:
        constraints = [
            Je.Ge(self.angle, -0.05),
            Je.Le(self.angle, 0.05),
            Je.Ge(self.angular_velocity, -0.1),
            Je.Le(self.angular_velocity, 0.1),
            Je.Eq(self.episode, 0)
        ]
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Generating " + str(max_number_states) + " random states ...")
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state={})
        sample = []
        while len(sample) < max_number_states:
            state = {
                self.angle: round(random.uniform(-0.05, 0.05), 4),
                self.angular_velocity: round(random.uniform(-0.1, 0.1), 4),
                self.episode: 0
            }
            sample.append(state)
        for state in sample:
            rlt = states_values.add(state)
            if rlt is not None:
                raise Exception("Impossible!!! Encountered a duplicate.")
        return states_values.generate_states_values()

    def get_nn_inputs(self) -> list:
        nn_inputs = [self.episode, self.angle, self.angular_velocity]
        return nn_inputs


if __name__ == "__main__":
    args = InvertedPendulumModelGenerationOptionParser().arg_parse()
    generator = InvertedPendulumModelGenerator(args)
    generator.generate()
