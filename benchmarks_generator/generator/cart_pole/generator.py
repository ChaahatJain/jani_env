import json
import os
import random
import math
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JeEval, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

random.seed(2020)


class CartPoleModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--timestep", type=float, default=0.001, help="Time step for each episode.")
        self.optionParser.add_argument("--cart_mass", type=float, default=1, help="Mass of the cart.")
        self.optionParser.add_argument("--poll_mass", type=float, default=0.1, help="Mass of the poll.")
        self.optionParser.add_argument("--half_poll_length", type=float, default=0.5, help="Half of the polls length.")
        self.optionParser.add_argument("--gravity", type=float, default=9.8067, help="Gravitational acceleration.")


class CartPoleModelGenerator(JaniModelGeneratorPddlInJani):
    position = "position"
    velocity = "velocity"
    episode = "episode"
    angle = "angle"
    angular_velocity = "angular_velocity"
    timestep = "timestep"
    cart_mass = "cart_mass"
    poll_mass = "poll_mass"
    half_poll_length = "half_poll_length"
    gravity = "gravity"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.action_labels = ["push_left", "push_right"]
        self.timestep_value = options.timestep
        self.cart_mass_value = options.cart_mass
        self.poll_mass_value = options.poll_mass
        self.half_poll_length_value = options.half_poll_length
        self.gravity_value = options.gravity
        self.max_ep_length = 1001
        self.max_position = 10
        self.min_position = -10
        self.max_velocity = 3
        self.min_velocity = -3
        self.max_angle = 0.3
        self.min_angle = -0.3
        self.max_angular_velocity = 5
        self.min_angular_velocity = -5

    def compute_model_initial_and_goal_state(self):
        self.initial_state = dict([(self.position, 0), (self.velocity, 0), (self.angle, 0), (self.angular_velocity, 0), (self.episode, 0)])
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
            self.cart_mass,
            JaniStructureGenerator.generate_real_type(),
            self.cart_mass_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.poll_mass,
            JaniStructureGenerator.generate_real_type(),
            self.poll_mass_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.half_poll_length,
            JaniStructureGenerator.generate_real_type(),
            self.half_poll_length_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.gravity,
            JaniStructureGenerator.generate_real_type(),
            self.gravity_value
        )]

    def generate_variables(self) -> list:
        return [JaniStructureGenerator.generate_bounded_int_variable(self.episode, 0, self.max_ep_length, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.position, self.min_position, self.max_position, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.velocity, self.min_velocity, self.max_velocity, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.angle, self.min_angle, self.max_angle, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.angular_velocity, self.min_angular_velocity, self.max_angular_velocity, 0)]

    def __push(self, left):
        force = -10 if left else 10
        return [
            JaniStructureGenerator.generate_assignment(self.position, Je.Add(self.position, Je.Mult(self.velocity, self.timestep))),
            JaniStructureGenerator.generate_assignment(self.velocity, Je.Add(self.velocity, Je.Mult(Je.Sub(Je.Div(force, Je.Add(self.cart_mass, self.poll_mass)), Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle), Je.Div(force, Je.Add(self.cart_mass, self.poll_mass))), Je.Mult(Je.Add(self.cart_mass, self.poll_mass), Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass, Je.Add(self.cart_mass, self.poll_mass)))))), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.angle, Je.Add(self.angle, Je.Mult(self.angular_velocity, self.timestep))),
            JaniStructureGenerator.generate_assignment(self.angular_velocity, Je.Add(self.angular_velocity, Je.Mult(Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle), Je.Div(force, Je.Add(self.cart_mass, self.poll_mass))), Je.Mult(self.half_poll_length, Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass, Je.Add(self.cart_mass, self.poll_mass))))), self.timestep))),
            JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
        ]

    def __is_unsafe(self):
        return Je.Or(Je.Ge(self.angle, 0.21), Je.Le(self.angle, -0.21))

    def generate_edges(self) -> list:
        def push_left():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__push(True))],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Ge(Je.Add(self.position, Je.Mult(self.velocity, self.timestep)), self.min_position),
                    Je.Le(Je.Add(self.position, Je.Mult(self.velocity, self.timestep)), self.max_position),
                    Je.Ge(Je.Add(self.velocity, Je.Mult(Je.Sub(Je.Div(-10, Je.Add(self.cart_mass, self.poll_mass)),
                        Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle), Je.Div(-10, Je.Add(self.cart_mass, self.poll_mass))),
                            Je.Mult(Je.Add(self.cart_mass, self.poll_mass), Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass,
                                Je.Add(self.cart_mass, self.poll_mass)))))), self.timestep)), self.min_velocity),
                    Je.Le(Je.Add(self.angular_velocity, Je.Mult(Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle),
                        Je.Div(-10, Je.Add(self.cart_mass, self.poll_mass))), Je.Mult(self.half_poll_length,
                            Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass, Je.Add(self.cart_mass, self.poll_mass))))),
                                self.timestep)), self.max_angular_velocity),
                    Je.Le(Je.Add(self.episode, 1), self.max_ep_length)
                )
            )

        def push_right():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__push(False))],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Ge(Je.Add(self.position, Je.Mult(self.velocity, self.timestep)), self.min_position),
                    Je.Le(Je.Add(self.position, Je.Mult(self.velocity, self.timestep)), self.max_position),
                    Je.Le(Je.Add(self.velocity, Je.Mult(Je.Sub(Je.Div(10, Je.Add(self.cart_mass, self.poll_mass)),
                        Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle), Je.Div(10, Je.Add(self.cart_mass, self.poll_mass))),
                            Je.Mult(Je.Add(self.cart_mass, self.poll_mass), Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass,
                                Je.Add(self.cart_mass, self.poll_mass)))))), self.timestep)), self.max_velocity),
                    Je.Ge(Je.Add(self.angular_velocity, Je.Mult(Je.Div(Je.Sub(Je.Mult(self.gravity, self.angle),
                        Je.Div(10, Je.Add(self.cart_mass, self.poll_mass))), Je.Mult(self.half_poll_length,
                            Je.Sub(Je.Div(4, 3), Je.Div(self.poll_mass, Je.Add(self.cart_mass, self.poll_mass))))),
                                self.timestep)), self.min_angular_velocity),
                    Je.Le(Je.Add(self.episode, 1), self.max_ep_length)
                )
            )

        return [push_left(), push_right()]

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
            Je.Ge(self.position, self.min_position/10),
            Je.Le(self.position, self.max_position/10),
            Je.Ge(self.velocity, self.min_velocity/10),
            Je.Le(self.velocity, self.max_velocity/10),
            Je.Ge(self.angle, -0.2),
            Je.Le(self.angle, 0.2),
            Je.Ge(self.angular_velocity, -0.5),
            Je.Le(self.angular_velocity, 0.5),
            Je.Eq(self.episode, 0)
        ]
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Generating " + str(max_number_states) + " random states ...")
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state={})
        sample = []
        while len(sample) < max_number_states:
            state = {
                self.position: round(random.uniform(self.min_position/10, self.max_position/10), 4),
                self.velocity: round(random.uniform(self.min_velocity, self.max_velocity), 4),
                self.angle: round(random.uniform(-0.2, 0.2), 4),
                self.angular_velocity: round(random.uniform(-0.5, 0.5), 4),
                self.episode: 0
            }
            sample.append(state)
        for state in sample:
            rlt = states_values.add(state)
            if rlt is not None:
                raise Exception("Impossible!!! Encountered a duplicate.")
        return states_values.generate_states_values()

    def get_nn_inputs(self) -> list:
        nn_inputs = [self.episode, self.position, self.velocity, self.angle, self.angular_velocity]
        return nn_inputs


if __name__ == "__main__":
    args = CartPoleModelGenerationOptionParser().arg_parse()
    generator = CartPoleModelGenerator(args)
    generator.generate()
