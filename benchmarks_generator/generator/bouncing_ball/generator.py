import json
import random
import math
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JeEval, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani



random.seed(2020)


class BouncingBallModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--timestep", type=float, default=0.1, help="Time step for each episode.")
        self.optionParser.add_argument("--gravity", type=float, default=9.8067, help="Gravitational acceleration.")
        self.optionParser.add_argument("--bounce", type=float, default=0.9, help="How much energy is retained.")
        self.optionParser.add_argument("--push_lower", type=int, default=5, help="Push lower bound.")
        self.optionParser.add_argument("--push_upper", type=int, default=9, help="Push upper bound.")
        self.optionParser.add_argument("--complex", type=int, default=0, help="Accuracy of physical simulation.")


class BouncingBallModelGenerator(JaniModelGeneratorPddlInJani):
    height = "height"
    velocity = "velocity"
    episode = "episode"
    timestep = "timestep"
    gravity = "gravity"
    h_bounce = "height_coefficient"
    v_bounce = "velocity_coefficient"
    lower = "push_lower"
    upper = "push_upper"
    return_velocity = "return_velocity"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.action_labels = ["push", "skip"]
        self.timestep_value = options.timestep
        self.gravity_value = -options.gravity
        self.height_coefficient = options.bounce
        self.velocity_coefficient = -round(math.sqrt(options.bounce), 4)
        self.push_lower = options.push_lower
        self.push_upper = options.push_upper
        self.return_velocity_value = round(math.sqrt(2 * options.gravity * options.push_lower), 4)
        self.complex = bool(options.complex)

    def compute_model_initial_and_goal_state(self):
        self.initial_state = dict([(self.height, 5), (self.velocity, 0), (self.episode, 0)])
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
            self.gravity,
            JaniStructureGenerator.generate_real_type(),
            self.gravity_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.h_bounce,
            JaniStructureGenerator.generate_real_type(),
            self.height_coefficient
        ), JaniStructureGenerator.generate_constant_declaration(
            self.v_bounce,
            JaniStructureGenerator.generate_real_type(),
            self.velocity_coefficient
        ), JaniStructureGenerator.generate_constant_declaration(
            self.return_velocity,
            JaniStructureGenerator.generate_real_type(),
            self.return_velocity_value
        ), JaniStructureGenerator.generate_constant_declaration(
            self.lower,
            JaniStructureGenerator.generate_int_type(),
            self.push_lower
        ), JaniStructureGenerator.generate_constant_declaration(
            self.upper,
            JaniStructureGenerator.generate_int_type(),
            self.push_upper
        )]

    def generate_variables(self) -> list:
        return [JaniStructureGenerator.generate_bounded_int_variable(self.episode, 0, 4000, 0),
                JaniStructureGenerator.generate_bounded_real_variable(self.height, 0, 1000, 5),
                JaniStructureGenerator.generate_bounded_real_variable(self.velocity, -100, 100, 1)]

    def __timestep(self, push=False):
        return [
            JaniStructureGenerator.generate_assignment(self.height, Je.Add(self.height, Je.Add(Je.Mult(self.velocity, self.timestep), Je.Mult(Je.Mult(0.5, self.gravity), Je.Mult(self.timestep, self.timestep))))),
            JaniStructureGenerator.generate_assignment(self.velocity, Je.Add(self.velocity, Je.Sub(Je.Mult(self.timestep, self.gravity), 4 if push else 0))),
            JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
        ]

    def __bounce_timestep(self, push=False):
        if self.complex:
            return [
                JaniStructureGenerator.generate_assignment(self.height, Je.Add(Je.Mult(Je.Mult(Je.Add(self.velocity, Je.Sub(Je.Mult(Je.Div(self.height, Je.Neg(self.velocity)), self.gravity), 4)), self.v_bounce), Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))), Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity))), Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))))))),
                JaniStructureGenerator.generate_assignment(self.velocity, Je.Add(Je.Mult(Je.Add(self.velocity, Je.Sub(Je.Mult(Je.Div(self.height, Je.Neg(self.velocity)), self.gravity), 4)), self.v_bounce), Je.Mult(self.gravity, Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))))),
                JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
            ] if push else [
                JaniStructureGenerator.generate_assignment(self.height, Je.Add(Je.Mult(Je.Mult(Je.Add(self.velocity, Je.Mult(Je.Div(self.height, Je.Neg(self.velocity)), self.gravity)), self.v_bounce), Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))), Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity))), Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))))))),
                JaniStructureGenerator.generate_assignment(self.velocity, Je.Add(Je.Mult(Je.Add(self.velocity, Je.Mult(Je.Div(self.height, Je.Neg(self.velocity)), self.gravity)), self.v_bounce), Je.Mult(self.gravity, Je.Sub(self.timestep, Je.Div(self.height, Je.Neg(self.velocity)))))),
                JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
            ]
        else:
            return [
                JaniStructureGenerator.generate_assignment(self.height, Je.Mult(self.height, self.h_bounce)),
                JaniStructureGenerator.generate_assignment(self.velocity, Je.Mult(Je.Sub(self.velocity, 4 if push else 0), self.v_bounce)),
                JaniStructureGenerator.generate_assignment(self.episode, Je.Add(self.episode, 1))
            ]

    def __is_unsafe(self):
        return Je.And(Je.Le(Je.Add(self.height, Je.Mult(self.velocity, Je.Div(self.lower, self.return_velocity))), Je.Sub(self.lower, 0.0001)), Je.Le(Je.Sub(self.height, Je.Mult(self.velocity, Je.Div(self.lower, self.return_velocity))), Je.Sub(self.lower, 0.0001)), Je.Le(self.episode, 999))

    def generate_edges(self) -> list:
        def push_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__timestep(push=True))],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Ge(self.height, self.push_lower),
                    Je.Le(self.height, self.push_upper),
                    Je.Gt(self.height, Je.Sub(Je.Mult(Je.Neg(self.velocity), self.timestep),
                        Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(self.timestep, self.timestep))))),
                    Je.Ge(Je.Add(Je.Sub(self.velocity, 4), Je.Mult(self.gravity, self.timestep)), -100),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def bounce_push_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__bounce_timestep(push=True))],
                action=self.action_labels[0],
                guard=Je.And(
                    Je.Ge(self.height, self.push_lower),
                    Je.Le(self.height, self.push_upper),
                    Je.Le(self.height, Je.Sub(Je.Mult(Je.Neg(self.velocity), self.timestep),
                        Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(self.timestep, self.timestep))))),
                    Je.Le(Je.Mult(Je.Sub(self.velocity, 4), self.velocity_coefficient), 100),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def skip_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__timestep())],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Gt(self.height, Je.Sub(Je.Mult(Je.Neg(self.velocity), self.timestep),
                        Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(self.timestep, self.timestep))))),
                    Je.Ge(Je.Add(self.velocity, Je.Mult(self.gravity, self.timestep)), -100),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        def bounce_skip_edge():
            return JaniStructureGenerator.generate_edge(
                location="loc_0",
                destinations=[JaniStructureGenerator.generate_destination(
                    location="loc_0", assignments=self.__bounce_timestep())],
                action=self.action_labels[1],
                guard=Je.And(
                    Je.Le(self.height, Je.Sub(Je.Mult(Je.Neg(self.velocity), self.timestep),
                        Je.Mult(0.5, Je.Mult(self.gravity, Je.Mult(self.timestep, self.timestep))))),
                    Je.Le(Je.Mult(self.velocity, self.velocity_coefficient), 100),
                    Je.Le(Je.Add(self.episode, 1), 4000)
                )
            )

        return [push_edge(), bounce_push_edge(), skip_edge(), bounce_skip_edge()]

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
            Je.Ge(self.height, 5),
            Je.Le(self.height, 9),
            Je.Ge(self.velocity, -1),
            Je.Le(self.velocity, 1),
            Je.Eq(self.episode, 0)
        ]
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Generating " + str(max_number_states) + " random states ...")
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state={})
        sample = []
        while len(sample) < max_number_states:
            state = {
                self.height: round(random.uniform(5.0, 9.0), 4),
                self.velocity: round(random.uniform(-1.0, 1.0), 4),
                self.episode: 0
            }
            sample.append(state)
        for state in sample:
            rlt = states_values.add(state)
            if rlt is not None:
                raise Exception("Impossible!!! Encountered a duplicate.")
        return states_values.generate_states_values()

    def get_nn_inputs(self) -> list:
        nn_inputs = [self.episode, self.height, self.velocity]
        return nn_inputs


if __name__ == "__main__":
    args = BouncingBallModelGenerationOptionParser().arg_parse()
    generator = BouncingBallModelGenerator(args)
    generator.generate()
