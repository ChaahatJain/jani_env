#
import json
import os

from jani_generation.jani_2_nnet_structure_generator import Jani2NNetStructureGenerator
from jani_generation.jani_model_generator import JaniModelGenerator, BoundedVariable, VarSplitSpec
from jani_generation.jani_structure_generator import JaniModelType, JaniStructureGenerator, Je, JeEval
from python_utils import PythonUtils


class JaniModelGeneratorPddlInJani(JaniModelGenerator):
    # model characteristics
    AUTOMATON_NAME = "automaton_0"
    LOCATION_NAME = "loc_0"
    LOCATION_NAMES = [LOCATION_NAME]
    INITIAL_LOCATIONS = [LOCATION_NAME]
    # sample-in-model
    SAMPLE_LOCATION = "sample-state"  # Default locations for sampling. Should add additional sampling locations as required.

    def __init__(self, options):
        JaniModelGenerator.__init__(self, options)
        # dummy init
        self.variable_names = list()
        self.action_labels = list()
        self.initial_state = dict()
        self.goal_state = dict()
        #
        self.model_name = "jani-model"
        self.model_type = JaniModelType.LTS

        if self.sample_in_model:
            JaniModelGeneratorPddlInJani.INITIAL_LOCATIONS = [self.SAMPLE_LOCATION]
            JaniModelGeneratorPddlInJani.LOCATION_NAMES.append(self.SAMPLE_LOCATION)

    # model generation #################################################################################################

    @staticmethod
    def generate_destination_aux(assignments: list, probability: json = None) -> json:
        return JaniStructureGenerator.generate_destination(JaniModelGeneratorPddlInJani.LOCATION_NAME, assignments, probability)

    @staticmethod
    def generate_edge_aux(destinations: list, action: str, guard) -> json:
        return JaniStructureGenerator.generate_edge(JaniModelGeneratorPddlInJani.LOCATION_NAME, destinations, action, guard)

    @staticmethod
    def generate_automaton_aux(name: str, edges: list) -> json:
        return JaniStructureGenerator.generate_automaton(name, JaniModelGeneratorPddlInJani.generate_locations(), JaniModelGeneratorPddlInJani.INITIAL_LOCATIONS, edges)

    @staticmethod
    def generate_locations() -> list:
        return [JaniStructureGenerator.generate_location(name) for name in JaniModelGeneratorPddlInJani.LOCATION_NAMES]

    def generate_edges(self) -> list:
        return list()

    def generate_automaton(self) -> json:
        return self.generate_automaton_aux(self.AUTOMATON_NAME, self.generate_edges())

    def generate_automata(self) -> list:
        return [self.generate_automaton()]

    def generate_syncs(self) -> list:
        return [JaniStructureGenerator.generate_synchronization([action_label], action_label) for action_label in self.action_labels]

    def generate_composition(self) -> json:
        return JaniStructureGenerator.generate_composition([JaniStructureGenerator.generate_composition_element(self.AUTOMATON_NAME)], self.generate_syncs())

    def generate_actions(self) -> list:
        return [JaniStructureGenerator.generate_action(name) for name in self.action_labels]

    def generate_variables(self) -> list:
        return list()

    def generate_goal_expression(self) -> json:
        conjunction = [Je.Eq(var, value) for var, value in self.goal_state.items()]
        return JaniStructureGenerator.generate_large_conjunction(conjunction)

    def generate_goal_property(self) -> json:
        return JaniStructureGenerator.generate_reachability_property(self.model_name + "-goal", self.generate_goal_expression())

    def generate_safety_property(self) -> json:
        return JaniStructureGenerator.generate_pa_property(self.model_name + "-safety", start=self.generate_compact_starts(), reach=self.generate_reach(), objective=self.generate_objective())

    def generate_learning_property(self) -> json:
        assert self.sample_in_model
        return JaniStructureGenerator.generate_pa_property(self.model_name + "-learning", start=None, reach=self.generate_reach(), objective=self.generate_objective())

    def generate_properties(self) -> list:
        if self.sample_in_model:
            return [self.generate_learning_property()]
        elif self.safety_in_model:
            return [self.generate_safety_property()]
        else:
            return [self.generate_goal_property()]

    def generate_constants(self) -> list:
        return list()

    def generate_model(self) -> None:
        model = JaniStructureGenerator.generate_model(self.model_name, self.model_type, self.generate_automata(), self.generate_composition(), self.generate_actions(), self.generate_variables(), self.generate_properties(), self.generate_constants())
        PythonUtils.write_json(model, self.out)

    # additional properties

    def generate_objective(self) -> json:
        return JaniStructureGenerator.generate_objective_expression(goal=JaniStructureGenerator.generate_state_condition_expression([], self.generate_goal_expression()))

    def generate_goal_potential(self) -> json:
        if self.use_goal_potential:
            print("Goal potential is not supported.")
            exit(1)
        return None

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], False)

    def generate_random_states(self, max_number_states: int) -> list:
        print("Warning: No random start states generated (" + str(max_number_states) + ").")
        return list()

    def generate_or_load_random_starts(self, max_number_states: int):
        # cache_file = "./cached_random_states/cached_random_states_" + PythonUtils.extract_filename(self.model_file) + "_" + str(max_number_states) + ".jani"
        if self.reuse_random_states is not None and PythonUtils.exists(self.reuse_random_states, as_file=False):
            cache_file = PythonUtils.join_path(self.reuse_random_states, "cached_random_states_" + PythonUtils.extract_filename(self.model_file) + "_" + str(max_number_states) + ".jani")
            if PythonUtils.exists(cache_file, as_file=True):
                print("Trying to load random states from: " + cache_file)
                return PythonUtils.load_json(cache_file)
        else:
            cache_file = None
        # generate from scratch:
        states_values = self.generate_random_states(max_number_states)
        if cache_file is not None and PythonUtils.exists(self.reuse_random_states, as_file=False):
            PythonUtils.write_json(states_values, cache_file)
        return states_values

    def generate_start(self) -> list:
        print("Warning: No start states generated.")
        return list()

    def generate_compact_starts(self):
        return JaniStructureGenerator.generate_state_condition_expression(self.generate_automaton_location_value(), self.generate_start())

    def generate_stepwise_splits(self) -> list:
        return list()

    @staticmethod
    def add_safety_predicates(predicates, safety_predicates) -> list:
        for safety_predicate in safety_predicates:
            if safety_predicate not in predicates:
                predicates.append(safety_predicate)
        return predicates

    def generate_predicates(self, splits: list) -> list:
        print("Warning: No predicates generated!")
        print("Splits: " + str(splits))
        return list()

    def generate_additional_properties(self):
        self.property_structure["model"] = PythonUtils.extract_filename(self.model_file)
        # self.non_property_file_structures.add("model")
        self.property_structure["prop"] = self.property_type

        # starts
        if "random_starts_" in self.property_type:
            number_starts = int(self.property_type.split("random_starts_", 1)[1])
            start_expression = self.generate_or_load_random_starts(number_starts)
        elif "compact_starts" in self.property_type:
            start_expression = self.generate_compact_starts()
        else:
            assert False
        if self.property_inline:
            start_construct = start_expression
        else:
            path_to_start = PythonUtils.join_path(os.path.dirname(self.generate_property_file_string()), "start.jani")
            PythonUtils.write_json(start_expression, path_to_start)
            start_construct = JaniStructureGenerator.generate_external(self.extract_relative_path_to(path_to_start))

        # reach
        if self.property_inline:
            reach_construct = self.generate_reach()
        else:
            path_to_reach = PythonUtils.join_path(os.path.dirname(self.generate_property_file_string()), "reach.jani")
            PythonUtils.write_json(self.generate_reach(), path_to_reach)
            reach_construct = JaniStructureGenerator.generate_external(self.extract_relative_path_to(path_to_reach))

        # objective
        if self.property_inline:
            objective_construct = self.generate_objective()
        else:
            path_to_objective = PythonUtils.join_path(os.path.dirname(self.generate_property_file_string()), "objective.jani")
            PythonUtils.write_json(self.generate_objective(), path_to_objective)
            objective_construct = JaniStructureGenerator.generate_external(self.extract_relative_path_to(path_to_objective))

        # predicates
        if len(self.splits) == 0:
            splits_sequences = self.generate_stepwise_splits()
            self.non_property_file_structures.add("domain_splits")  # disable for property file
        else:
            splits_sequences = [self.splits] if sum(self.splits) > 0 else []  # no predicates
        splits_to_external = dict()
        for splits in splits_sequences:
            splits_str = self.splits_to_str(splits)
            path_to_preds = PythonUtils.join_path(os.path.dirname(self.generate_property_file_string()), "predicates", splits_str + ".jani")
            PythonUtils.write_json(self.generate_predicates(splits), path_to_preds)
            splits_to_external[splits_str] = JaniStructureGenerator.generate_external(self.extract_relative_path_to(path_to_preds))

        # nn and finalize ...
        if not os.path.exists(self.networks):
            properties = [self.generate_pa_property(predicates=None, start=start_construct, reach=reach_construct, objective=objective_construct, network_filename=None)]
            if self.append_property:
                model = PythonUtils.load_json(self.model_file)
                assert "properties" in model
                model["properties"] += properties
                PythonUtils.write_json(model, self.model_file)
            else:
                PythonUtils.write_json(JaniStructureGenerator.generate_properties(properties), self.generate_property_file_string())
            return
        # else:
        assert not self.append_property
        for f_nn in sorted(os.listdir(self.networks)):
            if f_nn.endswith(self.nn_ending):
                self.property_structure["nn"] = PythonUtils.extract_filename(f_nn)
                properties = []
                for splits in splits_sequences:
                    splits_str = self.splits_to_str(splits)
                    self.property_structure["domain_splits"] = splits_str
                    prop = self.generate_pa_property(splits_to_external[splits_str], start_construct, reach_construct, objective_construct, f_nn)
                    properties.append(prop)
                # no predicates case:
                if len(splits_sequences) == 0:
                    self.property_structure["domain_splits"] = "0"
                    prop = self.generate_pa_property(predicates=None, start=start_construct, reach=reach_construct, objective=objective_construct, network_filename=f_nn)
                    properties.append(prop)
                #
                PythonUtils.write_json(JaniStructureGenerator.generate_properties(properties), self.generate_property_file_string())

    # jani2nnet

    def restrict_nn_inputs(self, size: int, nn_inputs: list, fixed: set) -> list:
        nn_input_restricted = list(fixed)
        for nn_input in nn_inputs:
            if len(nn_input_restricted) >= size:
                break
            if nn_input not in fixed:
                nn_input_restricted.append(nn_input)
        return sorted(nn_input_restricted)

    def get_nn_inputs(self) -> list:
        return self.variable_names

    def get_nn_action_labels(self) -> list:
        return self.action_labels

    # jani2nnet legacy:

    def generate_automaton_location_value(self, location=None) -> json:
        return JaniStructureGenerator.generate_location_values([self.AUTOMATON_NAME], [self.LOCATION_NAME] if location is None else [location])

    def generate_automaton_instance(self) -> json:
        return Jani2NNetStructureGenerator.generate_automaton_instance(self.AUTOMATON_NAME, 0)

    # generation routines ##############################################################################################

    def generate(self):
        if self.generation == 0:
            self.generate_model()
        elif self.generation == 1:
            self.generate_additional_properties()
        elif self.generation == 2:
            self.generate_jani2nnet_file(self.get_nn_inputs(), self.get_nn_action_labels())
        else:
            assert False


# cost generator #######################################################################################################
# This is a compositional substructure, not a derived class of model generator !!!

class ModelCostGenerator(object):
    # constants:
    FAILING_PROB_NAME = "failing_probability"
    ITEM_COST_BOUND_NAME = "item_cost_bound"  # Blocksworld 4:40/50, 6:12/12, 8:8/10, 10:5/10 (det/non-det; 8-puzzle: 8
    CONTINUOUS_COST = "continuous_cost"
    NON_DET_COST_NAME = "non_deterministic_cost"
    ACCUMULATE_COST_BOUND_NAME = "accumulate_cost_bound"  # 1000
    COST_TERMINAL_NAME = "cost_terminal"  # states are terminal if cost bound is reached
    # characteristics
    COST_PREDS_AFTERWARDS = True
    MAX_COST_SPLITS = 15

    @staticmethod
    def set_cost_preds_afterwards(value: bool):
        ModelCostGenerator.COST_PREDS_AFTERWARDS = value

    @staticmethod
    def set_max_cost_splits(value: int):
        ModelCostGenerator.MAX_COST_SPLITS = value

    #

    def __init__(self, parent: JaniModelGenerator):
        self.parent = parent
        self.cost_vars = dict()  # cost vars per item
        self.step_cost = None
        self.accumulated_cost = None

    @staticmethod
    def add_options(option_parser):
        option_parser.add_argument("--failing-prob", type=float, default=None, help="Probability for a move to fail.")
        option_parser.add_argument("--cost-per-item", type=int, default=None, help="Set move costs per item (with provided upper bound; -1 for disabling).")
        option_parser.add_argument("--continuous-cost", type=PythonUtils.str2bool, default=None, help="Cost are continuous.")
        option_parser.add_argument("--non-det-cost", type=PythonUtils.str2bool, default=None, help="Cost are increased non-deterministically when a move fails.")
        option_parser.add_argument("--accumulate-cost", type=int, default=None, help="Accumulate move costs (up to provided upper bound; -1 for disabling).")
        option_parser.add_argument("--cost-terminal", type=PythonUtils.str2bool, default=None, help="By-pass stalling when cost-per-item upper bound is reached.")
        # property/nn options (thus not saved in model file):
        option_parser.add_argument("--zero-cost-start", action="store_true", default=False, help="Cost-per-item set to 0 at start.")
        option_parser.add_argument("--cost-ignoring-nn", action="store_true", default=False, help="Do not add cost variables to network interface.")

    # noinspection PyAttributeOutsideInit
    def load_options(self, options):
        self.failing_prob = options.failing_prob

        self.cost_per_item = options.cost_per_item
        self.continuous_cost = options.continuous_cost
        self.non_det_cost = options.non_det_cost
        self.accumulate_cost = options.accumulate_cost
        self.cost_terminal = options.cost_terminal

        # TODO check must happen somewhere else, since load_options might be called before read_constants
        # assert not self.do_cost() or all(flag is not None for flag in [self.continuous_cost, self.non_det_cost, self.accumulate_cost, self.cost_terminal])

        self.set_parameterized_constants()

        self.zero_cost_start = options.zero_cost_start
        self.cost_ignoring_nn = options.cost_ignoring_nn
        assert self.do_cost() or not self.do_accumulate_cost()  # accumulatation is a cost extension

    # noinspection PyAttributeOutsideInit
    def read_constants(self, constants: json):
        _, self.failing_prob = JaniModelGenerator.read_constant(constants, self.FAILING_PROB_NAME)

        exists, self.cost_per_item = JaniModelGenerator.read_constant(constants, self.ITEM_COST_BOUND_NAME)
        self.cost_per_item = self.cost_per_item if exists else -1

        exists, self.continuous_cost = JaniModelGenerator.read_constant(constants, self.CONTINUOUS_COST)
        # assert not self.do_cost() or exists

        exists, self.non_det_cost = JaniModelGenerator.read_constant(constants, self.NON_DET_COST_NAME)
        # assert not self.do_cost() or exists
        assert not self.non_det_cost or self.do_cost()

        exists, self.accumulate_cost = JaniModelGenerator.read_constant(constants, self.ACCUMULATE_COST_BOUND_NAME)
        self.accumulate_cost = self.accumulate_cost if exists else -1

        exists, self.cost_terminal = JaniModelGenerator.read_constant(constants, self.COST_TERMINAL_NAME)
        # assert not self.do_cost() or exists

        self.set_parameterized_constants()

    # noinspection PyAttributeOutsideInit
    def set_parameterized_constants(self):
        if self.failing_prob is not None and self.parent.inline_constant(self.FAILING_PROB_NAME):
            self.failing_prob_inline = self.failing_prob
            self.non_failing_prob_inline = 1 - self.failing_prob
        else:
            self.failing_prob_inline = self.FAILING_PROB_NAME
            self.non_failing_prob_inline = Je.Sub(1, self.FAILING_PROB_NAME)

        self.cost_per_item_inline = self.cost_per_item if PythonUtils.is_numeric(self.cost_per_item) and self.parent.inline_constant(self.ITEM_COST_BOUND_NAME) else self.ITEM_COST_BOUND_NAME
        self.accumulate_cost_inline = self.accumulate_cost if PythonUtils.is_numeric(self.accumulate_cost) and self.parent.inline_constant(self.ACCUMULATE_COST_BOUND_NAME) else self.ACCUMULATE_COST_BOUND_NAME

    def get_model_type(self):
        return JaniModelType.LTS if PythonUtils.is_numeric(self.failing_prob) and self.failing_prob == 0 and self.parent.inline_constant(self.FAILING_PROB_NAME) else JaniModelType.MDP

    def do_cost(self) -> bool:
        return self.cost_per_item != -1

    def do_accumulate_cost(self) -> bool:
        return self.accumulate_cost != -1

    def do_non_det_cost(self) -> bool:
        return self.non_det_cost is True

    def fix_cost_terminal(self) -> bool:
        assert self.do_cost()
        return self.parent.inline_constant(self.COST_TERMINAL_NAME) and isinstance(self.cost_terminal, bool)

    def set_cost_vars(self, items: list, item_str: str):
        self.cost_vars = dict([(item, "cost_" + item_str + "_" + str(item)) for item in items])
        self.step_cost = "step_cost"
        self.accumulated_cost = "accumulated_cost" if self.do_accumulate_cost() else None

    # noinspection PyAttributeOutsideInit
    def set_variable_names(self, variable_names: list):
        self.variable_names = variable_names
        self.variable_names += list(self.cost_vars.values()) if self.do_cost() else []
        self.variable_names += [self.step_cost] if self.step_cost else []
        self.variable_names += [self.accumulated_cost] if self.do_accumulate_cost() else []

    # noinspection PyAttributeOutsideInit
    def set_initial_cost(self, initial_state: dict):
        self.initial_state = initial_state

        for cost_var in self.cost_vars.values():
            initial_state[cost_var] = 0

        if self.step_cost:
            initial_state[self.step_cost] = 0

        if self.do_accumulate_cost():
            initial_state[self.accumulated_cost] = 0

    # model:

    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.FAILING_PROB_NAME, JaniStructureGenerator.generate_real_type(), JaniStructureGenerator.none_if_non_numeric(self.failing_prob))] + \
            ([JaniStructureGenerator.generate_constant_declaration(self.ITEM_COST_BOUND_NAME, JaniStructureGenerator.generate_int_type(), self.cost_per_item)] if self.do_cost() else []) + \
            ([JaniStructureGenerator.generate_constant_declaration(self.CONTINUOUS_COST, JaniStructureGenerator.generate_bool_type(), self.continuous_cost)] if self.do_cost() else []) + \
            ([JaniStructureGenerator.generate_constant_declaration(self.NON_DET_COST_NAME, JaniStructureGenerator.generate_bool_type(), self.non_det_cost)] if self.do_cost() else []) + \
            ([JaniStructureGenerator.generate_constant_declaration(self.ACCUMULATE_COST_BOUND_NAME, JaniStructureGenerator.generate_int_type(), self.accumulate_cost)] if self.do_accumulate_cost() else []) + \
            ([JaniStructureGenerator.generate_constant_declaration(self.COST_TERMINAL_NAME, JaniStructureGenerator.generate_bool_type(), self.cost_terminal)] if self.do_cost() else [])

    def generate_variables(self):
        fun_gen = JaniStructureGenerator.generate_bounded_real_variable if self.continuous_cost else JaniStructureGenerator.generate_bounded_int_variable
        return [fun_gen(cost_var, 0, self.cost_per_item_inline, self.initial_state[cost_var]) for cost_var in self.cost_vars.values()] + \
            ([fun_gen(self.step_cost, 0, self.cost_per_item_inline, self.initial_state[self.step_cost])] if self.do_cost() else []) + \
            ([fun_gen(self.accumulated_cost, 0, self.accumulate_cost_inline, self.initial_state[self.accumulated_cost])] if self.do_accumulate_cost() else [])

    # edges:

    def respects_cost_bounds(self, item, inc: json) -> json:
        assert self.do_cost()
        assert not self.do_non_det_cost()
        guard = Je.Le(self.cost_vars[item], JeEval.Sub(self.cost_per_item_inline, inc))
        if self.do_accumulate_cost():
            guard = Je.And(guard, self.respect_accumulated_cost_bounds(item, 1))
        return guard

    def disrespects_cost_bounds(self, item, inc: json) -> json:
        assert self.do_cost()
        assert not self.do_non_det_cost()
        guard = Je.Ge(self.cost_vars[item], JeEval.Sub(self.cost_per_item_inline, JeEval.Sub(inc, 1)))
        if self.do_accumulate_cost():
            guard = Je.And(guard, self.respect_accumulated_cost_bounds(item, 1))  # still have to respect accumulated cost if present
        return guard

    def respect_accumulated_cost_bounds(self, item, inc: int):
        assert self.do_accumulate_cost()
        accumulation_sum = Je.Add([self.accumulated_cost, self.cost_vars[item]] + ([inc] if inc > 0 else []))
        return Je.Le(accumulation_sum, self.accumulate_cost_inline)

    def inc_cost(self, item, inc: int) -> list:
        assert self.do_cost()
        move_cost_var = self.cost_vars[item]
        step_cost_var = self.step_cost
        cost_ass = [JaniStructureGenerator.generate_assignment(step_cost_var, move_cost_var)]
        if inc > 0:
            cost_ass.append(JaniStructureGenerator.generate_assignment(move_cost_var, Je.Add(move_cost_var, inc)))
        if self.do_accumulate_cost():
            cost_ass.append(self.inc_accumulated_cost(item, inc))
        return cost_ass

    def inc_accumulated_cost(self, item, inc: int) -> list:
        assert self.do_accumulate_cost()
        move_cost_var = self.cost_vars[item]
        accumulated_cost_var = self.accumulated_cost
        accumulation_sum = Je.Add([accumulated_cost_var, move_cost_var] + ([inc] if inc > 0 else []))
        return JaniStructureGenerator.generate_assignment(accumulated_cost_var, accumulation_sum)

    def append_failing_destination(self, item, action: str, guard: json, assignments: list) -> list:
        if self.do_cost():
            destinations = [JaniModelGeneratorPddlInJani.generate_destination_aux(assignments + self.inc_cost(item, 0), self.non_failing_prob_inline)]

            if self.do_non_det_cost():
                fail_assignments = [JaniStructureGenerator.generate_non_det_assignment(self.cost_vars[item], lower_bound=self.cost_vars[item], upper_bound=self.cost_per_item_inline)]
                fail_assignments += self.inc_cost(item, 0)
                destinations += [JaniModelGeneratorPddlInJani.generate_destination_aux(fail_assignments, self.failing_prob_inline)]

                if self.do_accumulate_cost():
                    guard = Je.And(guard, self.respect_accumulated_cost_bounds(item, 1))  # still have to respect accumulated cost if present

                return [JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action=action, guard=guard)]

            else:
                destinations += [JaniModelGeneratorPddlInJani.generate_destination_aux(self.inc_cost(item, 1), self.failing_prob_inline)]

                edges = [JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, Je.And(guard, self.respects_cost_bounds(item, 1)))]

                if (self.fix_cost_terminal() and not self.cost_terminal) or not self.fix_cost_terminal():
                    cost_terminal_guard = Je.And(guard, self.disrespects_cost_bounds(item, 1))
                    cost_terminal_guard = cost_terminal_guard if self.fix_cost_terminal() else Je.And(Je.Not(self.COST_TERMINAL_NAME), cost_terminal_guard)
                    edges += [JaniModelGeneratorPddlInJani.generate_edge_aux([JaniModelGeneratorPddlInJani.generate_destination_aux(assignments + self.inc_cost(item, 0))], action, guard=cost_terminal_guard)]  # still accumulate cost if present

                return edges

        else:
            destinations = [JaniModelGeneratorPddlInJani.generate_destination_aux(assignments, self.non_failing_prob_inline)]
            destinations += [JaniModelGeneratorPddlInJani.generate_destination_aux([], self.failing_prob_inline)]
            return [JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, guard)]

    # properties:

    def generate_objective(self, cost_scale: float) -> tuple:
        if self.do_cost():
            return Je.Mult(cost_scale, self.step_cost), ["steps"]
        else:
            print("Warning: Cost not considered in objective.")
            return None, None

    def generate_0_cost_start_constraints(self) -> list:
        constraints = list()
        # costs are initially zero:
        if self.zero_cost_start and self.do_cost():
            constraints += [Je.Eq(cost_var, 0) for cost_var in list(self.cost_vars.values()) + [self.step_cost]]
        if self.zero_cost_start and self.do_accumulate_cost():
            constraints.append(Je.Eq(self.accumulated_cost, 0))
        return constraints

    # predicates:
    def generate_stepwise_splits(self, splits_list) -> list:
        if self.do_cost():
            max_cost_splits = min(self.cost_per_item, self.MAX_COST_SPLITS)
            if self.COST_PREDS_AFTERWARDS:
                return [splits + [0] for splits in splits_list] + [splits_list[-1] + [current_splits] for current_splits in range(1, max_cost_splits + 1)]
            else:
                cost_splits_list = [[splits] for splits in range(0, max_cost_splits + 1)]
                if len(cost_splits_list) >= len(splits_list):
                    return [splits + cost_splits for splits, cost_splits in zip(splits_list, cost_splits_list)] + [splits_list[-1] + cost_splits for cost_splits in cost_splits_list[len(splits_list):]]
                else:
                    return [splits + cost_splits for splits, cost_splits in zip(splits_list, cost_splits_list)] + [splits + cost_splits_list[-1] for splits in splits_list[len(cost_splits_list):]]
        return splits_list

    def generate_splits_mapping(self, splits_specs: list, splits: list):
        for cost_var in self.cost_vars.values():
            splits_specs.append(VarSplitSpec(BoundedVariable(cost_var, 0, self.cost_per_item), splits[-1]))
        return splits_specs

    # interface:
    def adapt_nn_inputs(self, variables: list) -> list:
        if self.do_cost():
            if self.cost_ignoring_nn:
                for cost_var in self.cost_vars.values():
                    variables.remove(cost_var)
            #
            variables.remove(self.step_cost)
        if self.do_accumulate_cost():
            variables.remove(self.accumulated_cost)
        return variables
