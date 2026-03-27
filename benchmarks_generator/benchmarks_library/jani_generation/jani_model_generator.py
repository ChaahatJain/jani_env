#
import argparse
import json
import os

from jani_generation.jani_2_nnet_structure_generator import Jani2NNetStructureGenerator
from jani_generation.jani_structure_generator import JaniStructureGenerator
from python_utils import PythonUtils


# auxiliary classes ####################################################################################################

class JaniModelGenerationOptionParser:
    def __init__(self):
        self.optionParser = argparse.ArgumentParser(description="Jani model generator")

    def add_options(self):
        self.optionParser.add_argument("--out", default=None, help="Output jani file")
        self.optionParser.add_argument("--generation", type=int, default=0, help="The number of the generation routine to apply.")
        self.optionParser.add_argument("--model-file", default=None, help="Model file (in JANI).")
        self.optionParser.add_argument("--inline-constants", nargs='+', default=[], help="Inline the values of the listed constants.")
        self.optionParser.add_argument("--sample-in-model", action="store_true", default=False, help="Compound the model with a structure that samples a random start state.")
        self.optionParser.add_argument("--safety-in-model", action="store_true", default=False, help="Add safety property to model.")
        self.optionParser.add_argument("--terminal-at-unsafe", type=int, default=None, help="Unsafe states are terminal.")
        self.optionParser.add_argument("--terminal-at-goal", type=int, default=None, help="Goal states are terminal.")
        self.optionParser.add_argument("--property-type", type=str, default="customized", help="The name of the properties routine to apply.")
        self.optionParser.add_argument("--append-property", action="store_true", default=False, help="Append property in model file (if applicable).")
        self.optionParser.add_argument("--property-inline", action="store_true", default=False, help="Do always inline property substructures.")
        self.optionParser.add_argument("--reuse-random-states", type=str, default=None, help="Try to load random states from the specified folder")
        self.optionParser.add_argument("--use-goal-potential", action="store_true", default=False, help="Use goal potential in (learning) objective.")
        self.optionParser.add_argument("--ground-terminal-potential", action="store_true", default=False, help="Ground goal potential of goal and avoid states in (learning) objective.")
        self.optionParser.add_argument("--ground-start-potential", action="store_true", default=False, help="Ground goal potential of start states in (learning) objective.")
        self.optionParser.add_argument("--networks", type=str, default="", help="The networks sub-directory.")
        self.optionParser.add_argument("--nn-ending", type=str, default=".jani2nnet", help="The file extension of neural network files.")
        self.optionParser.add_argument("--hidden-layers", nargs='+', default=[64, 64], type=int, help="Hidden layer sizes.")
        self.optionParser.add_argument("--applicability-filtering", action="store_true", default=False, help="If in usage, whether to apply with applicability filtering.")
        self.optionParser.add_argument("--splits", nargs='+', default=[], type=int, help="Split sizes (semantics is class dependent).")

    def arg_parse(self):
        self.add_options()
        return self.optionParser.parse_args()


class BoundedVariable:
    def __init__(self, name_: str, lower_bound: int, upper_bound: int, comment: str = ""):
        self.name = name_
        self.lowerBound = lower_bound
        self.upperBound = upper_bound
        self.comment = comment

    def domain_size(self) -> int:
        return self.upperBound - self.lowerBound + 1

    def dump(self) -> str:
        return "(" + self.name + ", " + str(self.lowerBound) + ", " + str(self.upperBound) + ", " + self.comment + ")"

    @staticmethod
    def load_variables_from_file(model_file):
        model_jani = PythonUtils.load_json(model_file)
        return BoundedVariable.load_variables(model_jani)

    @staticmethod
    def load_variables(model_jani: json):
        bounded_variables = list()
        for variable_jani in model_jani["variables"]:  # currently only globals
            type_jani = variable_jani["type"]
            bounded_variables.append(BoundedVariable(variable_jani["name"], type_jani["lower-bound"], type_jani["upper-bound"], variable_jani["comment"] if "comment" in variable_jani else ""))

        return bounded_variables


class VarSplitSpec:
    def __init__(self, var: BoundedVariable, num_splits: int):
        self.var = var
        # spec
        self.LE = True
        self.num_splits = min(self.var.upperBound, num_splits)
        self.required_splits = set()

    def add_required_split(self, split: int):
        self.required_splits.add(split)

    ####################################################################################################################


class JaniModelGenerator(object):

    # Constants:
    TERMINAL_AT_UNSAFE = "terminal-at-unsafe"
    TERMINAL_AT_GOAL = "terminal-at-goal"

    def __init__(self, options):
        self.prop_name_prefix = "pa"
        self.property_structure = dict()
        self.non_property_file_structures = set()

        self.model = None

        # load options (intended virtual)
        self.load_options(options)

    # noinspection PyAttributeOutsideInit
    def load_options(self, options):
        self.out = options.out
        if os.path.basename(self.out) == self.out:
            self.out = "./" + self.out
        #
        self.generation = options.generation
        self.model_file = options.model_file
        self.inline_constants = options.inline_constants
        self.sample_in_model = options.sample_in_model
        self.safety_in_model = options.safety_in_model
        assert not (self.sample_in_model and self.safety_in_model)
        self.property_type = options.property_type
        self.append_property = options.append_property
        self.property_inline = options.property_inline
        self.reuse_random_states = options.reuse_random_states
        self.use_goal_potential = options.use_goal_potential
        self.ground_terminal_potential = options.ground_terminal_potential
        self.ground_start_potential = options.ground_start_potential
        self.networks = options.networks
        self.nn_ending = options.nn_ending
        self.hidden_layers = options.hidden_layers
        self.applicability_filtering = options.applicability_filtering
        self.splits = options.splits

        self.constant_values = dict()

        # Use existing model.
        if self.model_file is None:
            self.constant_values[self.TERMINAL_AT_UNSAFE] = None if options.terminal_at_unsafe is None else False if options.terminal_at_unsafe == 0 else True
            self.constant_values[self.TERMINAL_AT_GOAL] = None if options.terminal_at_goal is None else False if options.terminal_at_goal == 0 else True
        else:
            self.model = PythonUtils.load_json(self.model_file)
            # Constants.
            constants = self.model["constants"]
            _, self.constant_values[self.TERMINAL_AT_UNSAFE] = self.read_constant(constants, self.TERMINAL_AT_UNSAFE)
            _, self.constant_values[self.TERMINAL_AT_GOAL] = self.read_constant(constants, self.TERMINAL_AT_GOAL)

        self.terminal_at_unsafe_jani = self.set_inline_constant(False)
        self.terminal_at_goal_jani = self.set_inline_constant(self.TERMINAL_AT_GOAL)

    @staticmethod
    def read_constant(constants: json, name: str):
        for constant in constants:
            if constant["name"] == name:
                return True, constant["value"] if "value" in constant else None
        return False, None

    def inline_constant(self, constant: str) -> bool:
        return constant in self.inline_constants

    def set_inline_constant(self, constant: str) -> json:
        assert not self.inline_constant(constant) or self.constant_values[constant] is not None
        return self.constant_values[constant] if self.inline_constant(constant) else constant

    # output #####################################################################################################

    # output file aux
    def generate_property_string(self, prop_name_postfix=""):
        prop_name = self.prop_name_prefix
        for key in self.property_structure:
            prop_name += "_"
            prop_name += key
            prop_name += "-"
            prop_name += self.property_structure[key]
        #
        if prop_name_postfix != "":
            prop_name += "_" + prop_name_postfix
        return prop_name

    def generate_property_file_string(self):
        if os.path.basename(self.out) == "":
            output_file = self.out + self.prop_name_prefix
        else:
            output_file = self.out  # local variable due to several output files

        for key in self.property_structure:
            if key in self.non_property_file_structures:
                continue
            output_file += "_"
            output_file += self.property_structure[key]
        #
        output_file += ".jani"
        return output_file

    def extract_relative_path_to(self, path: str):
        return os.path.relpath(path, os.path.dirname(self.generate_property_file_string()))

    def extract_relative_path_to_networks(self):
        return self.extract_relative_path_to(self.networks)

    # auxiliary ########################################################################################################

    def generate_pa_property(self, predicates, start, reach, objective, network_filename):
        if network_filename is None:
            return JaniStructureGenerator.generate_pa_property(self.generate_property_string(), predicates, start, reach, objective, None)
        else:
            network_file_path = PythonUtils.make_file_path(self.extract_relative_path_to_networks(), network_filename)
            return JaniStructureGenerator.generate_pa_property(self.generate_property_string(), predicates, start, reach, objective, network_file_path)

    def generate_jani2nnet_file(self, variable_names: list, action_labels: list) -> None:
        # output file
        if os.path.basename(self.out) == "":
            output_file = PythonUtils.join_path(self.out, "nn_interface")
        else:
            output_file = self.out
        # hidden layers
        for hidden_layer in self.hidden_layers:
            output_file += "_" + str(hidden_layer)
        output_file += ".jani2nnet"
        #
        file = PythonUtils.extract_filename(output_file) + ".nnet"
        input_features = [Jani2NNetStructureGenerator.generate_input_feature_to_global_value_variable(var_name) for var_name in variable_names]
        PythonUtils.write_json(Jani2NNetStructureGenerator.generate_jani2nnet_file(file, self.applicability_filtering, input_features, self.hidden_layers, action_labels), output_file)

    # split domain  ####################################################################################################

    @staticmethod
    def generate_stepwise_splits(splits_max: list) -> list:
        min_splits = min(splits_max)
        step_sizes = [split_max // min_splits for split_max in splits_max]
        current_num_splits_list = [0 for _ in range(len(splits_max))]
        splits = [list(current_num_splits_list)]
        if max(step_sizes) > 1:  # initial split if not same-step-size
            splits.append([1 for _ in range(len(splits_max))])
        #
        for _ in range(min_splits):
            for index in range(0, len(step_sizes)):
                current_num_splits_list[index] += step_sizes[index]
            splits.append(list(current_num_splits_list))

        # if some variables not completely encoded:
        if sum(current_num_splits_list) < sum(splits_max):
            splits.append(list(splits_max))
        #
        return splits

    @staticmethod
    def split_domain_binary_aux(var_lower, var_upper, num_splits, split_seq) -> int:
        domain_size = var_upper - var_lower
        # recursion termination:
        if num_splits <= 0 or domain_size < 2:  # no splits or no domain to split
            return num_splits

        num_splits -= 1
        split_pos = var_lower + (domain_size // 2)
        split_seq.append(split_pos)
        num_splits_rest = JaniModelGenerator.split_domain_binary_aux(var_lower, split_pos, (num_splits + 1) // 2, split_seq)
        return JaniModelGenerator.split_domain_binary_aux(split_pos, var_upper, num_splits // 2 + num_splits_rest, split_seq)

    @staticmethod
    def split_domain_binary(var_lower, var_upper, num_splits) -> list:
        split_seq = list()
        JaniModelGenerator.split_domain_binary_aux(var_lower, var_upper, num_splits, split_seq)
        return split_seq

    @staticmethod
    def binary_domain_lower_bounds(split_spec: VarSplitSpec) -> list:
        assert isinstance(split_spec, VarSplitSpec)
        var = split_spec.var
        lower_bounds = JaniModelGenerator.split_domain_binary(var.lowerBound, var.upperBound + 1, split_spec.num_splits)  # "+1" to also have a predicate for the largest possible value
        for split in split_spec.required_splits:
            if split not in lower_bounds:
                lower_bounds.append(split)
        lower_bounds.sort()
        return lower_bounds

    @staticmethod
    def generate_splitting_predicates(split_specs: list):
        predicates = list()
        for split_spec in split_specs:
            isinstance(split_spec, VarSplitSpec)
            predicates += JaniStructureGenerator.lower_bounds(split_spec.var.name, JaniModelGenerator.binary_domain_lower_bounds(split_spec))
        return predicates

    @staticmethod
    def splits_to_str(splits: list) -> str:
        assert (len(splits) > 0)
        tmp_str = ""
        for split in splits:
            tmp_str += "_" + str(split)
        return tmp_str[1:]  # skip first "_"

    @staticmethod
    def unused_splits_check(splits: list, expected_size: int):
        assert (len(splits) >= expected_size)
        if len(splits) > expected_size:
            print("Warning: unused split values!")

    # start states ####################################################################################################

    @staticmethod
    def generate_state(var_range, val_range, default_vals: dict = None) -> dict:
        rlt_state = dict()
        for var, val in zip(var_range, val_range):
            if default_vals is None or val != default_vals[var]:  # no need to explicitly specify default values
                rlt_state[var] = val
        return rlt_state

    @staticmethod
    def generate_state_tuple(sub_state: dict, default_state: dict) -> tuple:
        state = PythonUtils.update_dict(default_state, sub_state, inplace=False)
        return tuple([state[var] for var in sorted(state.keys())])

    @staticmethod
    def random_start_states_early_termination(states_values: list):
        print("Forced termination: Did not find new random state within reasonable number of iterations!")
        print("Generated " + str(len(states_values)) + " random start states.")
        return JaniStructureGenerator.generate_states_values(states_values)

    class StateValuesGenerator:
        def __init__(self, max_fails: int, default_state: dict):
            self.states_values = list()
            self.states_cache = set()  # to detect duplicates
            self.sequential_fails = 0  # to guarantee termination
            self.max_fails = max_fails
            self.default_state = default_state
            if self.default_state is not None:
                self.states_cache.add(tuple(self.default_state))
            self.loc_values = list()  # to modify start locations, unused so far

        def size(self) -> int:
            return len(self.states_values)

        def add(self, sub_state: dict):
            state_tuple = JaniModelGenerator.generate_state_tuple(sub_state, self.default_state)
            if state_tuple in self.states_cache:
                self.sequential_fails += 1
                if self.sequential_fails >= self.max_fails:
                    return JaniModelGenerator.random_start_states_early_termination(self.states_values)
            else:
                self.sequential_fails = 0
                self.states_cache.add(state_tuple)
                self.states_values.append(JaniStructureGenerator.generate_state_values(self.loc_values, JaniStructureGenerator.generate_variable_values_from_map(sub_state)))
            return None

        def generate_states_values(self) -> json:
            return JaniStructureGenerator.generate_states_values(self.states_values)
