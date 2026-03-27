#
import fractions
import json

from python_utils import PythonUtils


# Je class adapted from racetrack generator (https://gitlab.perspicuous-computing.science/project-c6/models/racetrack.git [November 2020]),
# functional changes documented with keyword "PlaJA"
# noinspection PyPep8Naming


class Je(object):

    @staticmethod
    def _binary(var, val, op="="):
        return {"op": op, "left": var, "right": val}

    # Arithmetic:

    @staticmethod
    def Neg(exp):
        return Je.Sub(0, exp)

    @staticmethod
    def Abs(exp):
        return {"op": "abs", "exp": exp}

    @staticmethod
    def Min(x, y):
        return Je._binary(x, y, "min")

    @staticmethod
    def Max(x, y):
        return Je._binary(x, y, "max")

    @staticmethod
    def Eq(var, val):
        return Je._binary(var, val, "=")

    @staticmethod
    def Ne(var, val):
        return Je._binary(var, val, "≠")

    @staticmethod
    def Lt(var, val):
        return Je._binary(var, val, "<")

    @staticmethod
    def Le(var, val):
        return Je._binary(var, val, "≤")

    @staticmethod
    def Gt(var, val):
        return Je.Lt(val, var)

    @staticmethod
    def Ge(var, val):
        return Je.Le(val, var)

    @staticmethod
    def Add(var, val):
        return Je._binary(var, val, "+")

    @staticmethod
    def Sub(var, val):
        return Je._binary(var, val, "-")

    @staticmethod
    def Mult(var, val):
        return Je._binary(var, val, "*")

    @staticmethod
    def Mod(var, val):  # PlaJA added
        return Je._binary(var, val, "%")

    @staticmethod
    def Div(var, val=None):
        if val is None:
            assert (isinstance(var, fractions.Fraction))
            return Je._binary(var.numerator, var.denominator)
        return Je._binary(var, val, "/")

    @staticmethod
    def Pow(x, y):
        if y == 2:  # PlaJA special case to avoid semantics issues as well as problems with z3
            return Je.Mult(x, x)
        return Je._binary(x, y, "pow")

    @staticmethod
    def concat(exps, i=0, op="∧"):
        if i + 1 == len(exps):
            return exps[i]
        return {"op": op, "left": exps[i], "right": Je.concat(exps, i + 1, op)}

    @staticmethod
    def concat_root(args, op):
        if len(args) == 1 and isinstance(args[0], list):
            return Je.concat(args[0], 0, op)
        return Je.concat(args, 0, op)

    # Logical:

    @staticmethod
    def And(*args):
        return Je.concat_root(args, "∧")

    @staticmethod
    def Or(*args):
        return Je.concat_root(args, "∨")

    @staticmethod
    def Add(*args):
        return Je.concat_root(args, "+")

    @staticmethod
    def Not(exp):
        return {"op": "¬", "exp": exp}

    @staticmethod
    def Implies(cond, conseq):
        return Je.Or([Je.Not(cond), conseq]) # PLAJA use JANI implication instead
        # return {"op": "⇒", "left": cond, "right": conseq}

    @staticmethod
    def Iff(arg1, arg2):  # PlaJA added
        return Je.And(Je.Implies(arg1, arg2), Je.Implies(arg2, arg1))

    # Special:

    @staticmethod
    def Round(exp):
        return {"op": "floor", "exp": Je.Add(exp, 0.5)}

    # @staticmethod
    # def Round(exp):
    #     return Je.Ite(Je.Lt(exp, 0), Je.Sub(0, Je._Round(Je.Sub(0, exp))), Je._Round(exp))

    @staticmethod
    def MultRounded(var, val):
        return Je.Round(Je.Mult(var, val))

    @staticmethod
    def SubRounded(var, val):
        return Je.Round(Je.Sub(var, val))

    @staticmethod
    def AddRounded(var, val):
        return Je.Round(Je.Add(var, val))

    # PLAJA to correctly model integer division:
    @staticmethod
    def IntDiv(var, val):
        return {"op": "floor", "exp": Je.Div(var, val)}

    @staticmethod
    def ArrayConstruct(var, length, expr):
        return {"op": "ac", "var": var, "length": length, "exp": expr}

    @staticmethod
    def ArrayAccess(exp, ind):
        return {"op": "aa", "exp": exp, "index": ind}

    @staticmethod
    def ArrayValue(elem):
        return {"op": "av", "elements": elem}

    @staticmethod
    def Ite(i, t, e):
        return {"op": "ite", "if": i, "then": t, "else": e}

    @staticmethod
    def Value(var, val):
        return {"ref": var, "value": val}

    # PlaJA extension:

    @staticmethod
    def Add(*args):
        if len(args) == 1 and isinstance(args[0], list):
            return Je.concat(args[0], 0, "+")
        return Je.concat(args, 0, "+")

    @staticmethod
    def AddNeg(var, val):
        return Je.Add(var, Je.Mult(-1, val))

    @staticmethod
    def min(*args):
        if len(args) == 1 and isinstance(args[0], list):
            return Je.concat(args[0], 0, "min")
        return Je.concat(args, 0, "min")

    @staticmethod
    def max(*args):
        if len(args) == 1 and isinstance(args[0], list):
            return Je.concat(args[0], 0, "max")
        return Je.concat(args, 0, "max")

    @staticmethod
    def to_int(condition):
        return Je.Ite(condition, 1, 0)

    @staticmethod
    def IffFlag(flag, consequence, alternative=None):
        return Je.Or(Je.And(flag, consequence), Je.And(Je.Not(flag), alternative if alternative else Je.Not(consequence)))

    @staticmethod
    def IffIntFlag(flag, consequence, alternative=None):
        return Je.Or(Je.And(Je.Ge(flag, 1), consequence), Je.And(Je.Le(flag, 0), alternative if alternative else Je.Not(consequence)))


# Try to evaluate expression.
class JeEval(Je):

    @staticmethod
    def Add(left, right):
        if PythonUtils.is_numeric(left) and PythonUtils.is_numeric(right):
            return left + right
        else:
            return Je.Add(left, right)

    @staticmethod
    def Sub(left, right):
        if PythonUtils.is_numeric(left) and PythonUtils.is_numeric(right):
            return left - right
        else:
            return Je.Sub(left, right)

    @staticmethod
    def Mult(left, right):
        if PythonUtils.is_numeric(left) and PythonUtils.is_numeric(right):
            return left * right
        else:
            return Je.Mult(left, right)

    @staticmethod
    def Eq(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left == right
        else:
            return Je.Eq(left, right)

    @staticmethod
    def Ne(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left != right
        else:
            return Je.Ne(left, right)

    @staticmethod
    def Lt(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left < right
        else:
            return Je.Lt(left, right)

    @staticmethod
    def Le(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left <= right
        else:
            return Je.Le(left, right)

    @staticmethod
    def Gt(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left > right
        else:
            return Je.Gt(left, right)

    @staticmethod
    def Ge(left, right):
        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return left >= right
        else:
            return Je.Ge(left, right)

    @staticmethod
    def And(left, right):

        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return bool(left) and bool(right)

        if PythonUtils.is_constant_value(left):
            return right if bool(left) else False

        if PythonUtils.is_constant_value(right):
            return left if bool(left) else False

        return Je.And(left, right)

    @staticmethod
    def Or(left, right):

        if PythonUtils.is_constant_value(left) and PythonUtils.is_constant_value(right):
            return bool(left) or bool(right)

        if PythonUtils.is_constant_value(left):
            return True if bool(left) else right

        if PythonUtils.is_constant_value(right):
            return True if bool(left) else left

        return Je.Or(left, right)

    @staticmethod
    def Not(exp):
        if PythonUtils.is_constant_value(exp):
            return not bool(exp)
        else:
            return Je.Not(exp=exp)


########################################################################################################################

class JaniModelType(object):
    LTS = "lts"
    MDP = "mdp"

########################################################################################################################


class JaniStructureGenerator(object):

    # auxiliary ########################################################################################################

    @staticmethod
    def generate_large_conjunction(constraint_list: list):
        assert (len(constraint_list) > 0)
        if len(constraint_list) <= 2:
            return Je.And(constraint_list)
        else:
            size = len(constraint_list)
            return Je.And(JaniStructureGenerator.generate_large_conjunction(constraint_list[0:(size // 2)]), JaniStructureGenerator.generate_large_conjunction(constraint_list[(size // 2):]))

    @staticmethod
    def generate_large_disjunction(constraint_list: list):
        assert (len(constraint_list) > 0)
        if len(constraint_list) <= 2:
            return Je.Or(constraint_list)
        else:
            size = len(constraint_list)
            return Je.Or(JaniStructureGenerator.generate_large_disjunction(constraint_list[0:(size // 2)]), JaniStructureGenerator.generate_large_disjunction(constraint_list[(size // 2):]))

    @staticmethod
    def generate_array_index(x, y, x_dim):
        if isinstance(x, int) and isinstance(y, int):  # non-dynamic, i.e., x and y are integer constants
            return x + y * x_dim
        else:
            return Je.Add(x, Je.Mult(y, x_dim))

    @staticmethod
    def none_if_non_numeric(value: json):
        return value if PythonUtils.is_numeric(value) else None

    # synchronization ##################################################################################################

    @staticmethod
    def generate_action(name: str):
        return {"name": name}

    @staticmethod
    def generate_synchronization(synchronise: list, result=None):
        synchronization = {"synchronise": synchronise}
        if result is not None:
            synchronization["result"] = result
        return synchronization

    @staticmethod
    def generate_composition_element(automaton: str):
        return {"automaton": automaton}

    @staticmethod
    def generate_composition_elements(automata_names: list[str]):
        return [JaniStructureGenerator.generate_composition_element(automaton) for automaton in automata_names]

    @staticmethod
    def generate_composition(elements, syncs=None):
        composition = {"elements": elements}
        if syncs is not None:
            composition["syncs"] = syncs
        return composition

    # types ############################################################################################################

    @staticmethod
    def generate_bool_type():
        return "bool"

    @staticmethod
    def generate_int_type():
        return "int"

    @staticmethod
    def generate_real_type():
        return "real"

    @staticmethod
    def generate_bounded_type(base: str, lower_bound: json, upper_bound: json):
        if isinstance(lower_bound, int) and isinstance(upper_bound, int):
            assert lower_bound < upper_bound  # non-empty
        return {"kind": "bounded", "base": base, "lower-bound": lower_bound, "upper-bound": upper_bound}

    @staticmethod
    def generate_array_type(base: json) -> json:
        return {"kind": "array", "base": base}

    @staticmethod
    def generate_bounded_int_type(lower_bound: json, upper_bound: json):
        return JaniStructureGenerator.generate_bounded_type(base="int", lower_bound=lower_bound, upper_bound=upper_bound)

    @staticmethod
    def generate_bounded_real_type(lower_bound: json, upper_bound: json):
        return JaniStructureGenerator.generate_bounded_type(base="real", lower_bound=lower_bound, upper_bound=upper_bound)

    # variables ########################################################################################################

    @staticmethod
    def generate_variable_declaration(name: str, decl_type, initial_value: json = None):
        variable_declaration = {"name": name, "type": decl_type}
        if initial_value is not None:
            variable_declaration["initial-value"] = initial_value
        return variable_declaration

    @staticmethod
    def generate_constant_declaration(name: str, decl_type, value=None):
        constant_declaration = {"name": name, "type": decl_type}
        if value is not None:
            constant_declaration["value"] = value
        return constant_declaration

    @staticmethod
    def generate_bool_variable(name: str, initial_value: bool):
        return JaniStructureGenerator.generate_variable_declaration(name, JaniStructureGenerator.generate_bool_type(), initial_value)

    @staticmethod
    def generate_bounded_int_variable(name: str, lower_bound: json, upper_bound: json, initial_value: json):
        return JaniStructureGenerator.generate_variable_declaration(name, JaniStructureGenerator.generate_bounded_int_type(lower_bound, upper_bound), initial_value)

    @staticmethod
    def generate_bounded_real_variable(name: str, lower_bound: json, upper_bound: json, initial_value: json):
        return JaniStructureGenerator.generate_variable_declaration(name, JaniStructureGenerator.generate_bounded_real_type(lower_bound, upper_bound), initial_value)

    # automata #########################################################################################################

    @staticmethod
    def generate_location(name: str):
        return {"name": name}

    @staticmethod
    def generate_locations(names: list[str]):
        return [JaniStructureGenerator.generate_location(name) for name in names]

    @staticmethod
    def generate_assignment(ref: json, value: json):
        return {"ref": ref, "value": value}

    @staticmethod
    def generate_self_assignment(ref: json, value: json):
        return {"ref": ref, "value": Je.Add(ref, value)}

    @staticmethod
    def generate_non_det_assignment(ref: json, lower_bound: json = None, upper_bound: json = None):
        non_det_assignment = {"ref": ref}

        if lower_bound is not None:
            non_det_assignment["lower-bound"] = lower_bound

        if upper_bound is not None:
            non_det_assignment["upper-bound"] = upper_bound

        return non_det_assignment

    @staticmethod
    def generate_destination(location: str, assignments: json = None, probability: json = None):
        destination = {"location": location}
        if assignments is not None:
            destination["assignments"] = assignments
        if probability is not None and ((not isinstance(probability, float) and not isinstance(probability, int)) or probability < 1):
            destination["probability"] = {"exp": probability}
        return destination

    @staticmethod
    def generate_edge(location: str, destinations, action=None, guard=None):
        destinations = [destination for destination in destinations if "probability" not in destination or destination["probability"]["exp"] != 0]  # remove 0-prob edges
        assert len(destinations) > 0
        edge = {"location": location, "destinations": destinations}
        if action is not None:
            edge["action"] = action
        if guard is not None:
            edge["guard"] = {"exp": guard}

        return edge

    @staticmethod
    def annotate_edge_guard(edge: json, guard_annotation: json) -> json:
        if "guard" not in edge:
            edge["guard"] = {"exp": guard_annotation}
        else:
            edge["guard"]["exp"] = Je.And(guard_annotation, edge["guard"]["exp"])
        return edge

    @staticmethod
    def generate_automaton(name: str, locations, initial_locations, edges, variables=None):
        assert len(locations) > 0 and len(initial_locations) > 0
        automaton = {"name": name, "locations": locations, "initial-locations": initial_locations, "edges": edges}
        if variables is not None:
            automaton["variables"] = variables
        return automaton

    # properties #######################################################################################################

    @staticmethod
    def generate_properties(properties: list):
        return {"properties": properties}

    @staticmethod
    def generate_property(name: str, expression):
        return {"name": name, "expression": expression}

    @staticmethod
    def generate_pa_property(name: str, predicates=None, start=None, reach=None, objective=None, file_name=None):
        return JaniStructureGenerator.generate_property(name, JaniStructureGenerator.generate_pa_expression(predicates, start, reach, objective, file_name))

    @staticmethod
    def generate_reachability_property(name: str, exp=None):
        expression = {"op": "filter", "fun": "∃", "states": {"op": "initial"}, "values": {"op": "F", "exp": exp}}
        return JaniStructureGenerator.generate_property(name, expression)

    # model ############################################################################################################

    @staticmethod
    def generate_model(name: str, model_type: str, automata, system, actions=None, variables=None, properties=None, constants=None):
        model = {"jani-version": 1, "name": name, "type": model_type, "automata": automata, "system": system}
        if actions is not None:
            model["actions"] = actions
        if variables is not None:
            model["variables"] = variables
        if constants is not None:
            model["constants"] = constants
        if properties is not None:
            model["properties"] = properties
        return model

    ####################################################################################################################
    # non standard #####################################################################################################

    @staticmethod
    def generate_label_def(name: str, element: json) -> json:
        return {"name": name, "element": element}

    @staticmethod
    def generate_label_ref(name: str) -> json:
        return {"label": name}

    @staticmethod
    def add_labels(struct: json, label_defs: list) -> json:
        struct["labels"] = label_defs
        return struct

    @staticmethod
    def generate_external(path: str) -> json:
        return {"file": path}

    @staticmethod
    def generate_free_variable_declaration(name: str, decl_type: json) -> json:
        variable_declaration = {"name": name, "type": decl_type}
        return variable_declaration

    # expressions # ####################################################################################################

    @staticmethod
    def generate_let(free_vars: list, exp: json) -> json:
        return exp if len(free_vars) == 0 else {"op": "let", "variables": free_vars, "expression": exp}

    @staticmethod
    def generate_location_values(automaton_list: list, location_list: list, index_list: list[int] = None):
        assert index_list is None or len(index_list) == 0 or isinstance(index_list[0], int)  # sanity due to changing parameter order
        index_list = list(range(0, len(automaton_list))) if index_list is None else index_list
        assert (len(automaton_list) == len(index_list) and len(index_list) == len(location_list))
        location_values = list()
        for automaton, index, location in zip(automaton_list, index_list, location_list):
            location_values.append({"automaton": automaton, "index": index, "location": location})
        #
        return location_values

    @staticmethod
    def generate_variable_values(var_list: list, value_list: list):
        assert (len(var_list) == len(value_list))
        state_variable_values = list()
        for var, val in zip(var_list, value_list):
            state_variable_values.append({"var": var, "value": val})

        return state_variable_values

    @staticmethod
    def generate_variable_values_from_map(var_values: dict):
        state_variable_values = list()
        for var in sorted(var_values.keys()):
            state_variable_values.append({"var": var, "value": var_values[var]})

        return state_variable_values

    @staticmethod
    def generate_state_values(location_values: list, state_variable_values: list):
        state_values = {}
        if len(location_values) > 0:
            state_values["locations"] = location_values
        if len(state_variable_values) > 0:
            state_values["variables"] = state_variable_values

        return state_values

    @staticmethod
    def generate_states_values(states_values: list):
        return {"op": "states-values", "values": states_values}

    @staticmethod
    def generate_states_values_from_variable_values(variable_states_values: list):
        return JaniStructureGenerator.generate_states_values([JaniStructureGenerator.generate_state_values([], variable_values) for variable_values in variable_states_values])

    @staticmethod
    def generate_state_condition_expression(location_values: list, state_var_constraint):
        condition = {"op": "state-condition"}
        if len(location_values) > 0:
            condition["locations"] = location_values
        if state_var_constraint is not None:
            condition["exp"] = state_var_constraint

        return condition

    @staticmethod
    def generate_objective_expression(goal=None, goal_potential=None, step_reward=None, accumulate=None):
        if accumulate is None:
            accumulate = []
        objective_exp = {"op": "objective"}
        if goal is not None:
            objective_exp["goal"] = goal
        if goal_potential is not None:
            objective_exp["goal-potential"] = goal_potential
        if step_reward is not None:
            objective_exp["step-reward"] = step_reward
        if len(accumulate) > 0:
            objective_exp["accumulate"] = accumulate
        return objective_exp

    @staticmethod
    def generate_pa_expression(predicates: list[json] = None, start: json = None, reach: json = None, objective: json = None, file_name: str = None):
        pa_exp = {"op": "PA"}

        if predicates is not None:
            pa_exp["predicates"] = predicates

        if start is not None:
            pa_exp["start"] = start

        if reach is not None:
            pa_exp["reach"] = reach

        if objective is not None:
            pa_exp["objective"] = objective

        if file_name is not None:
            pa_exp["file"] = file_name

        return pa_exp

    # variable (bounds) ################################################################################################

    @staticmethod
    def split_domain_into_intervals(var: str, var_lower, var_upper, num_split: int):
        split_intervals = []
        if num_split <= 0:
            return split_intervals

        split_size = var_upper - var_lower + 1
        while num_split > split_size:
            num_split = num_split - 1
        # num_split is at least 1, split_size is greater equal num_split:
        split_size = split_size // num_split

        lower = var_lower
        upper = var_lower + split_size - 1  # if num_splits == 1, this is var_upper
        split_intervals.append(JaniStructureGenerator.bound_var(var, lower, upper))  # 1th split
        num_split -= 1
        # rest splits:
        while num_split > 0 and upper < var_upper:
            lower = upper + 1
            upper = upper + split_size
            num_split -= 1
            if num_split == 0:
                upper = var_upper
            split_intervals.append(JaniStructureGenerator.bound_var(var, lower, upper))

        return split_intervals

    @staticmethod
    def lower_bound_var(var: str, lower):
        return Je.Le(lower, var)

    @staticmethod
    def upper_bound_var(var: str, upper):
        return Je.Le(var, upper)

    @staticmethod
    def bound_var(var: str, lower, upper):
        return Je.And(Je.Le(lower, var), Je.Le(var, upper))

    @staticmethod
    def lower_bounds(var: str, lower_bounds: list) -> list:
        lb_exps = list()
        for lb in lower_bounds:
            lb_exps.append(JaniStructureGenerator.lower_bound_var(var, lb))
        return lb_exps
