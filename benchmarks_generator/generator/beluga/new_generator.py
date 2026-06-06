#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random
import re 
from itertools import product

from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

random.seed(2020)

# Setting: The beluga is stationed at the front whereas the factory is stationed at the back. The trailers can unload from beluga. The truck is only at the back. 

class BelugaGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--swaps", type=int, default=50, help="How many swaps allowed for safe behaviour")
        # options for property generation (and thus not saved in model file):
        self.optionParser.add_argument("--single-safe-start-value", action="store_true", default=False, help="Fix safety variable to a single safe value at start.")
        self.optionParser.add_argument("--reachable-objective", action="store_true", default=False, help="Change objective to additional reachability conditions.")


class BelugaGenerator(JaniModelGeneratorPddlInJani):
    # constants:

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        
        if self.model_file is None:
            # model generation
            self.description_name = options.description
            self.max_num_swaps = options.swaps
        else:
            # property generation
            model = PythonUtils.load_json(self.model_file)
            constants = model["constants"]
            self.description_name = model["name"]
            _, self.max_num_swaps = self.read_constant(constants, "max_swaps")
            #
        self.single_safe_start_value = options.single_safe_start_value
        self.reachable_objective = options.reachable_objective

        self.model_type = JaniModelType.MDP
        self.model_name = self.description_name
        self.description = PythonUtils.load_json(self.description_name)
        self.num_racks = self.description["racks"]
        self.num_trailers = 1
        self.num_jigs = self.description["jigs"]
        self.num_belugas = 1
        self.production_lines = self.description["production-lines"]
        # self.max_num_swaps = self.description["max_swaps"]
        self.swap_unsafe = True
        
        # unique bounds. Others can be inferred using semantics.
        self.rack_max_capacity = (self.num_jigs // self.num_racks) + (self.num_jigs % self.num_racks)
        self.trailer_load = 7
        
        self.beluga_bounds = self.trailer_bounds = self.rack_bounds = self.truck_bounds = self.prod_line_bounds = [0, self.num_jigs]
        self.rack_size_bounds = [0, self.rack_max_capacity]
        self.next_line_bounds = [1, self.production_lines]
        

        # indexes:
        self.racks = list(range(0, self.num_racks))
        self.rack_indices = list(range(0, self.rack_max_capacity))
        self.jig_indices = list(range(1, self.num_jigs + 1))
        self.line_indices = list(range(1, self.production_lines + 1))
        self.line_orders = dict()
        for line in self.line_indices:
            line_order = self.description[f"pl{line - 1}"]
            self.line_orders[line] = line_order
        
        
        # variables: 
        # Every rack and beluga is modelled as a queue. Rather than using JANI's inbuilt functionality for arrays, we enumerate every index in the queue.
        self.num_jigs_on_rack = dict([(rack, "num_jigs_on_rack_" + str(rack)) for rack in self.racks])
        self.rack_vars = dict([((rack, index), f"rack_{rack}_{index}") for (rack, index) in product(self.racks, self.rack_indices)]) 
        self.trailer = "trailer_0"  # For unloading jigs from the beluga or pushing them onto racks
        self.beluga_vars = dict([(index, f"beluga_{index}") for index in self.jig_indices]) # num jigs on beluga
        self.next_line = "next_line"
        self.line_vars = dict([(line, f"line_{line}") for line in self.line_indices])
        self.truck = "truck"      # For moving jigs between two racks, or delivering the correct jig to the production line
        self.num_swaps = "num_swaps"
        
        self.beluga_variables = list(self.beluga_vars.values())
        self.rack_variables = list(self.num_jigs_on_rack.values()) + list(self.rack_vars.values())
        self.line_variables = [self.next_line] + list(self.line_vars.values())
        
        self.variable_names = self.rack_variables  + [self.trailer] + self.beluga_variables + self.line_variables + [self.truck] + [self.num_swaps] 
       

        # actions
        self.pop_beluga_action = "pop_beluga"
        self.push_rack_actions = dict([(rack, f"push_rack_{rack}") for rack in self.racks])
        self.pop_rack_actions = dict([(rack, f"pop_rack_{rack}") for rack in self.racks])
        self.deliver_jig_action = "deliver"
        self.send_back_action = "send_back"
        self.no_op_action = "no_op"
        self.action_labels = [self.pop_beluga_action] + list(self.push_rack_actions.values()) + list(self.pop_rack_actions.values()) + [self.deliver_jig_action] + [self.send_back_action] + [self.no_op_action]
        
        
        self.compute_model_initial_and_goal_state()

    # noinspection PyAttributeOutsideInit
    def compute_model_initial_and_goal_state(self):
        # initial
        
        self.initial_state = dict(
            [(rack_load, 0) for rack_load in self.num_jigs_on_rack.values()] +
            [(rack_idx, 0) for rack_idx in self.rack_vars.values()] +
            [(self.trailer, 0)] + [(self.next_line, 1)] + [(self.truck, 0)] + [(self.num_swaps, 0)] +
            [(self.line_vars[line], self.line_orders[line][0]) for line in self.line_indices] +
            [(self.beluga_vars[jig], jig) for jig in self.jig_indices]
        )

        # goal
        self.goal_state = dict([(line, 0) for line in self.line_vars.values()])

    # constraints #############################################################################################
    def is_empty_trailer(self):
        return Je.Le(self.trailer, 0)
    
    def is_empty_truck(self):
        return Je.Le(self.truck, 0)
    
    def not_is_empty_trailer(self):
        return Je.Gt(self.trailer, 0)
    
    def not_is_empty_truck(self):
        return Je.Gt(self.truck, 0)
    
    def beluga_has_jigs(self):
        return Je.Gt(self.beluga_vars[1], 0) # b1 > 0 implies b1 has a jig
    
    def not_is_empty_rack(self, rack: int):
        return Je.Gt(self.num_jigs_on_rack[rack], 0)
    
    def is_empty_rack(self, rack: int):
        return Je.Le(self.num_jigs_on_rack[rack], 0)
    
    def line_to_deliver(self, line: int):
        return Je.Eq(self.next_line, line)
    
    def truck_has_jig(self, jig: int):
        return Je.Eq(self.truck, jig)
    
    def rack_used_size(self, rack: int, size: int):
        return Je.Eq(self.num_jigs_on_rack[rack], size)
    
    def line_done_after_delivery(self, line: int):
        return Je.Eq(self.line_vars[line], self.get_current_jig_in_line(line, len(self.line_orders[line]) - 1))
    
    def line_not_done_after_delivery(self, line: int):
        return Je.Ne(self.line_vars[line], self.get_current_jig_in_line(line, len(self.line_orders[line]) - 1))
    
    def line_done(self, line: int):
        return Je.Le(self.line_vars[line], 0)
    
    def get_current_jig_in_line(self, line: int, order: int):
        if order >= len(self.line_orders[line]):
            return 0
        return self.line_orders[line][order]
    
    def get_next_jig_in_line(self, line: int, order: int):
        if order >= len(self.line_orders[line]) - 1: 
            return 0
        return self.line_orders[line][order + 1]
    
    ### Updates
    def unload_beluga_jig_on_trailer(self):
        return JaniStructureGenerator.generate_assignment(self.trailer, self.beluga_vars[1])
    
    def unload_truck_on_trailer(self):
        return JaniStructureGenerator.generate_assignment(self.trailer, self.truck)
    
    def unload_trailer(self):
        return JaniStructureGenerator.generate_assignment(self.trailer, 0)
    
    def shift_jigs_on_beluga(self):
        shifts = [JaniStructureGenerator.generate_assignment(self.beluga_vars[j], self.beluga_vars[j + 1]) for j in range(1, self.num_jigs)] + [JaniStructureGenerator.generate_assignment(self.beluga_vars[self.num_jigs], 0)]
        return shifts
    
    def unload_rack_jig_on_truck(self, rack: int):
        return JaniStructureGenerator.generate_assignment(self.truck, self.rack_vars[(rack, 0)])
    
    def unload_truck(self):
        return JaniStructureGenerator.generate_assignment(self.truck, 0)
    
    def shift_jigs_on_rack(self, rack: int):
        shifts = [JaniStructureGenerator.generate_assignment(self.rack_vars[(rack, j)], self.rack_vars[(rack, j + 1)]) for j in range(0, self.rack_max_capacity - 1)] + \
        [JaniStructureGenerator.generate_assignment(self.rack_vars[(rack, self.rack_max_capacity - 1)], 0)] + \
        [self.change_rack_used_size(rack, -1)]
        return shifts
    
    def load_jig_on_rack(self, rack: int, used_size: int):
        return [JaniStructureGenerator.generate_assignment(self.rack_vars[(rack, used_size)], self.trailer), self.change_rack_used_size(rack, 1)]
    
    def change_rack_used_size(self, rack: int, delta: int):
        return JaniStructureGenerator.generate_self_assignment(self.num_jigs_on_rack[rack], delta)
    
    def set_next_jig_in_line(self, line: int, order: int):
        return JaniStructureGenerator.generate_assignment(self.line_vars[line], self.get_next_jig_in_line(line, order))
    
    def set_next_line(self, line: int):
        return JaniStructureGenerator.generate_assignment(self.next_line, line)
    
    def inc_num_swaps(self):
        return JaniStructureGenerator.generate_self_assignment(self.num_swaps, 1)
    
    # Variable generation #################################################################################################

    
    def generate_num_jigs_on_rack_variables(self, rack: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.num_jigs_on_rack[rack], 0, self.rack_max_capacity, self.initial_state[self.num_jigs_on_rack[rack]])
    
    def generate_rack_variables(self, rack: int, index: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.rack_vars[(rack, index)], 0, self.num_jigs, self.initial_state[self.rack_vars[(rack, index)]])
    
    def generate_trailer_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.trailer, 0, self.num_jigs, self.initial_state[self.trailer])
    
    def generate_beluga_variables(self, index: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.beluga_vars[index], 0, self.num_jigs, self.initial_state[self.beluga_vars[index]])
    
    def generate_next_line_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.next_line, 1, self.production_lines, self.initial_state[self.next_line])
    
    def generate_line_variables(self, line: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.line_vars[line], 0, self.num_jigs, self.initial_state[self.line_vars[line]])
    
    def generate_truck_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck, 0, self.num_jigs, self.initial_state[self.truck])
    
    def generate_num_swaps_variable(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.num_swaps, 0, self.max_num_swaps, self.initial_state[self.num_swaps])
    
    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration("max_swaps", JaniStructureGenerator.generate_int_type(), self.max_num_swaps),
                JaniStructureGenerator.generate_constant_declaration("num_jigs", JaniStructureGenerator.generate_int_type(), self.num_jigs),
                JaniStructureGenerator.generate_constant_declaration("num_racks", JaniStructureGenerator.generate_int_type(), len(self.racks)),
                JaniStructureGenerator.generate_constant_declaration("rack_max_capacity", JaniStructureGenerator.generate_int_type(), self.rack_max_capacity),
                JaniStructureGenerator.generate_constant_declaration("num_production_lines", JaniStructureGenerator.generate_int_type(), len(self.line_indices))
                ]
        
    def generate_variables(self):
        variables = []
        variables += [self.generate_num_jigs_on_rack_variables(rack) for rack in self.racks]
        variables += [self.generate_rack_variables(rack, index) for (rack, index) in product(self.racks, self.rack_indices)]
        variables += [self.generate_trailer_variable()]
        variables += [self.generate_beluga_variables(index) for index in self.jig_indices]
        variables += [self.generate_next_line_variable()]
        variables += [self.generate_line_variables(line) for line in self.line_indices]
        variables += [self.generate_truck_variable()]
        variables += [self.generate_num_swaps_variable()]
        return variables

    ########### Actions Edge generation

    def generate_edge_aux(self, destinations: list, action: str, guard) -> json:
        return JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, guard)
        
    def generate_edges(self):
        edges = self.generate_no_op_edges() + self.generate_send_back_edge() + self.generate_pop_beluga_edge()
        for rack in self.racks:
            edges += self.generate_pop_rack_edge(rack)
            for size in range(self.rack_max_capacity):
                edges += self.generate_push_rack_edge(rack, size)
        
        for line in self.line_indices:
            for order in range(len(self.line_orders[line])):    
                edges += self.generate_deliver_jig_edge(line, order)
        # edges = self.generate_jig_edges() + (self.generate_move_edges() if self.move_aware else []) + self.generate_rack_edges() + self.generate_push_truck_trailer_edges()
        return edges
    
    def generate_no_op_edges(self):
        probabality = 1/(self.production_lines - 1)
        edges = []
        action = self.no_op_action
        for line in self.line_indices:
            guard = Je.And([self.line_to_deliver(line), self.line_done(line)])
            destinations = []
            for other_line in self.line_indices:
                if other_line == line:
                    continue
                destinations += [self.generate_destination_aux([self.set_next_line(other_line)], probabality)]
            edges += [self.generate_edge_aux(destinations, action, guard)]
        return edges
    
    def generate_send_back_edge(self):
        action = self.send_back_action
        guard = Je.And([self.is_empty_trailer(), self.not_is_empty_truck()])
        destinations = [self.generate_destination_aux([self.unload_truck_on_trailer(), self.unload_truck(), self.inc_num_swaps()])]
        edges = [self.generate_edge_aux(destinations, action, guard)]
        return edges
    
    def generate_pop_rack_edge(self, rack: int):
        action = self.pop_rack_actions[rack]
        guard = Je.And([self.is_empty_truck(), self.not_is_empty_rack(rack)])
        update = [self.unload_rack_jig_on_truck(rack)] + self.shift_jigs_on_rack(rack)
        destinations = [self.generate_destination_aux(update)]
        edges = [self.generate_edge_aux(destinations, action, guard)]
        return edges
    
    def generate_pop_beluga_edge(self):
        action = self.pop_beluga_action
        guard = Je.And([self.is_empty_trailer(), self.beluga_has_jigs()])
        update = [self.unload_beluga_jig_on_trailer()] + self.shift_jigs_on_beluga()
        destinations = [self.generate_destination_aux(update)]
        edges = [self.generate_edge_aux(destinations, action, guard)]
        return edges
    
    def generate_push_rack_edge(self, rack: int, size: int):
        action = self.push_rack_actions[rack]
        guard = Je.And([self.not_is_empty_trailer(), self.rack_used_size(rack, size)])
        update = [self.unload_trailer()] + self.load_jig_on_rack(rack, size)
        destinations = [self.generate_destination_aux(update)]
        edges = [self.generate_edge_aux(destinations, action, guard)]
        return edges
    
    def generate_deliver_jig_edge(self, line: int, order: int):
        action = self.deliver_jig_action
        guard = Je.And([self.line_to_deliver(line), self.truck_has_jig(self.get_current_jig_in_line(line, order))])
        common_update = [self.unload_truck(), self.set_next_jig_in_line(line, order)]
        probabality = 1/self.production_lines
        destinations = []
        for new_line in self.line_indices:
            destinations += [self.generate_destination_aux(common_update + [self.set_next_line(new_line)], probabality)]
        edges = [self.generate_edge_aux(destinations, action, guard)]
        return edges

    # property generation ##############################################################################################
    def swaps_exhausted(self):
        return Je.Ge(self.num_swaps, self.max_num_swaps)
    
    def swaps_possible(self):
        return Je.Lt(self.num_swaps, self.max_num_swaps)
    
    def is_safe(self):
        return self.swaps_possible()
    
    def is_unsafe(self) -> json:
        return self.swaps_exhausted()
    
    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], Je.And(self.generate_goal_expression(), self.is_safe()))        
        return JaniStructureGenerator.generate_objective_expression(goal=goal, goal_potential=self.generate_goal_potential())

    def generate_goal_potential(self) -> json:
        if not self.use_goal_potential:
            return None
        return None

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.is_unsafe())

    # Start Condition ###########################################################################################################
    
    def generate_start(self):
        constraints = list()
      
        constraints += [Je.Le(self.num_swaps, 0)]   # Constraint: safe at start
        
        # Constraint: Racks are initially empty
        for rack in self.racks:
            constraints += [Je.Le(self.num_jigs_on_rack[rack], 0)]
            for index in self.rack_indices: 
                constraints += [Je.Le(self.rack_vars[(rack, index)], 0)]
                
        # Constraint: Trailer is empty
        constraints += [self.is_empty_trailer()]
        
        # Constraint: Each jig is in the beluga
        for jig in self.jig_indices:
            constraints += [Je.Gt(self.beluga_vars[jig], 0), Je.Le(self.beluga_vars[jig], self.num_jigs)]
        
        # Constraint: each beluga index must have a different jig
        for j1 in self.jig_indices:
            for j2 in self.jig_indices:
                if j1 >= j2: 
                    continue
                constraints.append(Je.Ne(self.beluga_vars[j1], self.beluga_vars[j2]))
             
        # Constraint: A production line is chosen    
        constraints += [Je.Gt(self.next_line, 0), Je.Le(self.next_line, self.production_lines)]
        
        # Constraint: Production lines initialized to request the first jig in a predetermined order
        for line in self.line_indices:
            constraints += [Je.Eq(self.line_vars[line], self.get_current_jig_in_line(line, 0))]
        
        # Constraint: Truck is Empty
        constraints += [self.is_empty_truck()]
 
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, number_starts: int) -> list:
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=number_starts, default_state=self.initial_state)
        return states_values.generate_states_values()

    # Input Variables #########################################################################################################

    # In ECAI24 and 25 papers, we had removed swaps as an input from the policy. This has been readded here due to internal discussion that the policy might find this information useful during decision making.
    def get_nn_inputs(self) -> list:
        nn_vars = list(self.variable_names)
        return nn_vars

    ####################################################################################################################


if __name__ == "__main__":
    args = BelugaGenerationOptionParser().arg_parse()
    generator = BelugaGenerator(args)
    generator.generate()
