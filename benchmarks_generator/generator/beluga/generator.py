#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import random
import re 

from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator, JaniModelType
from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani
from python_utils import PythonUtils

random.seed(2020)

# NOTE: The beluga is stationed at the front whereas the factory is stationed at the back. The trailers can go wherever. The truck is only at the back. Unstacking from rack on back goes to truck whereas from front goes to trailer

class BelugaGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        self.optionParser.add_argument("--description", type=str, default=None, help="Description of the instance in json format.")
        self.optionParser.add_argument("--failing-prob", type=float, default=0, help="Probability to fail a action.")
        # options for property generation (and thus not saved in model file):
        self.optionParser.add_argument("--single-safe-start-value", action="store_true", default=False, help="Fix safety variable to a single safe value at start.")
        self.optionParser.add_argument("--reachable-objective", action="store_true", default=False, help="Change objective to additional reachability conditions.")


class BelugaGenerator(JaniModelGeneratorPddlInJani):
    # constants:
    failing_PROB_NAME = "failing_prob"

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        
        if self.model_file is None:
            # model generation
            self.description_name = options.description
            self.failing_prob = options.failing_prob
        else:
            # property generation
            model = PythonUtils.load_json(self.model_file)
            constants = model["constants"]
            _, self.failing_prob = self.read_constant(constants, self.failing_PROB_NAME)
            self.description_name = model["name"]
            #
        self.single_safe_start_value = options.single_safe_start_value
        self.reachable_objective = options.reachable_objective

        self.model_type = JaniModelType.MDP if self.failing_prob > 0 else JaniModelType.LTS
        self.model_name = self.description_name
        self.description = PythonUtils.load_json(self.description_name)
        self.num_racks = self.description["racks"]
        self.num_trailers = self.description["trailers"]
        self.num_jigs = self.description["jigs"]
        self.num_hangars = self.description["hangars"]
        self.num_belugas = self.description["belugas"]
        self.move_aware = self.description["move-aware"]
        self.load_aware = self.description["load-aware"]
        self.production_lines = self.description["production-lines"]
        try:
            self.max_num_swaps = self.description["max_swaps"]
            self.swap_unsafe = False
        except:
            self.max_num_swaps = int(self.num_jigs*3)
            self.swap_unsafe = True
        assert(self.num_trailers == 1 and self.num_hangars == 1 and self.num_belugas == 1 and self.production_lines == 2) # hardcoded for exactly 1 trailer.

        # indexes:
        self.rack_indexes = list(range(0, self.num_racks))
        self.trailer_indexes = list(range(0, self.num_trailers))
        self.hangar_indexes = list(range(0, self.num_hangars))
        self.beluga_indexes = list(range(0, self.num_belugas))
        self.jig_indexes = list(range(0, self.num_jigs))
        self.line_indexes = list(range(0, self.production_lines))
        self.line_orders = list()
        for i in self.line_indexes:
            line_order = self.description[f"pl{i}"]
            self.line_orders.append(line_order)
        
        # variables: 
        self.rack_loads = dict([(rack, "rack_load_" + str(rack)) for rack in self.rack_indexes])
        self.num_jigs_on_rack = dict([(rack, "num_jigs_on_rack_" + str(rack)) for rack in self.rack_indexes])
        self.front_rack_vars = dict([(rack, "front_rack_" + str(rack)) for rack in self.rack_indexes])
        self.back_rack_vars = dict([(rack, "back_rack_" + str(rack)) for rack in self.rack_indexes])
        self.trailer_vars = dict([(trailer, "trailer_" + str(trailer)) for trailer in self.trailer_indexes] if self.move_aware else [])               # trailer position
        self.empty_trailer_vars = dict([(trailer, "empty_trailer_" + str(trailer)) for trailer in self.trailer_indexes])   # is trailer empty
        self.hangar_vars = dict([(hangar, "hangar_" + str(hangar)) for hangar in self.hangar_indexes] if self.move_aware else [])                     # is hangar empty
        self.beluga_vars = dict([(beluga, "beluga_" + str(beluga)) for beluga in self.beluga_indexes])                     # num jigs on beluga
        self.beluga_phases = dict([(beluga, f"beluga_{beluga}_phase") for beluga in self.beluga_indexes] if self.load_aware else [])
        self.jig_vars = dict([(jig, "jig_" + str(jig)) for jig in self.jig_indexes])                                       # jig position
        self.front_jig_vars = dict([(jig, "front_jig_" + str(jig)) for jig in self.jig_indexes])
        self.back_jig_vars = dict([(jig, "back_jig_" + str(jig)) for jig in self.jig_indexes])
        self.next_jig = "next_jig"
        self.line_vars = dict([(line, f"line_{line}") for line in self.line_indexes])
        self.truck = "empty_truck"      # For moving trailer between two racks, we use this truck
        self.num_swaps = "num_swaps"
        
        self.rack_variables = list(self.rack_loads.values())  + list(self.front_rack_vars.values()) + list(self.back_rack_vars.values())
        self.jig_variables = list(self.jig_vars.values()) + list(self.front_jig_vars.values()) + list(self.back_jig_vars.values())
        self.trailer_variables = list(self.trailer_vars.values()) +  list(self.empty_trailer_vars.values())
        self.line_variables = [self.next_jig] + list(self.line_vars.values())
        
        self.variable_names = self.jig_variables + self.rack_variables + self.trailer_variables + \
            list(self.hangar_vars.values()) + list(self.beluga_vars.values()) + list(self.beluga_phases.values()) + \
            self.line_variables + [self.truck] + [self.num_swaps]

        # unique bounds. Others can be inferred using semantics.
        self.rack_max_capacity = self.num_jigs*3
        self.trailer_load = 7
        self.jig_load = 1
        self.jig_position_bounds = [0, self.num_belugas*self.num_jigs + self.num_trailers + self.num_racks + 1] # Beluga, Trailer, Rack, Truck
        self.jig_front_back_bounds = [-1, self.num_jigs - 1]
        self.trailer_position_bounds = [-self.num_racks, self.num_racks + self.num_hangars + self.num_belugas + 1] # rack front, back. Beluga. Storage. Hangar, Front Truck
        self.line_bounds = [-1, self.num_jigs - 1]

        # actions
        self.action_labels_beluga_operations = dict([(beluga, ("load_or_unload_" if self.load_aware else "unload_") + self.beluga_vars[beluga]) for beluga in self.beluga_indexes])
        self.action_labels_push_front_rack = dict([(rack, "push_front_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_push_back_rack = dict([(rack, "push_back_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_pop_front_rack = dict([(rack, "pop_front_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_pop_back_rack = dict([(rack, "pop_back_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_push_rack = dict([(rack, "push_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_pop_rack = dict([(rack, "pop_rack_" + str(rack)) for rack in self.rack_indexes])
        self.action_labels_send_to_factory = "send_to_factory"
        self.action_labels_push_truck_trailer = dict([(trailer, f"unload_truck_{trailer}") for trailer in self.trailer_indexes])
        if self.move_aware:
            move_hangar_rack_combinations = [((hangar, side, rack), f"move_{self.hangar_vars[hangar]}_{side}_rack_{rack}") for hangar in self.hangar_indexes for rack in self.rack_indexes for side in ["front", "back"]]
            self.action_labels_move_hangar_rack = dict(move_hangar_rack_combinations)
            self.action_labels_move_hangar_storage = dict([(hangar, f"send_{self.hangar_vars[hangar]}_storage") for hangar in self.hangar_indexes])
            self.action_labels_move_rack_truck = dict([((rack, side), f"move_{side}_rack_{rack}_truck") for rack in self.rack_indexes for side in ["front", "back"]])
            self.action_labels_move_truck_rack = dict([((rack, side), f"move_truck_{side}_rack_{rack}") for rack in self.rack_indexes for side in ["front", "back"]])
            self.action_labels_move_rack_rack = dict([((rack_i, side_i, rack_f, side_f), f"move_{side_i}_rack_{rack_i}_to_{side_f}_rack_{rack_f}") for rack_i in self.rack_indexes for rack_f in self.rack_indexes for side_i in ["front", "back"] for side_f in ["front", "back"]])
            self.action_labels_move_rack_storage = dict([((rack, side), f"move_{side}_rack_{rack}_storage") for rack in self.rack_indexes for side in ["front", "back"]])
            self.action_labels_move_storage_rack = dict([((rack, side), f"move_storage_{side}_rack_{rack}") for rack in self.rack_indexes for side in ["front", "back"]])
            self.action_labels_move_trailer_to_beluga = dict([((trailer, beluga), f"move_{self.trailer_vars[trailer]}_to_" + str(self.beluga_vars[beluga])) for beluga in self.beluga_indexes for trailer in self.trailer_indexes])
            self.move_operations = list(self.action_labels_move_hangar_rack.values()) + list(self.action_labels_move_hangar_storage.values()) +\
                    list(self.action_labels_move_truck_rack.values()) + list(self.action_labels_move_rack_truck.values()) + \
                    list(self.action_labels_move_rack_storage.values()) + list(self.action_labels_move_storage_rack.values()) + list(self.action_labels_move_trailer_to_beluga.values())            
        # else:
        #     self.action_labels_truck_push_front_rack = dict([(rack, "truck_push_front_rack_" + str(rack)) for rack in self.rack_indexes])
        #     self.action_labels_truck_push_back_rack = dict([(rack, "truck_push_back_rack_" + str(rack)) for rack in self.rack_indexes])
        if self.move_aware:
            self.rack_operations = list(self.action_labels_push_rack.values()) + list(self.action_labels_pop_rack.values())
        else:
            self.rack_operations = list(self.action_labels_push_front_rack.values()) + list(self.action_labels_push_back_rack.values()) +\
                list(self.action_labels_pop_front_rack.values()) + list(self.action_labels_pop_back_rack.values())
                # + list(self.action_labels_truck_push_front_rack.values()) + list(self.action_labels_truck_push_back_rack.values())
        
        self.action_labels = list(self.action_labels_beluga_operations.values()) + [self.action_labels_send_to_factory] + self.rack_operations + (self.move_operations if self.move_aware else list(self.action_labels_push_truck_trailer.values()))
        self.compute_model_initial_and_goal_state()

    # noinspection PyAttributeOutsideInit
    def compute_model_initial_and_goal_state(self):
        # initial
        self.initial_state = dict([(rack_load, self.rack_max_capacity) for rack_load in self.rack_loads.values()]
                                  + [(front_rack, -1) for front_rack in self.front_rack_vars.values()]  + [(back_rack, -1) for back_rack in self.back_rack_vars.values()]
                                  + ([(trailer, self.trailer_assignment(0, "storage")) for trailer in self.trailer_vars.values()])
                                  + [(trailer, 1) for trailer in self.empty_trailer_vars.values()] + [(hangar, 1) for hangar in self.hangar_vars.values()]
                                  + [(front, -1) for front in self.front_jig_vars.values()] + [(back, -1) for back in self.back_jig_vars.values()]
                                  + [(self.next_jig, 0)] + [(self.line_vars[line], self.line_orders[i][0]) for i, line in enumerate(self.line_vars)] \
                                  + [(bp, 1) for bp in self.beluga_phases.values()] + [(self.truck, 1)]
                                  + [(self.jig_vars[jig], jig) for jig in self.jig_indexes] + [(self.beluga_vars[beluga], self.num_jigs if beluga == 0 else 0) for beluga in self.beluga_indexes]
                                  + [(self.num_swaps, 0)]
                                  )
        # goal
        load_goal = [(self.jig_vars[jig], jig) for jig in self.jig_indexes] + [(self.beluga_vars[beluga], self.num_jigs if beluga == 0 else 0) for beluga in self.beluga_indexes] + [(bp, 0) for bp in self.beluga_phases.values()]
        self.goal_state = dict((load_goal if self.load_aware else [(self.jig_vars[jig], self.jig_position_assignment(0, "factory")) for jig in self.jig_indexes]))

    # constraints #############################################################################################
    def trailer_assignment(self, location: int, location_type: str):
        assignments = {
            "back_rack": -location - 1,
            "front_rack": location,
            "hangar": self.num_racks + location,
            "beluga": self.num_racks + self.num_hangars + location,
            "storage": self.num_racks + self.num_hangars + self.num_belugas,
            "truck": self.num_racks + self.num_hangars + self.num_belugas + 1
        }
        return assignments[location_type]

    # Trailer position assignments
    def trailer_front_of_beluga(self, trailer: int, beluga: int):
        return Je.Eq(self.trailer_vars[trailer], self.trailer_assignment(beluga, "beluga"))
    
    def trailer_not_front_of_beluga(self, trailer: int, beluga: int):
        return Je.Ne(self.trailer_vars[trailer], self.trailer_assignment(beluga, "beluga"))

    def trailer_front_of_rack(self, trailer: int, rack: int):
        return Je.Eq(self.trailer_vars[trailer], self.trailer_assignment(rack, "front_rack"))
    
    def trailer_not_front_of_rack(self, trailer: int, rack: int):
        return Je.Ne(self.trailer_vars[trailer], self.trailer_assignment(rack, "front_rack"))
    
    def trailer_back_of_rack(self, trailer: int, rack: int):
        return Je.Eq(self.trailer_vars[trailer], self.trailer_assignment(rack, "back_rack"))

    def trailer_not_back_of_rack(self, trailer: int, rack: int):
        return Je.Ne(self.trailer_vars[trailer], self.trailer_assignment(rack, "back_rack"))

    def trailer_on_hangar(self, trailer: int, hangar: int):
        return Je.Eq(self.trailer_vars[trailer],  self.trailer_assignment(hangar, "hangar"))
    
    def trailer_in_storage(self, trailer: int):
        return Je.Eq(self.trailer_vars[trailer], self.trailer_assignment(0, "storage"))
    
    def trailer_on_truck(self, trailer: int):
        return Je.Eq(self.trailer_vars[trailer], self.trailer_assignment(0, "truck"))

    def trailer_back_of_any_rack(self, trailer: int):
        return Je.Le(self.trailer_vars[trailer], -1)
    
    # Jig position assignments
    def jig_position_assignment(self, location: int, location_type : str):
        assignments =  {
            "trailer": self.num_racks + (self.num_jigs * self.num_belugas) + location,
            "rack": location + (self.num_jigs * self.num_belugas),
            "truck": self.num_jigs*self.num_belugas + self.num_racks + self.num_trailers,
            "factory": self.num_jigs*self.num_belugas + self.num_racks + self.num_trailers + 1
        }
        return assignments[location_type]
    
    def jig_on_trailer(self, jig: int, trailer: int):
        return Je.Eq(self.jig_vars[jig], self.jig_position_assignment(trailer, "trailer"))
        
    def jig_on_rack(self, jig: int, rack: int):
        return Je.Eq(self.jig_vars[jig], self.jig_position_assignment(rack, "rack")) 
    
    def jig_not_on_rack(self, jig: int, rack: int): 
        return Je.Ne(self.jig_vars[jig], self.jig_position_assignment(rack, "rack"))
    
    def jig_on_beluga(self, jig: int, beluga: int):
        return Je.And([Je.Le(self.jig_vars[jig], self.num_jigs*(beluga + 1) - 1), Je.Ge(self.jig_vars[jig], self.num_jigs*beluga)])
    
    def jig_on_truck(self, jig: int):
        return Je.Eq(self.jig_vars[jig], self.jig_position_assignment(0, "truck"))
    
    def is_jig_delivered(self, jig: int):
        return Je.Eq(self.jig_vars[jig], self.jig_position_assignment(0, "factory"))  

    def next_unload_jig_on_beluga(self, jig: int, beluga: int):
        return Je.Eq(self.jig_vars[jig], Je.Add(self.beluga_vars[beluga], -1 + self.num_jigs * beluga))
    
    def is_jig_factory_or_rack(self, jig: int) -> json:
        return Je.Or([self.is_jig_delivered(jig)] + [self.jig_on_rack(jig, rack) for rack in self.rack_indexes])
    
    def atleast_one_jig_at_position(self, position: int) -> json:
        return Je.Or([Je.Eq(self.jig_vars[jig], position) for jig in self.jig_indexes])
    
    # front/back assignments for jigs & racks:
    def jig_front_of_rack(self, jig: int, rack: int):
        return Je.Eq(self.front_rack_vars[rack], jig)
    
    def jig_not_front_of_rack(self, jig: int, rack: int):
        return Je.Ne(self.front_rack_vars[rack], jig)

    def jig_back_of_rack(self, jig: int, rack: int):
        return Je.Eq(self.back_rack_vars[rack], jig)
    
    def jig_not_back_of_rack(self, jig: int, rack: int):
        return Je.Ne(self.back_rack_vars[rack], jig)  
      
    def jig_front_of_jig(self, jig: int, jig1: int):
        return Je.Eq(self.front_jig_vars[jig], jig1)
    
    def jig_back_of_jig(self, jig: int, jig1: int):
        return Je.Eq(self.back_jig_vars[jig], jig1)
    
    # Load constraints:
    def jig_smaller_than_trailer(self, jig: int, trailer: int):
        return True

    def jig_smaller_than_rack(self, jig: int, rack: int):
        return Je.Ge(self.rack_loads[rack], self.jig_load)
    
    def is_jig_next_for_factory(self, jig: int):
        return Je.Eq(self.next_jig, jig)
    
    def all_jigs_delivered(self):
        return Je.Ge(self.next_jig, 0)
    
    # Beluga constraints:    
    def beluga_in_loading_phase(self, beluga: int):
        return Je.Le(self.beluga_phases[beluga], 0)
    
    def beluga_in_unloading_phase(self, beluga: int):
        return Je.Ge(self.beluga_phases[beluga], 1)
    
    def beluga_has_one_jig(self, beluga: int):
        return Je.Eq(self.beluga_vars[beluga], 1)
    
    def beluga_has_more_than_one_jig(self, beluga: int):
        return Je.Ge(self.beluga_vars[beluga], 2)
    
    def jigs_to_unload(self, beluga: int, i : int):
        return Je.Ge(self.beluga_vars[beluga], i)
    
    # Empty constraints:    
    def is_empty_truck(self):
        return Je.Ge(self.truck, 1)
    
    def is_not_empty_truck(self):
        return Je.Le(self.truck, 0)
    
    def is_empty_trailer(self, trailer: int):
        return Je.Ge(self.empty_trailer_vars[trailer], 1)
    
    def is_empty_rack(self, rack: int):
        return Je.And([self.rack_back_empty(rack), self.rack_front_empty(rack), self.rack_load_empty(rack), self.rack_has_no_jigs(rack)])
    
    def is_non_empty_rack(self, rack: int):
        return Je.And([Je.Ge(self.front_rack_vars[rack], 0), Je.Ge(self.back_rack_vars[rack], 0), self.rack_has_atleast_one_jig(rack)])
    
    def is_hangar_empty(self, hangar: int):
        return Je.Ge(self.hangar_vars[hangar], 1)
    
    def rack_front_empty(self, rack: int):
        return Je.Le(self.front_rack_vars[rack], -1)
    
    def rack_back_empty(self, rack: int):
        return Je.Le(self.back_rack_vars[rack], -1)
    
    def rack_load_empty(self, rack: int):
        return Je.Ge(self.rack_loads[rack], self.rack_max_capacity)
    
    def rack_front_clear(self, rack: int):
        return Je.And([self.trailer_not_front_of_rack(trailer, rack) for trailer in self.trailer_indexes])
    
    def rack_back_clear(self, rack: int):
        return Je.And([self.trailer_not_back_of_rack(trailer, rack) for trailer in self.trailer_indexes])
    
    def rack_has_atleast_one_jig(self, rack: int): 
        return Je.Ge(self.num_jigs_on_rack[rack], 1)
        
    def rack_has_no_jigs(self, rack: int):
        return Je.Le(self.num_jigs_on_rack[rack], 0)
    
    def rack_has_one_jig(self, rack: int):
        return Je.Eq(self.num_jigs_on_rack[rack], 1)
    
    def rack_has_many_jigs(self, rack: int):
        return Je.Ge(self.num_jigs_on_rack[rack], 2)
    
    def beluga_front_clear(self, beluga: int):
        return Je.And([self.trailer_not_front_of_beluga(trailer, beluga) for trailer in self.trailer_indexes])
    
    def swaps_possible(self):
        return Je.Le(self.num_swaps, self.max_num_swaps - 1)
    
    def line_done(self, line: int):
        return Je.Le(self.line_vars[line], -1)
    
    def line_not_done(self, line: int):
        return Je.Ge(self.line_vars[line], 0)
    
    def is_line_next_jig(self, line: int, jig: int):
        return Je.Eq(self.line_vars[line], jig)

    # UPDATE Assignments ####################################################################################################
    def set_jig_on_trailer(self, jig: int, trailer: int):
        return JaniStructureGenerator.generate_assignment(self.jig_vars[jig], self.jig_position_assignment(trailer, "trailer"))
    
    def set_jig_on_beluga(self, jig: int, beluga: int):
        return JaniStructureGenerator.generate_assignment(self.jig_vars[jig], Je.Add(self.beluga_vars[beluga], self.num_jigs * beluga))
    
    def set_jig_on_truck(self, jig: int):
        return JaniStructureGenerator.generate_assignment(self.jig_vars[jig], self.jig_position_assignment(0, "truck"))
    
    def set_jig_on_factory(self, jig: int):
        return JaniStructureGenerator.generate_assignment(self.jig_vars[jig], self.jig_position_assignment(0, "factory"))
    
    def set_trailer_empty(self, trailer: int):
        return JaniStructureGenerator.generate_assignment(self.empty_trailer_vars[trailer], 1)
    
    def set_trailer_not_empty(self, trailer: int):
        return JaniStructureGenerator.generate_assignment(self.empty_trailer_vars[trailer], 0)
    
    def dec_jigs_on_beluga(self, beluga: int):
        return JaniStructureGenerator.generate_self_assignment(self.beluga_vars[beluga], -1)

    def inc_jigs_on_beluga(self, beluga: int):
        return JaniStructureGenerator.generate_self_assignment(self.beluga_vars[beluga], 1)
    
    def inc_num_swaps(self):
        return JaniStructureGenerator.generate_self_assignment(self.num_swaps, 1)
    
    def dec_jigs_on_rack(self, rack: int):
        return JaniStructureGenerator.generate_self_assignment(self.num_jigs_on_rack[rack], -1)
    
    def inc_jigs_on_rack(self, rack: int):
        return JaniStructureGenerator.generate_self_assignment(self.num_jigs_on_rack[rack], 1)
    
    def set_beluga_loading(self, beluga: int):
        return JaniStructureGenerator.generate_assignment(self.beluga_phases[beluga], 0)
    
    def set_jig_on_rack(self, jig: int, rack: int):
        return JaniStructureGenerator.generate_assignment(self.jig_vars[jig], self.jig_position_assignment(rack, "rack"))
    
    def dec_rack_load(self, rack: int, jig: int):
        return JaniStructureGenerator.generate_self_assignment(self.rack_loads[rack], -self.jig_load)
    
    def inc_rack_load(self, rack: int, jig: int):
        return JaniStructureGenerator.generate_self_assignment(self.rack_loads[rack], self.jig_load)
    
    def set_front_rack(self, rack: int, jig: json):
        return JaniStructureGenerator.generate_assignment(self.front_rack_vars[rack], jig)
    
    def set_back_rack(self, rack: int, jig: json):
        return JaniStructureGenerator.generate_assignment(self.back_rack_vars[rack], jig)

    def set_jig_front(self, jig: int, jig1: json):
        return JaniStructureGenerator.generate_assignment(self.front_jig_vars[jig], jig1)
    
    def set_jig_back(self, jig: int, jig1: json):
        return JaniStructureGenerator.generate_assignment(self.back_jig_vars[jig], jig1)
    
    def set_trailer_on_hangar(self, trailer: int, hangar: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(hangar, "hangar"))
    
    def set_trailer_front_of_rack(self, trailer: int, rack: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(rack, "front_rack"))
    
    def set_trailer_back_of_rack(self, trailer: int, rack: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(rack, "back_rack"))
    
    def set_trailer_front_of_storage(self, trailer: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(0, "storage"))
    
    def set_trailer_front_of_beluga(self, trailer: int, beluga: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(beluga, "beluga"))
    
    def set_trailer_front_of_truck(self, trailer: int):
        return JaniStructureGenerator.generate_assignment(self.trailer_vars[trailer], self.trailer_assignment(0, "truck"))
    
    def set_hangar_empty(self, hangar: int):
        return JaniStructureGenerator.generate_assignment(self.hangar_vars[hangar], 1)
    
    def set_hangar_not_empty(self, hangar: int):
        return JaniStructureGenerator.generate_assignment(self.hangar_vars[hangar], 0)
    
    def set_truck_empty(self):
        return JaniStructureGenerator.generate_assignment(self.truck, 1)
    
    def set_truck_not_empty(self):
        return JaniStructureGenerator.generate_assignment(self.truck, 0)
    
    def get_next_jig_to_deliver(self, next_order):
        return JaniStructureGenerator.generate_assignment(self.next_jig, next_order)
    
    def set_line_done(self, line: int):
        return JaniStructureGenerator.generate_assignment(self.line_vars[line], -1)
    
    def set_line_next(self, line: int, order: int):
        return JaniStructureGenerator.generate_assignment(self.line_vars[line], self.line_orders[line][order + 1])

    def set_next_jig(self, line: int):
        return JaniStructureGenerator.generate_assignment(self.next_jig, self.line_vars[line])
    
    def unload_beluga(self, beluga: int, trailer: int, jig: int, set_loading: bool):
        if set_loading and self.load_aware:
            return [self.set_trailer_not_empty(trailer), self.set_jig_on_trailer(jig, trailer), self.dec_jigs_on_beluga(beluga), self.set_beluga_loading(beluga)]
        else:
            return [self.set_trailer_not_empty(trailer), self.set_jig_on_trailer(jig, trailer), self.dec_jigs_on_beluga(beluga)]

    def load_beluga(self, beluga: int, trailer: int, jig: int):
        return [self.set_trailer_empty(trailer), self.set_jig_on_beluga(jig, beluga), self.inc_jigs_on_beluga(beluga)]
    
    def push_front_rack(self, rack: int, trailer: int, jig: int, front_rack: int):
        if front_rack == -1:
            return [self.set_trailer_empty(trailer), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_back_rack(rack, jig), self.set_jig_front(jig, -1), self.set_jig_back(jig, -1), self.inc_num_swaps()]
        return [self.set_trailer_empty(trailer), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_jig_back(jig, front_rack), self.set_jig_front(front_rack, jig), self.inc_num_swaps()]

    def push_back_rack(self, rack: int, trailer: int, jig: int, back_rack: int):
        if back_rack == -1:
            return [self.set_trailer_empty(trailer), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_back_rack(rack, jig), self.set_jig_front(jig, -1), self.set_jig_back(jig, -1), self.inc_num_swaps()]
        return [self.set_trailer_empty(trailer), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_back_rack(rack, jig), self.set_jig_front(jig, back_rack), self.set_jig_back(back_rack, jig), self.inc_num_swaps()]

    def push_truck_front_rack(self, rack: int, jig: int, rack_empty: bool):
        if rack_empty:
            return [self.set_truck_empty(), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_back_rack(rack, jig)]
        return [self.set_truck_empty(), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_jig_back(jig, self.back_rack_vars[rack])]

    def push_truck_back_rack(self, rack: int, jig: int, rack_empty: bool):
        if rack_empty:
            return [self.set_truck_empty(), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_front_rack(rack, jig), self.set_back_rack(rack, jig)]
        return [self.set_truck_empty(), self.dec_rack_load(rack, jig), self.set_jig_on_rack(jig, rack), self.set_back_rack(rack, jig), self.set_jig_front(jig, self.front_rack_vars[rack])]

    def push_truck_trailer(self, jig: int, trailer: int):
        return [self.set_truck_empty(), self.set_trailer_not_empty(trailer), self.set_jig_on_trailer(jig, trailer)]

    def pop_front_rack(self, rack: int, jig: int, back_jig: int):
        return [self.set_truck_not_empty(), self.set_jig_on_truck(jig), self.set_jig_back(jig, -1), self.inc_rack_load(rack, jig), self.set_front_rack(rack, back_jig)] + ([self.set_back_rack(rack, -1)] if back_jig == -1 else [self.set_jig_front(back_jig, -1)])

    def pop_back_rack(self, rack: int, jig: int, front_jig: int):
        return [self.set_truck_not_empty(), self.set_jig_on_truck(jig), self.set_jig_front(jig, -1),  self.inc_rack_load(rack, jig), self.set_back_rack(rack, front_jig)] + ([self.set_front_rack(rack, -1)] if front_jig == -1 else [self.set_jig_back(front_jig, -1)])
           
    def send_factory(self, jig: int = 0, other_jig: int = 0):
        destinations = []
        generic = [self.set_truck_empty(), self.set_jig_on_factory(jig)]
        find_line = 0
        order = 0
        for line in self.line_indexes:
            if jig in self.line_orders[line]:
                find_line = line
                order = self.line_orders[line].index(jig)
                
        if (order == len(self.line_orders[find_line]) - 1):
            generic.append(self.set_line_done(find_line))
            destination = generic + [self.get_next_jig_to_deliver(other_jig)]
            destinations.append(destination)
        else:
            next_order = self.line_orders[find_line][order + 1]
            destination = generic + [self.get_next_jig_to_deliver(next_order),self.set_line_next(find_line, order)]
            destinations.append(destination)
            if other_jig != -1:
                destination = generic + [self.get_next_jig_to_deliver(other_jig),  self.set_line_next(find_line, order)]
                destinations.append(destination)
        return destinations
    
    def move_trailer_to_rack(self, trailer: int, rack: int, side: str):
        if side == "front":
            destinations = [self.set_trailer_front_of_rack(trailer, rack)]
        else:
            destinations = [self.set_trailer_back_of_rack(trailer, rack)]
        return destinations
    
    def move_hangar_to_rack(self, trailer: int, hangar: int, rack: int, side: str):
        return [self.set_hangar_empty(hangar)] + self.move_trailer_to_rack(trailer, rack, side)
    
    def move_hangar_to_storage(self, trailer: int, hangar: int):
        return [self.set_hangar_empty(hangar), self.set_trailer_front_of_storage(trailer)]
    
    def move_trailer_to_storage(self, trailer: int):
        return [self.set_trailer_front_of_storage(trailer)]
    
    def move_trailer_to_truck(self, trailer: int):
        return [self.set_trailer_front_of_truck(trailer), self.set_truck_not_empty()]
    
    def move_trailer_from_truck(self, trailer: int, rack: int, side: str):
        return self.move_trailer_to_rack(trailer, rack, side) + [self.set_truck_empty()]
    
    def move_trailer_to_beluga(self, trailer: int, beluga: int):
        return [self.set_trailer_front_of_beluga(trailer, beluga)]


    # model generation #################################################################################################
    def generate_rack_load_variables(self, rack: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.rack_loads[rack], 0, self.rack_max_capacity, self.initial_state[self.rack_loads[rack]])
    
    def generate_num_jigs_on_rack_variables(self, rack: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.num_jigs_on_rack[rack], 0, self.num_jigs, self.initial_state[self.num_jigs_on_rack[rack]])
    
    def generate_front_rack_variables(self, rack: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.front_rack_vars[rack], -1, self.num_jigs - 1, self.initial_state[self.front_rack_vars[rack]])
    
    def generate_back_rack_variables(self, rack: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.back_rack_vars[rack], -1, self.num_jigs - 1, self.initial_state[self.back_rack_vars[rack]])
    
    def generate_jig_variables(self, jig: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.jig_vars[jig], 0, self.jig_position_bounds[1], self.initial_state[self.jig_vars[jig]])
    
    def generate_front_jig_variables(self, jig: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.front_jig_vars[jig], -1, self.num_jigs - 1, self.initial_state[self.front_jig_vars[jig]])
    
    def generate_back_jig_variables(self, jig: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.back_jig_vars[jig], -1, self.num_jigs - 1, self.initial_state[self.back_jig_vars[jig]])
    
    def generate_trailer_variables(self, trailer: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.trailer_vars[trailer], self.trailer_position_bounds[0], self.trailer_position_bounds[1], self.initial_state[self.trailer_vars[trailer]])
    
    def generate_empty_trailer_variables(self, trailer: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.empty_trailer_vars[trailer], 0, 1, self.initial_state[self.empty_trailer_vars[trailer]])
    
    def generate_hangar_variables(self, hangar: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.hangar_vars[hangar], 0, 1, self.initial_state[self.hangar_vars[hangar]])
    
    def generate_beluga_variables(self, beluga: int): 
        return JaniStructureGenerator.generate_bounded_int_variable(self.beluga_vars[beluga], 0, self.num_jigs, self.initial_state[self.beluga_vars[beluga]])
    
    def generate_beluga_phase_variables(self, beluga: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.beluga_phases[beluga], 0, 1, self.initial_state[self.beluga_phases[beluga]])
    
    def generate_factory_delivered(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.next_jig, self.line_bounds[0], self.line_bounds[1], self.initial_state[self.next_jig])
    
    def generate_line_variables(self, line: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.line_vars[line], self.line_bounds[0], self.line_bounds[1], self.initial_state[self.line_vars[line]])
    
    def generate_truck(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.truck, 0, 1, self.initial_state[self.truck])
    
    def generate_num_swaps(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.num_swaps, 0, self.max_num_swaps, self.initial_state[self.num_swaps])
    
    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.failing_PROB_NAME, JaniStructureGenerator.generate_real_type(), self.failing_prob)]
 

    def generate_variables(self):
        variables = []        
        # self.jig_variables = list(self.jig_vars.values()) + list(self.front_jig_vars.values()) + list(self.back_jig_vars.values())
        variables += [self.generate_jig_variables(jig) for jig in self.jig_indexes]
        variables += [self.generate_front_jig_variables(jig) for jig in self.jig_indexes]
        variables += [self.generate_back_jig_variables(jig) for jig in self.jig_indexes]
        
        # self.rack_variables = list(self.rack_loads.values()) + list(self.front_rack_vars.values()) + list(self.back_rack_vars.values())
        variables += [self.generate_rack_load_variables(rack) for rack in self.rack_indexes]
        variables += [self.generate_front_rack_variables(rack) for rack in self.rack_indexes]
        variables += [self.generate_back_rack_variables(rack) for rack in self.rack_indexes]

        # self.trailer_variables = list(self.trailer_vars.values()) +  list(self.empty_trailer_vars.values())
        variables += [self.generate_trailer_variables(trailer) for trailer in self.trailer_indexes] if self.move_aware else []
        variables += [self.generate_empty_trailer_variables(trailer) for trailer in self.trailer_indexes]
        
# list(self.hangar_vars.values()) + list(self.beluga_vars.values()) + list(self.beluga_phases.values()) + [self.num_factory_delivered] + [self.truck] 
        variables += [self.generate_hangar_variables(hangar) for hangar in self.hangar_indexes] if self.move_aware else []
        variables += [self.generate_beluga_variables(beluga) for beluga in self.beluga_indexes]
        variables += [self.generate_beluga_phase_variables(beluga) for beluga in self.beluga_indexes] if self.load_aware else []
        variables += [self.generate_factory_delivered()]
        variables += [self.generate_line_variables(line) for line in self.line_indexes]
        variables += [self.generate_truck()]
        variables += [self.generate_num_swaps()]
        return variables

    #

    def generate_edge_aux(self, destinations: list, action: str, guard) -> json:
        return JaniModelGeneratorPddlInJani.generate_edge_aux(destinations, action, guard)

    def generate_push_rack_edges(self):
        
        def generate_push_front(rack: int): 
            edges = []
            action = self.action_labels_push_rack[rack] if self.move_aware else self.action_labels_push_front_rack[rack]
            for trailer in self.trailer_indexes:
                for jig in self.jig_indexes: 
                    for front_jig in ([-1] + self.jig_indexes):
                        if jig == front_jig:
                            continue
                        guard = [self.jig_on_trailer(jig, trailer), self.jig_smaller_than_rack(jig, rack), self.jig_front_of_rack(front_jig, rack), self.swaps_possible()]
                        guard += [self.trailer_front_of_rack(trailer, rack)] if self.move_aware else []
                        d1 = self.generate_destination_aux(self.push_front_rack(rack, trailer, jig, front_jig) + [self.set_next_jig(0)], self.failing_prob)
                        d2 = self.generate_destination_aux(self.push_front_rack(rack, trailer, jig, front_jig) + [self.set_next_jig(1)], 1 - self.failing_prob)
                        # destinations = [self.generate_destination_aux(self.push_front_rack(rack, trailer, jig, front_jig) + [self.set_next_jig(0), self.failing_prob)]
                        edges += [self.generate_edge_aux([d1, d2], action, Je.And(guard + [self.line_not_done(0), self.line_not_done(1)]))]
                        d1 = self.generate_destination_aux(self.push_front_rack(rack, trailer, jig, front_jig) + [self.set_next_jig(0)], 1)
                        d2 = self.generate_destination_aux(self.push_front_rack(rack, trailer, jig, front_jig) + [self.set_next_jig(1)], 1)
                        edges += [self.generate_edge_aux([d2], action, Je.And(guard + [self.line_done(0)]))]
                        edges += [self.generate_edge_aux([d1], action, Je.And(guard + [self.line_done(1)]))]
            return edges 
        
        def generate_push_back(rack: int): 
            edges = []
            action = self.action_labels_push_rack[rack] if self.move_aware else self.action_labels_push_back_rack[rack]
            for trailer in self.trailer_indexes:
                for jig in self.jig_indexes: 
                    for back_jig in ([-1] + self.jig_indexes):
                        if jig == back_jig: 
                            continue
                        guard = [self.jig_on_trailer(jig, trailer), self.jig_smaller_than_rack(jig, rack), self.jig_back_of_rack(back_jig, rack), self.swaps_possible()]
                        guard += [self.trailer_front_of_rack(trailer, rack)] if self.move_aware else []
                        d1 = self.generate_destination_aux(self.push_back_rack(rack, trailer, jig, back_jig) + [self.set_next_jig(0)], self.failing_prob)
                        d2 = self.generate_destination_aux(self.push_back_rack(rack, trailer, jig, back_jig) + [self.set_next_jig(1)], 1 - self.failing_prob)
                        # destinations = [self.generate_destination_aux(self.push_back_rack(rack, trailer, jig, back_jig))]
                        edges += [self.generate_edge_aux([d1, d2], action, Je.And(guard + [self.line_not_done(0), self.line_not_done(1)]))]
                        d1 = self.generate_destination_aux(self.push_back_rack(rack, trailer, jig, back_jig) + [self.set_next_jig(0)], 1)
                        d2 = self.generate_destination_aux(self.push_back_rack(rack, trailer, jig, back_jig) + [self.set_next_jig(1)], 1)
                        edges += [self.generate_edge_aux([d2], action, Je.And(guard + [self.line_done(0)]))]
                        edges += [self.generate_edge_aux([d1], action, Je.And(guard + [self.line_done(1)]))]
            return edges 
        final_edges = []   
        for rack in range(0, self.num_racks):
            final_edges += generate_push_front(rack)
            final_edges += generate_push_back(rack)
        return final_edges
    
    def generate_pop_rack_edges(self):
        def generate_pop_back_rack_edge(rack: int):
            edges = []
            action = self.action_labels_pop_rack[rack] if self.move_aware else self.action_labels_pop_back_rack[rack]
            for jig in self.jig_indexes:
                for front_jig in ([-1] + self.jig_indexes):
                    if front_jig == jig:
                        continue
                    guard = [self.jig_on_rack(jig, rack), self.jig_back_of_rack(jig, rack), self.jig_front_of_jig(jig, front_jig), self.is_empty_truck()]
                    # d1 = self.generate_destination_aux(self.pop_back_rack(rack, jig, front_jig) + [self.set_next_jig(0)], self.failing_prob)
                    # d2 = self.generate_destination_aux(self.pop_back_rack(rack, jig, front_jig) + [self.set_next_jig(1)], 1 - self.failing_prob)
                    destinations = [self.generate_destination_aux(self.pop_back_rack(rack, jig, front_jig))]
                    edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        def generate_pop_front_rack_edge(rack: int):
            edges = []
            action = self.action_labels_pop_rack[rack] if self.move_aware else self.action_labels_pop_front_rack[rack]
            for jig in self.jig_indexes:
                for back_jig in ([-1] + self.jig_indexes):
                    if back_jig == jig:
                        continue
                    guard = [self.jig_on_rack(jig, rack), self.jig_front_of_rack(jig, rack), self.jig_back_of_jig(jig, back_jig), self.is_empty_truck()]
                    # d1 = self.generate_destination_aux(self.pop_front_rack(rack, jig, back_jig) + [self.set_next_jig(0)], self.failing_prob)
                    # d2 = self.generate_destination_aux(self.pop_front_rack(rack, jig, back_jig) + [self.set_next_jig(1)], 1 - self.failing_prob)
                    destinations = [self.generate_destination_aux(self.pop_front_rack(rack, jig, back_jig))]
                    edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
                
        final_edges = []   
        for rack in self.rack_indexes:
            final_edges += generate_pop_back_rack_edge(rack)
            final_edges += generate_pop_front_rack_edge(rack)
        return final_edges
    
    def generate_rack_edges(self):
        return self.generate_push_rack_edges() + self.generate_pop_rack_edges()

    def generate_push_truck_trailer_edges(self):
        def generate_push_truck_trailer_edge(trailer: int):
            edges = []
            action = self.action_labels_push_truck_trailer[trailer]
            for jig in self.jig_indexes:
                guard = [self.jig_on_truck(jig), self.is_empty_trailer(trailer)]
                d1 = self.generate_destination_aux(self.push_truck_trailer(jig, trailer) + [self.set_next_jig(0)], self.failing_prob)
                d2 = self.generate_destination_aux(self.push_truck_trailer(jig, trailer) + [self.set_next_jig(1)], 1 - self.failing_prob)
                # destinations = [self.generate_destination_aux(self.push_truck_trailer(jig, trailer))]
                edges += [self.generate_edge_aux([d1, d2], action, Je.And(guard + [self.line_not_done(0), self.line_not_done(1)]))]
                d1 = self.generate_destination_aux(self.push_truck_trailer(jig, trailer) + [self.set_next_jig(0)], 1)
                d2 = self.generate_destination_aux(self.push_truck_trailer(jig, trailer) + [self.set_next_jig(1)], 1)
                edges += [self.generate_edge_aux([d2], action, Je.And(guard + [self.line_done(0)]))]
                edges += [self.generate_edge_aux([d1], action, Je.And(guard + [self.line_done(1)]))]                
            return edges
        final_edges = []
        for trailer in self.trailer_indexes:
            final_edges += generate_push_truck_trailer_edge(trailer)
        return final_edges

    def generate_move_hangar_edge(self):
        def move_hangar_rack_edge(hangar: int, rack: int, side: str):
            edges = []
            action = self.action_labels_move_hangar_rack[(hangar, side, rack)]
            for trailer in self.trailer_indexes:
                if side == "front":
                    guard = [self.trailer_on_hangar(trailer, hangar), self.rack_front_clear(rack)]
                else:
                    guard = [self.trailer_on_hangar(trailer, hangar), self.rack_back_clear(rack)]
                destinations = [self.generate_destination_aux(self.move_hangar_to_rack(trailer, hangar, rack, side))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges
        
        def move_hangar_storage_edge(hangar: int):
            edges = []
            action = self.action_labels_move_hangar_storage[hangar]
            for trailer in self.trailer_indexes:
                guard = [self.trailer_on_hangar(trailer, hangar)]
                destinations = [self.generate_destination_aux(self.move_hangar_to_storage(trailer, hangar))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges
        
        final_edges = []
        for hangar in self.hangar_indexes:
            final_edges += move_hangar_storage_edge(hangar)
            for rack in self.rack_indexes:
                final_edges += move_hangar_rack_edge(hangar, rack, "front")
                final_edges += move_hangar_rack_edge(hangar, rack, "back")
        return final_edges
    
    def generate_move_from_racks_edges(self):
        def move_rack_truck_edge(rack: int, side: int):
            action = self.action_labels_move_rack_truck[(rack, side)]
            edges = []
            for trailer in self.trailer_indexes:
                guard = [self.is_empty_truck()]
                if side == "front":
                    guard += [self.trailer_front_of_rack(trailer, rack)]
                else:
                    guard += [self.trailer_back_of_rack(trailer, rack)]
                destinations = [self.generate_destination_aux(self.move_trailer_to_truck(trailer))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        def move_truck_rack_edge(rack: int, side: int):
            action = self.action_labels_move_truck_rack[(rack, side)]
            edges = []
            for trailer in self.trailer_indexes:
                guard = [self.trailer_on_truck(trailer)]
                if side == "front":
                    guard += [self.rack_front_clear(rack)]
                if side == "back":
                    guard += [self.rack_back_clear(rack)]
                destinations = [self.generate_destination_aux(self.move_trailer_from_truck(trailer, rack, side))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges
                
        def move_rack_rack_edge(rack_i : int, side_i : str, rack_f : int, side_f : str):
            action = self.action_labels_move_rack_rack[(rack_i, side_i, rack_f, side_f)]
            edges = []
            for trailer in self.trailer_indexes:
                if side_f == "front":
                    guard = [self.rack_front_clear(rack_f)]
                    destinations = [self.generate_destination_aux(self.set_trailer_front_of_rack(trailer, rack_f))]
                else:
                    guard = [self.rack_back_clear(rack_f)]
                    destinations = [self.generate_destination_aux(self.set_trailer_back_of_rack(trailer, rack_f))]
                if side_i == "front":
                    guard += self.trailer_front_of_rack(trailer, rack_i)
                else:
                    guard += self.trailer_back_of_rack(trailer, rack_i)
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        def move_rack_storage_edge(rack: int, side: str):
            action = self.action_labels_move_rack_storage[(rack, side)]
            edges = []
            for trailer in self.trailer_indexes:
                if side == "front":
                    guard = [self.trailer_front_of_rack(trailer, rack)]
                else:
                    guard = [self.trailer_back_of_rack(trailer, rack)]
                destinations = [self.generate_destination_aux(self.move_trailer_to_storage(trailer))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        final_edges = []
        for rack in self.rack_indexes:
            for side in ["front", "back"]:
                final_edges += move_rack_storage_edge(rack, side)
                final_edges += move_rack_truck_edge(rack, side)
                final_edges += move_truck_rack_edge(rack, side)
        return final_edges
    
    def generate_move_edges(self):  
        def move_storage_rack_edge(rack: int, side: str):
            edges = []
            action = self.action_labels_move_storage_rack[(rack, side)]
            for trailer in self.trailer_indexes:
                guard = [self.trailer_in_storage(trailer)]
                if side == "front":
                    guard += [self.rack_front_clear(rack)]
                    destinations = [self.generate_destination_aux(self.move_trailer_to_rack(trailer, rack, side))]
                else:
                    guard += [self.rack_back_clear(rack)]
                    destinations = [self.generate_destination_aux(self.move_trailer_to_rack(trailer, rack, side))]
                edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges
        
        def move_trailer_beluga_edge(trailer : int, beluga: int):
            action = self.action_labels_move_trailer_to_beluga[(trailer, beluga)]
            guard = [self.trailer_not_front_of_beluga(trailer, beluga)]
            destinations = [self.generate_destination_aux(self.move_trailer_to_beluga(trailer, beluga))]
            return self.generate_edge_aux(destinations, action, Je.And(guard))
        
        final_edges = self.generate_move_hangar_edge()
        final_edges += self.generate_move_from_racks_edges()
        for rack in self.rack_indexes:
            for side in ["front", "back"]:
                final_edges += move_storage_rack_edge(rack, side)
        final_edges += [move_trailer_beluga_edge(trailer, beluga) for trailer in self.trailer_indexes for beluga in self.beluga_indexes]
        return final_edges
        
    def generate_jig_edges(self):
        def generate_beluga_load_edges(beluga: int):
            edges = []
            action = self.action_labels_beluga_operations[beluga]
            for jig in self.jig_indexes:
                for trailer in self.trailer_indexes:
                    guard = [self.beluga_in_loading_phase(beluga), self.jig_on_trailer(jig, trailer), self.trailer_front_of_beluga(trailer, beluga)]
                    destinations = [self.generate_destination_aux(self.load_beluga(beluga, trailer, jig))]
                    edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        def generate_beluga_unload_edges(beluga: int, last: bool):
            edges = []
            action = self.action_labels_beluga_operations[beluga]
            for jig in self.jig_indexes:
                for trailer in self.trailer_indexes:
                    guard = [self.is_empty_trailer(trailer), self.next_unload_jig_on_beluga(jig, beluga)]
                    guard += [self.trailer_front_of_beluga(trailer, beluga), self.jig_smaller_than_trailer(jig, trailer)] if self.move_aware else []
                    guard += [self.beluga_in_unloading_phase(beluga)] if self.load_aware else []
                    if last:
                        guard += [self.beluga_has_one_jig(beluga)]
                    destinations = [self.generate_destination_aux(self.unload_beluga(beluga, trailer, jig, last))]
                    edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]
            return edges 
        
        def generate_send_factory_edges():
            edges = []
            action = self.action_labels_send_to_factory
            for line in self.line_indexes:
                for jig in self.line_orders[line]:
                    other_line = 1 - line
                    for other_jig in ([-1] + self.line_orders[other_line]):
                        guard = [self.is_jig_next_for_factory(jig), self.jig_on_truck(jig), self.is_line_next_jig(other_line, other_jig)]
                        updates = self.send_factory(jig, other_jig)
                        if len(updates) == 2:
                            d1 = self.generate_destination_aux(updates[0], (self.failing_prob))
                            d2 = self.generate_destination_aux(updates[1], (1 - self.failing_prob))
                            destinations = [d1, d2]
                        else:
                            destinations = [self.generate_destination_aux(updates[0])]
                        edges += [self.generate_edge_aux(destinations, action, Je.And(guard))]                    
            return edges
        
        final_edges = generate_send_factory_edges()
        for beluga in self.beluga_indexes:
            final_edges += (generate_beluga_load_edges(beluga) if self.load_aware else [])
            final_edges += (generate_beluga_unload_edges(beluga, True) if self.load_aware else [])
            final_edges += generate_beluga_unload_edges(beluga, False)
        return final_edges
    
    def isJsonString(self, s):
        try:
            json.loads(s)
        except:
            assert(False)
        return
    
    def generate_edges(self):
        edges = self.generate_jig_edges() + (self.generate_move_edges() if self.move_aware else []) + self.generate_rack_edges() + self.generate_push_truck_trailer_edges()
        return edges

    # property generation ##############################################################################################
    def rack_exceeded_capacity(self, rack: int):
        return Je.Le(self.rack_loads[rack], 9)
    
    def did_not_exceed_capacity(self):
        return Je.Ge(self.rack_loads[self.num_racks - 1], 10)
    
    def swaps_exhausted(self):
        return Je.Ge(self.num_swaps, self.max_num_swaps)
    
    def is_safe(self):
        return self.did_not_exceed_capacity() if not self.swap_unsafe else self.swaps_possible()
    
    def is_unsafe(self) -> json:
        return self.rack_exceeded_capacity(self.num_racks - 1) if not self.swap_unsafe else self.swaps_exhausted()
    
    def generate_reachable_conditions(self) -> json:
        constraints = []
        # If jig not on rack, then front of rack and back of rack cannot be jig
        for rack in self.rack_indexes:
            for jig in self.jig_indexes:
                constraints.append(Je.Implies(self.jig_not_on_rack(jig, rack), Je.And(self.jig_not_front_of_rack(jig, rack), self.jig_not_back_of_rack(jig, rack))))
        # If front of rack or back of rack is -1 then all jigs not on rack
        for rack in self.rack_indexes:
            rack_empty = Je.Or(self.rack_front_empty(rack), self.rack_back_empty(rack))
            no_jigs = Je.And([self.jig_not_on_rack(jig, rack) for jig in self.jig_indexes])
            constraints.append(Je.Implies(rack_empty, no_jigs))
        # Beluga should be unloadable if jig has a position there.
        for position in range(0, self.num_jigs):
            constraints.append(Je.Iff(self.atleast_one_jig_at_position(position), self.jigs_to_unload(0, position + 1)))
        # Jigs cannot have same position unless delivered or on same rack
        for j1 in self.jig_indexes:
            for j2 in self.jig_indexes:
                if j1 >= j2: 
                    continue
                constraints.append(Je.Or(self.is_jig_factory_or_rack(j1), Je.Ne(self.jig_vars[j1], self.jig_vars[j2])))
        constraints.append(self.all_jigs_delivered())
        return Je.And(constraints)
    
    def generate_objective(self) -> json:
        if not self.reachable_objective:
            goal = JaniStructureGenerator.generate_state_condition_expression([], Je.And(self.generate_goal_expression()))
        else:
            goal = JaniStructureGenerator.generate_state_condition_expression([], self.generate_reachable_conditions())
        return JaniStructureGenerator.generate_objective_expression(goal=goal, goal_potential=self.generate_goal_potential())

    def generate_goal_potential(self) -> json:
        if not self.use_goal_potential:
            return None
        return None

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.is_unsafe())

    # starts ###########################################################################################################

    def generate_start(self):
        constraints = list()

        # safe at start!!!
        constraints.append(self.is_safe())
        if self.swap_unsafe:
            constraints += [Je.Eq(self.num_swaps, 0)]
        # full beluga at start
        constraints += [Je.Eq(self.beluga_vars[beluga], self.num_jigs if beluga == 0 else 0) for beluga in self.beluga_indexes]
        constraints += [Je.Eq(self.rack_loads[rack], self.rack_max_capacity) for rack in self.rack_indexes]
        constraints += [Je.Eq(self.hangar_vars[hangar], 0) for hangar in self.hangar_indexes] if self.move_aware else []
        constraints += [Je.Eq(self.line_vars[line], self.line_orders[line][0]) for line in self.line_indexes]
        constraints += [Je.Eq(self.next_jig, self.line_orders[1][0])]
        constraints += [Je.Eq(self.empty_trailer_vars[0], 1)]
        constraints += [Je.Eq(self.truck, 1)]

        # jigs must be ordered in beluga anyway we want. no two jigs can have same position.
        for jig in self.jig_indexes:
            constraints.append(self.jig_on_beluga(jig, 0))
        
        for rack in self.rack_indexes:
            constraints.append(Je.Eq(self.front_rack_vars[rack], -1))
            constraints.append(Je.Eq(self.back_rack_vars[rack], -1))

        for j1 in self.jig_indexes:
            constraints.append(Je.Eq(self.front_jig_vars[j1], -1))
            constraints.append(Je.Eq(self.back_jig_vars[j1], -1))
            for j2 in self.jig_indexes:
                if j1 >= j2: 
                    continue
                constraints.append(Je.Ne(self.jig_vars[j1], self.jig_vars[j2]))
        # trailers can be in any position (no overlap except storage)
        if not self.move_aware:
            return JaniStructureGenerator.generate_large_conjunction(constraints)
 
        return JaniStructureGenerator.generate_large_conjunction(constraints)

    def generate_random_states(self, number_starts: int) -> list:

        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=number_starts, default_state=self.initial_state)

        random_assmts = list(self.trailer_vars.values()) + list(self.jig_vars.values())
        for i in range(0, number_starts):
            candidate_mapping = dict([(var, 0) if var in random_assmts else (var, self.initial_state[var]) for var in self.initial_state.keys()])
            jigs_permutation = list(self.jig_vars.values())
            random.shuffle(jigs_permutation)
            for j in jigs_permutation:
                candidate_mapping[j] = jigs_permutation.index(j)
                
            if self.move_aware: # assign trailer positions if move operations exist
                trailer_positions = list(range(self.trailer_position_bounds[0], self.trailer_position_bounds[1] + 1))
                for trailer in self.trailer_indexes:
                    position = random.choice(trailer_positions)
                    candidate_mapping[self.trailer_vars[trailer]] = position
                    if position != self.trailer_assignment(0, "storage"):
                        trailer_positions.remove(position)
                        
            rlt = states_values.add(candidate_mapping)
            if rlt is not None:
                return rlt

        return states_values.generate_states_values()
    # predicate generation #############################################################################################

    def generate_stepwise_splits(self) -> list:
            return JaniModelGenerator.generate_stepwise_splits([self.jig_position_bounds[1], self.trailer_position_bounds[1], self.rack_max_capacity])


    # nn input #########################################################################################################

    def get_nn_inputs(self) -> list:
        nn_vars = list(self.variable_names)
        if not self.swap_unsafe:
            nn_vars.remove(self.num_swaps)
        return nn_vars

    ####################################################################################################################


if __name__ == "__main__":
    args = BelugaGenerationOptionParser().arg_parse()
    generator = BelugaGenerator(args)
    generator.generate()
