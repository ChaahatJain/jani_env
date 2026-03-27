#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import json
import math
import random

from jani_generation.jani_model_generator_pddl_in_jani import JaniModelGeneratorPddlInJani, ModelCostGenerator
from jani_generation.jani_model_generator import BoundedVariable, VarSplitSpec, JaniModelGenerationOptionParser, JaniModelGenerator
from jani_generation.jani_structure_generator import Je, JaniStructureGenerator
from python_utils import PythonUtils

random.seed(2020)


class BlocksWorldModelGenerationOptionParser(JaniModelGenerationOptionParser):
    def __init__(self):
        JaniModelGenerationOptionParser.__init__(self)

    def add_options(self):
        JaniModelGenerationOptionParser.add_options(self)
        ModelCostGenerator.add_options(self.optionParser)
        self.optionParser.add_argument("--num-blocks", type=int, default=None, help="The number of blocks.")
        self.optionParser.add_argument("--table-limit", type=int, default=None, help="Maximal number of blocks on the table.")
        self.optionParser.add_argument("--use-hand-empty-flag", action="store_true", default=False, help="Use explicit hand empty flags.")
        self.optionParser.add_argument("--use-clear-flags", action="store_true", default=False, help="Use explicit clear flags.")
        self.optionParser.add_argument("--use-height", action="store_true", default=False, help="Use height variables.")
        self.optionParser.add_argument("--use-table-counter", action="store_true", default=False, help="Use explicit table counter.")
        # options for property generation (and thus not saved in model file):
        self.optionParser.add_argument("--hand-empty-at-start", action="store_true", default=False, help="Hand is empty at start.")
        self.optionParser.add_argument("--ordered-index-start", action="store_true", default=False, help="Block stacks are ordered according to index at start.")
        self.optionParser.add_argument("--use-time", action="store_true", default=False, help="Use time variable.")

class BlocksWorldModelGenerator(JaniModelGeneratorPddlInJani):
    # model characteristics:
    LEGACY_DOM_ENC = True  # use block domain encoding for icaps22, rddps22
    # Flags (optional)
    HAND_EMPTY = 1
    NOT_HAND_EMPTY = 0
    CLEAR = 1
    NOT_CLEAR = 0
    #
    UNDEFINED = -1  # for sample-in-model
    # constants:
    NUMBER_OF_BLOCKS_NAME = "number_of_blocks"
    TABLE_LIMIT_NAME = "table_limit"
    USE_HAND_EMPTY_FLAG = "use-hand-empty-flag"
    USE_CLEAR_FLAGS = "use-clear-flags"
    USE_HEIGHT = "use-height"
    USE_TABLE_COUNTER = "use-table-counter"
    USE_TIME = "use-time"
    TIME_UB = 31
    #
    SUPPRESS_CYCLES = True

    def __init__(self, options):
        JaniModelGeneratorPddlInJani.__init__(self, options)
        self.cost_generator = ModelCostGenerator(self)
        self.cost_generator.set_cost_preds_afterwards("cost_predicates_afterwards" in self.property_type)
        self.cost_generator.set_max_cost_splits(15)
        self.cost_generator.load_options(options)

        if self.model_file is None:
            # model generation
            self.num_blocks = options.num_blocks
            self.table_limit = options.table_limit
            self.use_hand_empty_flag = options.use_hand_empty_flag
            self.use_clear_flags = options.use_clear_flags
            self.use_height = options.use_height
            self.use_table_counter = options.use_table_counter
            self.use_time = options.use_time
        else:
            # property generation
            model = PythonUtils.load_json(self.model_file)
            constants = model["constants"]
            self.cost_generator.read_constants(constants)
            _, self.num_blocks = self.read_constant(constants, self.NUMBER_OF_BLOCKS_NAME)
            _, self.table_limit = self.read_constant(constants, self.TABLE_LIMIT_NAME)
            _, self.use_hand_empty_flag = self.read_constant(constants, self.USE_HAND_EMPTY_FLAG)
            _, self.use_clear_flags = self.read_constant(constants, self.USE_CLEAR_FLAGS)
            _, self.use_height = self.read_constant(constants, self.USE_HEIGHT)
            _, self.use_table_counter = self.read_constant(constants, self.USE_TABLE_COUNTER)
            _, self.use_time = self.read_constant(constants, self.USE_TIME)
            #
        self.hand_empty_at_start = options.hand_empty_at_start
        self.ordered_index_start = options.ordered_index_start

        self.model_type = self.cost_generator.get_model_type()
        self.model_name = str(self.num_blocks) + "-blocks"

        self.blocks = list(range(0, self.num_blocks)) if self.LEGACY_DOM_ENC else list(range(1, self.num_blocks + 1))  # "0" corresponds to "table"
        self.TERMINAL_AT_UNSAFE_NAME = "terminal-at-unsafe"
        # variables:
        self.block_vars = dict([(block, "block_" + str(block)) for block in self.blocks])
        self.hand_empty_flag = "hand-empty" if self.use_hand_empty_flag else None
        self.clear_flags = dict([(block, "clear_" + str(block)) for block in self.blocks]) if self.use_clear_flags else None
        self.block_heights = dict([(block, "height_" + str(block)) for block in self.blocks]) if self.use_height else None
        self.table_counter = "table-counter" if self.use_table_counter else None
        self.time = "time" if self.use_time else None
        #
        self.variable_names = list(self.block_vars.values())
        self.variable_names = ([self.hand_empty_flag] if self.use_hand_empty_flag else []) + self.variable_names  # to preserve old order in NN file
        self.variable_names += list(self.clear_flags.values()) if self.use_clear_flags else []
        self.variable_names += list(self.block_heights.values()) if self.use_height else []
        if self.use_table_counter:
            if self.use_hand_empty_flag:
                self.variable_names = self.variable_names[0:1] + [self.table_counter] + self.variable_names[1:]  # hack to preserve old order in NN file
            else:
                self.variable_names += [self.table_counter] if self.use_table_counter else []

        # use cost
        if self.cost_generator.do_cost():
            self.cost_generator.set_cost_vars(list(self.blocks), "block")
            self.cost_generator.set_variable_names(self.variable_names)

        # auxiliary flags (used as free variables in let)
        self.variable_names += [self.time] if self.use_time else []

        if not self.ordered_index_start and not self.use_height:
            # relative position flags
            self.free_height_vars = dict([(block, "height_" + str(block)) for block in self.blocks])

        # actions
        self.action_label_table = "choose_table"
        self.action_labels_block = dict([(block, "choose_block_" + str(block)) for block in self.blocks])
        # meaning of "action_label_item": if hand not empty, then put block in hand on chosen item, if hand empty, then pick up chosen item(block)
        self.action_labels = [self.action_label_table] + list(self.action_labels_block.values())
        self.compute_block_domains()
        self.compute_model_initial_and_goal_state()

    # noinspection PyAttributeOutsideInit
    def compute_block_domains(self):
        self.block_domain_size = 2 + (self.num_blocks - 1)  # ON_TABLE, n - 1 block, IN_HAND
        self.table_domain_size = self.num_blocks + 1  # up to n blocks or no block (invalid)
        self.height_domain_size = self.num_blocks + 1  # ON_TABLE, n - 1 block, IN_HAND
        self.block_domains = dict()
        self.block_domains_reverse = dict()
        for block in self.blocks:
            self.block_domains[block] = dict()
            self.block_domains_reverse[block] = dict()
            if self.LEGACY_DOM_ENC:
                value = 2  # values 0,1 for IN_HAND, ON_TABLE
                for block_alt in self.blocks:
                    if block != block_alt:  # skipping self value
                        self.block_domains[block][block_alt] = value
                        self.block_domains_reverse[block][value] = block_alt
                        value += 1
            else:
                for block_alt in self.blocks:
                    if block != block_alt:  # skipping self value
                        self.block_domains[block][block_alt] = block_alt
                        self.block_domains_reverse[block][block_alt] = block_alt
        #
        self.on_table_values = dict([(block, 1 if self.LEGACY_DOM_ENC else 0) for block in self.blocks])
        self.in_hand_values = dict([(block, 0 if self.LEGACY_DOM_ENC else block) for block in self.blocks])

    # noinspection PyAttributeOutsideInit
    def compute_model_initial_and_goal_state(self):
        # start
        self.initial_state = dict()

        self.cost_generator.set_initial_cost(self.initial_state)  # set cost to 0 if present
        if self.use_time:
            self.initial_state[self.time] = 0
        if self.use_hand_empty_flag:
            self.initial_state[self.hand_empty_flag] = self.HAND_EMPTY  # hand is empty

        if self.sample_in_model:

            for block in self.blocks:
                self.initial_state[self.block_vars[block]] = self.UNDEFINED
                self.initial_state[self.clear_flags[block]] = self.CLEAR
                if self.use_height:
                    self.initial_state[self.block_heights[block]] = 0
                if self.use_table_counter:
                    self.initial_state[self.table_counter] = 0

        else:

            for block, block_alt in zip(self.blocks[:-1], self.blocks[1:]):
                self.initial_state[self.block_vars[block]] = self.block_domains[block][block_alt]  # sort blocks in ASC order
                if self.use_clear_flags:
                    self.initial_state[self.clear_flags[block_alt]] = self.NOT_CLEAR

            self.initial_state[self.block_vars[self.blocks[-1]]] = self.on_table_values[self.blocks[-1]]
            if self.use_clear_flags:
                self.initial_state[self.clear_flags[self.blocks[0]]] = self.CLEAR

            if self.use_height:
                height = self.num_blocks - 1
                for block in self.blocks:
                    self.initial_state[self.block_heights[block]] = height
                    height -= 1

            if self.use_table_counter:
                self.initial_state[self.table_counter] = 1  # exactly one block on table

        # goal
        self.goal_state = dict()
        for block, block_alt in zip(reversed(self.blocks[1:]), reversed(self.blocks[:-1])):
            self.goal_state[self.block_vars[block]] = self.block_domains[block][block_alt]  # sort blocks in DEC order
        self.goal_state[self.block_vars[self.blocks[0]]] = self.on_table_values[self.blocks[0]]

    def is_below_at_goal(self, block_below: int, block_above: int):
        goal_pos = self.goal_state[self.block_vars[block_above]]
        if goal_pos == self.on_table_values[block_above] or goal_pos == self.in_hand_values[block_above]:
            return False
        goal_block = self.block_domains_reverse[block_above][goal_pos]
        return True if goal_block == block_below else self.is_below_at_goal(block_below, goal_block)

    def is_above_at_goal(self, block_above: int, block_below: int):
        return self.is_below_at_goal(block_below=block_below, block_above=block_above)

    # auxiliary generation #############################################################################################
    
    def is_undefined(self, block: int) -> json:
        return Je.Eq(self.block_vars[block], self.UNDEFINED)

    def is_defined(self, block: int) -> json:
        return Je.Ne(self.block_vars[block], self.UNDEFINED)

    def block_on_table(self, block: int):
        return Je.Eq(self.block_vars[block], self.on_table_values[block])

    def block_not_on_table(self, block: int):
        return Je.Ne(self.block_vars[block], self.on_table_values[block])

    def block_on(self, block: int, block_alt):
        assert block != block_alt
        return Je.Eq(self.block_vars[block], self.block_domains[block][block_alt])

    def block_not_on(self, block: int, block_alt):
        assert block != block_alt
        return Je.Ne(self.block_vars[block], self.block_domains[block][block_alt])

    def block_in_hand(self, block: int):
        return Je.Eq(self.block_vars[block], self.in_hand_values[block])

    def block_not_in_hand(self, block: int):
        return Je.Ne(self.block_vars[block], self.in_hand_values[block])

    def blocks_on_table(self):
        return self.table_counter if self.use_table_counter else Je.Add([Je.to_int(self.block_on_table(block)) for block in self.blocks])

    def hand_is_empty(self):
        if self.use_hand_empty_flag:
            return Je.Ge(self.hand_empty_flag, 1)
        else:
            return Je.And([self.block_not_in_hand(block) for block in self.blocks])

    def hand_is_not_empty(self):
        if self.use_hand_empty_flag:
            return Je.Le(self.hand_empty_flag, 0)
        else:
            return Je.Or([self.block_in_hand(block) for block in self.blocks])

    def is_clear(self, block: int):
        if self.use_clear_flags:
            return Je.Ge(self.clear_flags[block], 1)
        else:
            return Je.And([self.block_not_in_hand(block)] + [self.block_not_on(block_alt, block) for block_alt in self.blocks if block_alt != block])

    def is_not_clear(self, block: int):
        if self.use_clear_flags:
            return Je.Le(self.clear_flags[block], 0)
        else:
            return Je.Or([self.block_in_hand(block)] + [self.block_on(block_alt, block) for block_alt in self.blocks if block_alt != block])

    def table_counter_is(self, value: int):
        return Je.Eq(self.blocks_on_table(), value)

    def table_counter_is_not(self, value: int):
        return Je.Ne(self.blocks_on_table(), value)

    def is_safe(self):
        return Je.Le(self.blocks_on_table(), self.table_limit) if  not self.use_time else Je.Le(self.time, 14) 

    def is_unsafe(self):
        return Je.Ge(self.blocks_on_table(), self.table_limit + 1) if  not self.use_time else Je.Ge(self.time, 15) 
    
    def has_time(self):
        return Je.Lt(self.time, self.TIME_UB)

    #

    def set_block(self, block: int, value: int):
        return JaniStructureGenerator.generate_assignment(self.block_vars[block], value)

    def put_block_on_table(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.block_vars[block], self.on_table_values[block])

    def set_block_on_block(self, block: int, block_alt: int):
        return JaniStructureGenerator.generate_assignment(self.block_vars[block], self.block_domains[block][block_alt])

    def pick_up_block(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.block_vars[block], self.in_hand_values[block])

    def set_hand_empty(self):
        return JaniStructureGenerator.generate_assignment(self.hand_empty_flag, self.HAND_EMPTY)

    def set_hand_non_empty(self):
        return JaniStructureGenerator.generate_assignment(self.hand_empty_flag, self.NOT_HAND_EMPTY)

    def set_clear(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.clear_flags[block], self.CLEAR)

    def set_not_clear(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.clear_flags[block], self.NOT_CLEAR)

    def set_table_height(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.block_heights[block], 0)

    def set_height(self, block: int, block_alt: int):
        return JaniStructureGenerator.generate_assignment(self.block_heights[block], Je.Add(self.block_heights[block_alt], 1))

    def set_hand_height(self, block: int):
        return JaniStructureGenerator.generate_assignment(self.block_heights[block], self.height_domain_size - 1)

    def inc_counter(self, inc: int = 1):
        table_counter_name = self.table_counter
        return JaniStructureGenerator.generate_assignment(table_counter_name, Je.Add(table_counter_name, inc))

    def inc_time(self, inc: int = 1):
        return JaniStructureGenerator.generate_assignment(self.time, Je.Add(self.time, inc))
    # model generation #################################################################################################

    def generate_constants(self):
        return [JaniStructureGenerator.generate_constant_declaration(self.TERMINAL_AT_UNSAFE_NAME, JaniStructureGenerator.generate_bool_type(), self.terminal_at_unsafe_jani)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.NUMBER_OF_BLOCKS_NAME, JaniStructureGenerator.generate_int_type(), self.num_blocks)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.TABLE_LIMIT_NAME, JaniStructureGenerator.generate_int_type(), self.table_limit)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.USE_HAND_EMPTY_FLAG, JaniStructureGenerator.generate_bool_type(), self.use_hand_empty_flag)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.USE_CLEAR_FLAGS, JaniStructureGenerator.generate_bool_type(), self.use_clear_flags)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.USE_HEIGHT, JaniStructureGenerator.generate_bool_type(), self.use_height)] + \
               [JaniStructureGenerator.generate_constant_declaration(self.USE_TABLE_COUNTER, JaniStructureGenerator.generate_bool_type(), self.use_table_counter)] + \
                [JaniStructureGenerator.generate_constant_declaration(self.USE_TIME, JaniStructureGenerator.generate_bool_type(), self.use_time)] + \
               self.cost_generator.generate_constants()

    def generate_block_variable(self, block: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.block_vars[block], (-1 if self.sample_in_model else 0), self.block_domain_size - 1, self.initial_state[self.block_vars[block]])

    def generate_hand_empty_flag(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.hand_empty_flag, 0, 1, self.initial_state[self.hand_empty_flag])

    def generate_clear_flag(self, block: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.clear_flags[block], 0, 1, self.initial_state[self.clear_flags[block]])

    def generate_height_variable(self, block: int):
        return JaniStructureGenerator.generate_bounded_int_variable(self.block_heights[block], 0, self.height_domain_size - 1, self.initial_state[self.block_heights[block]])

    def generate_table_counter(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.table_counter, 0, self.table_domain_size - 1, self.initial_state[self.table_counter])

    def generate_time(self):
        return JaniStructureGenerator.generate_bounded_int_variable(self.time, 0, 31, 0)

    def generate_variables(self):
        variables = [self.generate_block_variable(block) for block in self.blocks]
        variables = ([self.generate_hand_empty_flag()] if self.use_hand_empty_flag else []) + variables  # hack to preserve order of old versions
        variables += [self.generate_clear_flag(block) for block in self.blocks] if self.use_clear_flags else []
        variables += [self.generate_height_variable(block) for block in self.blocks] if self.use_height else []
        if self.use_table_counter:
            if self.use_hand_empty_flag:
                variables = variables[0:1] + [self.generate_table_counter()] + variables[1:]  # hack to preserve order of old versions
            else:
                variables += [self.generate_table_counter()] if self.use_table_counter else []
        # cost if present:
        variables += self.cost_generator.generate_variables()
        if self.use_time:
            variables += [self.generate_time()]
        return variables

    # aux flags:

    def generate_free_height_vars(self) -> list:
        return [JaniStructureGenerator.generate_free_variable_declaration(name, JaniStructureGenerator.generate_int_type()) for name in self.free_height_vars.values()]

    def generate_free_vars(self) -> list:
        return self.generate_free_height_vars() if not self.ordered_index_start and not self.use_height else list()

    #

    def generate_edges(self):
        def generate_pick_up_edge(block: int) -> list:
            action = self.action_labels_block[block]
            guard = Je.And(self.hand_is_empty(), self.is_clear(block), self.block_on_table(block))
            if self.use_time:
                guard = Je.And(self.has_time(), guard)
            if self.terminal_at_unsafe_jani:
                guard = Je.And(self.is_safe(), guard)
            assignments = list()
            assignments += [self.pick_up_block(block)]
            assignments = ([self.set_hand_non_empty()] if self.use_hand_empty_flag else []) + assignments  # hack to preserve order of old version
            assignments += [self.set_not_clear(block)] if self.use_clear_flags else []
            assignments += [self.set_hand_height(block)] if self.use_height else []
            assignments += [self.inc_counter(-1)] if self.use_table_counter else []
            assignments += [self.inc_time()] if self.use_time else []
            return self.cost_generator.append_failing_destination(block, action, guard, assignments)

        def generate_put_down_edge(block: int) -> list:
            action = self.action_label_table
            guard_list = [self.block_in_hand(block)]
            guard_list += [self.hand_is_not_empty()] if self.use_hand_empty_flag else []  # redundant but useful
            guard_list += [self.is_not_clear(block)] if self.use_clear_flags else []  # redundant but useful
            guard = Je.And(guard_list)
            if self.use_time:
                guard = Je.And(self.has_time(), guard)
            if self.terminal_at_unsafe_jani:
                guard = Je.And(self.is_safe(), guard)
            assignments = [self.put_block_on_table(block)]
            assignments = ([self.set_hand_empty()] if self.use_hand_empty_flag else []) + assignments  # hack to preserve order of old version
            assignments += [self.set_clear(block)] if self.use_clear_flags else []
            assignments += [self.set_table_height(block)] if self.use_height else []
            assignments += [self.inc_counter(1)] if self.use_table_counter else []
            assignments += [self.inc_time()] if self.use_time else []
            return self.cost_generator.append_failing_destination(block, action, guard, assignments)

        def generate_stack_edge(block: int, block_alt: int) -> list:
            assert block != block_alt
            action = self.action_labels_block[block_alt]
            guard_list = [self.is_clear(block_alt), self.block_in_hand(block)]
            guard_list += [self.hand_is_not_empty()] if self.use_hand_empty_flag else []  # redundant but useful
            guard_list += [self.is_not_clear(block)] if self.use_clear_flags else []  # redundant but useful
            guard = Je.And(guard_list)
            if self.use_time:
                guard = Je.And(self.has_time(), guard)
            if self.terminal_at_unsafe_jani:
                guard = Je.And(self.is_safe(), guard)
            assignments = [self.set_block_on_block(block, block_alt)]
            assignments = ([self.set_hand_empty()] if self.use_hand_empty_flag else []) + assignments  # hack to preserve order of old version
            assignments += [self.set_clear(block), self.set_not_clear(block_alt)] if self.use_clear_flags else []
            assignments += [self.set_height(block, block_alt)] if self.use_height else []
            assignments += [self.inc_time()] if self.use_time else []
            return self.cost_generator.append_failing_destination(block, action, guard, assignments)

        def generate_unstack_edge(block: int, block_alt: int) -> list:
            assert block != block_alt
            action = self.action_labels_block[block]
            guard_list = [self.hand_is_empty(), self.is_clear(block), self.block_on(block, block_alt)]
            guard_list += [self.is_not_clear(block_alt)] if self.use_clear_flags else []  # redundant but useful
            guard = Je.And(guard_list)
            if self.use_time:
                guard = Je.And(self.has_time(), guard)
            if self.terminal_at_unsafe_jani:
                guard = Je.And(self.is_safe(), guard)
            assignments = [self.pick_up_block(block)]
            assignments = ([self.set_hand_non_empty()] if self.use_hand_empty_flag else []) + assignments  # hack to preserve order of old version
            assignments += [self.set_not_clear(block), self.set_clear(block_alt)] if self.use_clear_flags else []
            assignments += [self.set_hand_height(block)] if self.use_height else []
            assignments += [self.inc_time()] if self.use_time else []
            return self.cost_generator.append_failing_destination(block, action, guard, assignments)

        def generate_sampling_edge() -> list:
            guard = Je.Or([self.is_undefined(block) for block in self.blocks])

            destinations = list()

            # Number of undefined blocks:
            undefined_blocks = Je.Add([Je.to_int(self.is_undefined(block)) for block in self.blocks])

            # For each block and possible value we have an destination ...
            for block in self.blocks:

                # Probability to choose this block
                choose_block_prob = Je.Div(Je.to_int(self.is_undefined(block)), undefined_blocks)

                # Can choose table or any clear block on table. This will always produce valid states.
                # Might drop "is-defined" constraint to sample invalid states as well.
                def block_is_possible_value(block__: int) -> json:
                    return Je.And(self.is_defined(block__), self.is_clear(block__))

                possible_values = Je.Add([Je.to_int(Je.Lt(self.blocks_on_table(), self.table_limit))] + [Je.to_int(block_is_possible_value(block_)) for block_ in self.blocks if block_ != block])

                # blocks:
                for block_alt in self.blocks:
                    if block_alt == block:
                        continue
                    choose_dest_block_prob = Je.Mult(choose_block_prob, Je.Div(Je.to_int(block_is_possible_value(block_alt)), possible_values))
                    assignments = [self.set_block_on_block(block, block_alt)]
                    assignments += ([self.set_not_clear(block_alt)] if self.use_clear_flags else [])
                    destinations.append(JaniStructureGenerator.generate_destination(self.SAMPLE_LOCATION, assignments=assignments, probability=choose_dest_block_prob))

                # table:
                choose_dest_table_prob = Je.Mult(choose_block_prob, Je.Div(Je.to_int(Je.Lt(self.blocks_on_table(), self.table_limit)), possible_values))
                assignments = [self.put_block_on_table(block)]
                assignments += ([self.inc_counter(1)] if self.use_table_counter else [])
                assignments += ([self.inc_time()] if self.use_time else [])
                destinations.append(JaniStructureGenerator.generate_destination(self.SAMPLE_LOCATION, assignments=assignments, probability=choose_dest_table_prob))

            return [JaniStructureGenerator.generate_edge(self.SAMPLE_LOCATION, destinations=destinations, guard=guard)]

        def generate_end_sampling_edge() -> list:
            guard = Je.And([self.is_defined(block) for block in self.blocks])
            destinations = [JaniStructureGenerator.generate_destination(location=self.LOCATION_NAME)]
            return [JaniStructureGenerator.generate_edge(location=self.SAMPLE_LOCATION, destinations=destinations, guard=guard)]

        edges = list()
        for block_ in self.blocks:
            edges += generate_pick_up_edge(block_)
            edges += generate_put_down_edge(block_)
            for block_alt_ in self.blocks:
                if block_alt_ != block_:
                    edges += generate_stack_edge(block_, block_alt_)
                    edges += generate_unstack_edge(block_, block_alt_)
        if self.sample_in_model:
            edges += generate_sampling_edge()
            edges += generate_end_sampling_edge()
        return edges

    # property generation ##############################################################################################

    def generate_objective(self) -> json:
        goal = JaniStructureGenerator.generate_state_condition_expression([], self.generate_goal_expression())
        step_reward, accumulate = self.cost_generator.generate_objective(-0.01)
        return JaniStructureGenerator.generate_objective_expression(goal=goal, goal_potential=self.generate_goal_potential(), step_reward=step_reward, accumulate=accumulate)

    def generate_goal_potential(self) -> json:
        if not self.use_goal_potential:
            return None
        additive_list = list()
        if False:  # check for sub-stack-goals
            for block in self.blocks:
                blocks_below = [block_below for block_below in self.blocks if block_below != block and self.is_below_at_goal(block_below, block)]
                blocks_below_at_goal = [Je.Eq(self.block_vars[block_below], self.goal_state[self.block_vars[block_below]]) for block_below in blocks_below]
                additive_list += [Je.Ite(Je.And(blocks_below_at_goal + [Je.Eq(self.block_vars[block], self.goal_state[self.block_vars[block]])]), 10, 0)]
        else:  # check for singleton goals
            additive_list += [Je.Ite(Je.Eq(self.block_vars[block], self.goal_state[self.block_vars[block]]), 10, 0) for block in self.blocks]
        # penalties on wrong stacking (as per Kormann Bsc 2022)
        if False:
            additive_list += [Je.Ite(Je.Ne(self.block_vars[block], self.goal_state[self.block_vars[block]]), -10, 0) for block in self.blocks]
            for block in self.blocks:
                blocks_above = [block_above for block_above in self.blocks if block_above != block and self.is_above_at_goal(block_above, block)]
                is_on_block_above = [Je.Eq(self.block_vars[block], self.block_domains[block][block_above]) for block_above in blocks_above]
                additive_list += [Je.Ite(Je.Or(is_on_block_above), -10, 0)] if len(is_on_block_above) > 0 else []
        goal_potential = Je.Add(additive_list)
        goal_potential = Je.Ite(self.generate_start(), 0, goal_potential) if self.ground_start_potential else goal_potential
        goal_potential = Je.Ite(Je.Or(self.generate_goal_expression(), self.generate_reach()), 0, goal_potential) if self.ground_terminal_potential else goal_potential
        return goal_potential

    def generate_reach(self) -> json:
        return JaniStructureGenerator.generate_state_condition_expression([], self.is_unsafe())

    # starts ###########################################################################################################

    def generate_start(self):
        case_enum = True  # explicit enumerate cases as far as possible

        constraints = list()

        # table counter
        constraints.append(Je.Le(self.blocks_on_table(), self.table_limit))  # safe
        if self.use_table_counter and not case_enum:
            constraints.append(Je.Ge(self.table_counter, 1))  # redundant (but helpful)
            constraints.append(Je.Eq(self.table_counter, Je.Add([Je.to_int(self.block_on_table(block)) for block in self.blocks])))
        print(self.use_table_counter, self.use_time)
        if self.use_time:
            constraints.append(Je.Eq(self.time, 0))
        if case_enum:
            assert self.use_table_counter
            tmp_constraint = constraints[-1]  # workaround to preserve order wrt. old version
            constraints = constraints[:-1]
            constraints.append(Je.Ge(self.table_counter, 1))  # redundant (but helpful)
            constraints.append(tmp_constraint)
            table_disjunction = list()
            for i in range(1, self.table_limit + 1):
                combinations = itertools.combinations(self.blocks, i)
                disjunction = list()
                for combination in combinations:
                    combination_set = set(combination)
                    conjunction = list()
                    for block in self.blocks:
                        if block in combination_set:
                            conjunction.append(self.block_on_table(block))
                        else:
                            conjunction.append(self.block_not_on_table(block))
                    #
                    disjunction.append(JaniStructureGenerator.generate_large_conjunction(conjunction))
                table_disjunction.append(Je.And(self.table_counter_is(i), JaniStructureGenerator.generate_large_disjunction(disjunction)))
            constraints.append(JaniStructureGenerator.generate_large_disjunction(table_disjunction))

        # hand flag
        constraints_tmp = constraints  # workaround to preserve order wrt. old version
        constraints = list()
        if self.use_hand_empty_flag and not case_enum:   # hand-empty-flag is 1 iff one block is in hand or 0 iff no block is in hand
            constraints.append(Je.Eq(Je.Add([Je.to_int(self.block_in_hand(block)) for block in self.blocks]), self.hand_empty_flag))
        elif not case_enum:  # at most one block in hand
            constraints.append(Je.Le(Je.Add([Je.to_int(self.block_in_hand(block)) for block in self.blocks]), 1))
        if case_enum:
            assert self.use_hand_empty_flag
            hand_flag_disjunction = list()
            # hand is empty:
            hand_flag_disjunction.append([self.hand_is_empty()] + [self.block_not_in_hand(block) for block in self.blocks])
            # block in hand:
            for block in self.blocks:
                hand_flag_disjunction.append([self.hand_is_not_empty(), self.block_in_hand(block)] + [self.block_not_in_hand(block_alt) for block_alt in self.blocks if block_alt != block])
            constraints.append(JaniStructureGenerator.generate_large_disjunction([JaniStructureGenerator.generate_large_conjunction(hand_flag_case) for hand_flag_case in hand_flag_disjunction]))
        constraints += constraints_tmp

        # clear
        if self.use_clear_flags and not case_enum:  # clear if no other block on it and not in hand
            constraints.append(Je.And([Je.Eq(self.clear_flags[block], Je.Sub(1, Je.Add([Je.to_int(self.block_in_hand(block))] + [Je.to_int(self.block_on(block_alt, block)) for block_alt in self.blocks if block_alt != block]))) for block in self.blocks]))
        elif not case_enum:  # still at most one other block on block (or in hand)  ...
            constraints.append(Je.And([Je.Le(Je.Add([Je.to_int(self.block_in_hand(block))] + [Je.to_int(self.block_on(block_alt, block)) for block_alt in self.blocks if block_alt != block]), 1) for block in self.blocks]))
        if case_enum:  # case enumeration:
            assert self.use_clear_flags
            for block in self.blocks:
                clear_flag_disjunction = list()
                # clear iff no other block on block and *not* in hand
                clear_flag_disjunction.append([self.is_clear(block), self.block_not_in_hand(block)] + [self.block_not_on(block_alt, block) for block_alt in self.blocks if block_alt != block])
                # not clear if in hand
                clear_flag_disjunction.append([self.is_not_clear(block), self.block_in_hand(block)] + [self.block_not_on(block_alt, block) for block_alt in self.blocks if block_alt != block])
                # not clear if block on block
                for block_alt in self.blocks:
                    if block_alt != block:
                        clear_flag_disjunction.append([self.is_not_clear(block), self.block_not_in_hand(block), self.block_on(block_alt, block)] + [self.block_not_on(block_alt_2, block) for block_alt_2 in self.blocks if block_alt_2 != block and block_alt_2 != block_alt])
                #
                constraints.append(JaniStructureGenerator.generate_large_disjunction([JaniStructureGenerator.generate_large_conjunction(clear_flag_case) for clear_flag_case in clear_flag_disjunction]))

        # OPTIONALS:
        if self.cost_generator.zero_cost_start:
            constraints += self.cost_generator.generate_0_cost_start_constraints()

        if self.hand_empty_at_start:
            constraints.append(self.hand_is_empty())

        if self.ordered_index_start:
            # quick fix to omit cycles we fix that blocks may only be stacked ordered with respect to their block id
            for block in self.blocks:
                if self.LEGACY_DOM_ENC:
                    constraints.append(Je.Or(Je.Ge(self.block_vars[block], block + 2), self.block_on_table(block), self.block_in_hand(block)))
                else:
                    constraints.append(Je.Or(Je.Ge(self.block_vars[block], block), self.block_on_table(block)))
        else:
            if self.SUPPRESS_CYCLES:
                if self.use_height:
                    height_vars = self.block_heights
                else:
                    height_vars = self.free_height_vars
                    for var in self.free_height_vars:
                        constraints.append(Je.And(Je.Le(0, var), Je.Le(var, self.num_blocks)))  # free vars must be explicitly bounded
                for block in self.blocks:  # TODO might also use ITE
                    # base cases:
                    constraints.append(Je.Implies(self.block_on_table(block), Je.Eq(height_vars[block], 0)))
                    constraints.append(Je.Implies(self.block_in_hand(block), Je.Eq(height_vars[block], self.num_blocks)))
                    # (inductive cases) if block is on block_alt, then the height(block) = height(block_alt) + 1
                    for block_alt in self.blocks:
                        if block != block_alt:
                            constraints.append(Je.Implies(self.block_on(block, block_alt), Je.Eq(height_vars[block], Je.Add(height_vars[block_alt], 1))))

            else:
                pass  # we tolerate invalid states

        return JaniStructureGenerator.generate_let(self.generate_free_vars(), JaniStructureGenerator.generate_large_conjunction(constraints))

    def generate_random_states(self, max_number_states: int):

        def generate_stacking(stacking_sizes: list, blocks_permutation) -> tuple:
            offset = 0
            stacking = list()
            for stacking_size in stacking_sizes:
                stacking.append(tuple(blocks_permutation[offset: offset + stacking_size]))
                offset += stacking_size
            stacking.sort(reverse=True)  # fixed order
            return tuple(stacking)

        def generate_state(stacking: tuple) -> dict:
            state_mapping = dict()
            for stack in stacking:
                state_mapping[self.block_vars[stack[0]]] = self.on_table_values[stack[0]]
                if self.use_height:
                    state_mapping[self.block_heights[stack[0]]] = 0
                for stack_index, stack_index_alt in zip(range(1, len(stack)), range(0, len(stack) - 1)):
                    block, block_alt = stack[stack_index], stack[stack_index_alt]
                    state_mapping[self.block_vars[block]] = self.block_domains[block][block_alt]
                    if self.use_clear_flags:
                        state_mapping[self.clear_flags[block_alt]] = self.NOT_CLEAR  # block below is not clear
                    if self.use_height:
                        state_mapping[self.block_heights[block]] = state_mapping[self.block_heights[block_alt]] + 1
            # blocks not set not clear are clear:
            if self.use_clear_flags:
                for clear_flag in self.clear_flags.values():
                    if clear_flag not in state_mapping:
                        state_mapping[clear_flag] = self.CLEAR
            if self.use_table_counter:
                state_mapping[self.table_counter] = len(stacking)
            return state_mapping

        def sample_state() -> dict:
            fixed_ordering = False  # fixed ordering as in compact representation

            num_stacks = random.randint(1, self.table_limit)

            stacking_sizes = list()
            num_of_stacks_remaining = num_stacks
            num_blocks_remaining = self.num_blocks
            for i in range(0, num_stacks):
                num_of_stacks_remaining -= 1
                stacking_sizes.append(random.randint(1, num_blocks_remaining - num_of_stacks_remaining))
                num_blocks_remaining -= stacking_sizes[-1]
            stacking_sizes[-1] += num_blocks_remaining
            assert sum(stacking_sizes) == self.num_blocks
            stacking_sizes.sort()  # fixed order

            if fixed_ordering:
                return generate_state(generate_stacking(stacking_sizes, list(reversed(self.blocks))))
            else:
                blocks_permutation = list(self.blocks)
                random.shuffle(blocks_permutation)
                return generate_state(generate_stacking(stacking_sizes, blocks_permutation))

        def generate_all_states() -> list:
            blocks_permutations = list(itertools.permutations(self.blocks))  # list to re-iterate several times
            stacking_size_permutations = list()
            stacking_size_permutations.append((self.num_blocks,))  # special case single stack
            # more than one stack
            for num_of_stacks in range(2, self.table_limit + 1):
                stacking_size_permutations_per_num_of_stacks = set()
                for split_combination in itertools.combinations(range(0, self.num_blocks - 1), num_of_stacks - 1):  # last stack obtains at least one block + remaining blocks
                    offset = 0
                    stacking_sizes = list()
                    for index in split_combination:  # split combination contains maximum index (and any index for last stack)
                        stacking_sizes.append(index + 1 - offset)
                        offset += stacking_sizes[-1]
                    stacking_sizes.append(self.num_blocks - sum(stacking_sizes))
                    stacking_sizes.sort()  # fixed order
                    stacking_size_permutations_per_num_of_stacks.add(tuple(stacking_sizes))
                for stacking in stacking_size_permutations_per_num_of_stacks:
                    stacking_size_permutations.append(stacking)

            states = list()
            for stacking_size in stacking_size_permutations:
                for blocks_permutation in blocks_permutations:
                    states.append(generate_state(generate_stacking(list(stacking_size), blocks_permutation)))
            return states

        print("Generating " + str(max_number_states) + " random states ...")
        states_values = JaniModelGenerator.StateValuesGenerator(max_fails=max_number_states, default_state=self.initial_state)

        # for small state spaces (generate all and choose):
        if math.factorial(self.num_blocks) < max_number_states * max_number_states and False:
            all_states = generate_all_states()
            if len(all_states) <= max_number_states:
                used_states = all_states
            else:
                used_states = random.sample(all_states, max_number_states)
            for state in used_states:
                rlt = states_values.add(state)
                assert rlt is None
            if len(all_states) < max_number_states:
                print("Warning: only " + str(states_values.size()) + " exist!")
        else:
            while states_values.size() < max_number_states:
                candidate = sample_state()
                rlt = states_values.add(candidate)
                if rlt is not None:
                    return rlt

        return states_values.generate_states_values()

    # predicate generation #############################################################################################

    def generate_stepwise_splits(self) -> list:
        splits_list = [[1, current_splits] for current_splits in range(0, self.block_domain_size)]  # # flags and blocks/table-counter + cost
        return [[0, 0] + ([0] if self.cost_generator.do_cost() else [])] + self.cost_generator.generate_stepwise_splits(splits_list)  # here additional 0-step to separately add flags

    def generate_splits_mapping(self, splits: list):
        JaniModelGenerator.unused_splits_check(splits, 3 if self.cost_generator.do_cost() else 2)  # flags and blocks/table-counter + cost
        splits_specs = list()
        # block variables:
        for block in self.blocks:
            splits_specs.append(VarSplitSpec(BoundedVariable(self.block_vars[block], 0, self.block_domain_size - 1), splits[1]))
        # table counter
        if self.use_table_counter:
            splits_specs.append(VarSplitSpec(BoundedVariable(self.table_counter, 0, self.table_domain_size - 1), splits[1]))
            splits_specs[-1].add_required_split(self.table_limit + 1)  # always used to distinguish unsafe states
        # cost:
        self.cost_generator.generate_splits_mapping(splits_specs, splits)
        return splits_specs

    def generate_predicates(self, splits: list):
        predicates_flag = ([self.hand_is_empty()] + [self.is_clear(block) for block in self.blocks]) if splits[0] == 1 else []
        predicates_splitting = self.generate_splitting_predicates(self.generate_splits_mapping(splits))
        return predicates_flag + predicates_splitting + ([] if self.use_table_counter else [Je.Le(self.table_limit + 1, self.blocks_on_table())])  # if possible added via splits to preserve order

    # interface generation #############################################################################################

    def get_nn_inputs(self) -> list:
        return self.cost_generator.adapt_nn_inputs(list(self.variable_names))

    ####################################################################################################################


if __name__ == "__main__":
    args = BlocksWorldModelGenerationOptionParser().arg_parse()
    generator = BlocksWorldModelGenerator(args)
    generator.generate()
