import re
import json
import pickle
pddl_predicates = ["on(part01, part02, fside/bside)", "in(part01, rack01/beluga/t1/t2/pla)", "empty(t1/t2)", "clear(part02, fside/bside)"]

pddl_actions = ["putdown(part01, rack01/plA, t1, bside)", "stack(part01, part02, rack01/plA, t2, fside, bside)",\
           "pickup(part01, rack01, t2, fside, bside)", "unstack(part01, part02, rack01, t2, fside, bside)"]

jani_action_labels =[
            "unload_beluga_0",
            "send_to_factory",
            "push_front_rack_0",
            "push_front_rack_1",
            "push_back_rack_0",
            "push_back_rack_1",
            "pop_front_rack_0",
            "pop_front_rack_1",
            "pop_back_rack_0",
            "pop_back_rack_1"]



def pddl_actions_to_jani_actions(pddl_action):
    # No plan ever pushes truck elements onto rack. Instead pushing/popping onto a trailer is allowed in PDDL.
    # This action is disallowed in JANI and all swaps must go through truck.
    print(pddl_action)
    if pddl_action.startswith("putdown"):
        # Can be either sending to factory or putting on rack
        pattern = r'putdown\((\w*),(\w*),(\w*),(\w*)\)'
        match = re.match(pattern, pddl_action)
        assert(match)
        jig = match.group(1)
        location = match.group(2)
        trailer = match.group(3)
        side = match.group(4)
        if location.startswith("rack"):
            # action is pushing to a rack
            pattern = r'rack([0-9]+)'
            match = re.match(pattern, location)
            assert(match)
            rack_num = int(match.group(1)) - 1
            return f'push_back_rack_{rack_num}' if side == "fside" else f'push_front_rack_{rack_num}'
        else:
            return "send_to_factory"
        assert(False)
    elif pddl_action.startswith("stack"):
        # Can be either sending to the factory or putting on rack
        pattern = r'stack\((\w*),(\w*),(\w*),(\w*),(\w*),(\w*)\)'
        match = re.match(pattern, pddl_action)
        assert(match)
        jig1 = match.group(1)
        jig2 = match.group(2)
        location = match.group(3)
        trailer = match.group(4)
        side = match.group(5)
        if location.startswith("rack"):
            pattern = r'rack([0-9]+)'
            match = re.match(pattern, location)
            assert(match)
            rack_num = int(match.group(1)) - 1
            return f'push_back_rack_{rack_num}' if side == "fside" else f'push_front_rack_{rack_num}'
        else:
            return "send_to_factory"
        assert(False)
    elif pddl_action.startswith("unstack"):
        # Used to unload from beluga, or pop from rack. 
        pattern = r'unstack\((\w*),(\w*),(\w*),(\w*),(\w*),(\w*)\)'
        match = re.match(pattern, pddl_action)
        assert(match)
        jig1 = match.group(1)
        jig2 = match.group(2)
        location = match.group(3)
        trailer = match.group(4)
        side = match.group(5)
        if location.startswith("rack"):
            pattern = r'rack([0-9]+)'
            match = re.match(pattern, location)
            assert(match)
            rack_num = int(match.group(1)) - 1
            return f'pop_back_rack_{rack_num}' if side == "fside" else f'pop_front_rack_{rack_num}'
        else:
            return "unload_beluga_0"
    elif pddl_action.startswith("pickup"):
        # Can be used to unload beluga or unload rack, or send to truck
        pattern = r'pickup\((\w*),(\w*),(\w*),(\w*),(\w*)\)'
        match = re.match(pattern, pddl_action)
        assert(match)
        jig = match.group(1)
        location = match.group(2)
        trailer = match.group(3)
        side = match.group(4)
        if location.startswith("rack"):
            pattern = r'rack([0-9]+)'
            match = re.match(pattern, location)
            assert(match)
            rack_num = int(match.group(1)) - 1
            return f'pop_back_rack_{rack_num}' if side == "fside" else f'pop_front_rack_{rack_num}'
        elif location.startswith("beluga"):
            return "unload_beluga_0"
        else:
            return "send_to_factory"
    assert(False)
    return 

def get_num_factory_delivered(pddl_state):
    pattern = r'in\(part(\d*),pla\)'
    return len(re.findall(pattern, pddl_state))

def num_jigs_on_beluga(pddl_state):
    pattern = r'in\(part(\d*),beluga\)'
    return len(re.findall(pattern, pddl_state))

def is_trailer_empty(pddl_state):
    pattern = r'in\(part(\d*),t1\)'
    return not len(re.findall(pattern, pddl_state))

def is_truck_empty(pddl_state):
    pattern = r'in\(part(\d*),t2\)'
    return not len(re.findall(pattern, pddl_state))

def num_jigs_on_rack(pddl_state, rack_index):
    pattern = r'in\(part(\d*),rack0{}\)'.format(rack_index)
    return len(re.findall(pattern, pddl_state))

def rack_load(pddl_state, rack_index):
    return 10 - num_jigs_on_rack(pddl_state, rack_index)*2

def get_jig_orderings_rack(pddl_state, rack_index):
    num_rack_jigs = num_jigs_on_rack(pddl_state, rack_index)
    parts = []
    if num_rack_jigs == 0: 
        return parts
    rack_pattern = r'in\(part(\d*),rack0{}\)'.format(rack_index)
    rack = re.findall(rack_pattern, pddl_state)
    bottom_pattern = r'clear\(part(\d*),bside\)'
    bottom = re.findall(bottom_pattern, pddl_state)
    print(rack, bottom)
    part = list(set(rack) & set(bottom))[0]
    parts.append(int(part))
    while len(parts) < num_rack_jigs:
        next_pattern = r'on\(part{},part(\d*),bside\)'.format(part)
        match = re.search(next_pattern, pddl_state)
        assert match, f"next pattern on rack fails for part {part} with pattern {next_pattern} for state {pddl_state}"
        part = match.group(1)
        parts.append(int(part))
    return parts

def get_jig_orderings_beluga(pddl_state):
    num_beluga_jigs = num_jigs_on_beluga(pddl_state)
    parts = []
    if num_beluga_jigs == 0: 
        return parts
    in_beluga_pattern = r'in\(part(\d*),beluga\)'
    beluga = re.findall(in_beluga_pattern, pddl_state)
    bottom_pattern = r'clear\(part(\d*),bside\)'
    bottom = re.findall(bottom_pattern, pddl_state)
    part = list(set(beluga) & set(bottom))[0] 
    parts.append(int(part))
    while len(parts) < num_beluga_jigs:
        next_pattern = r'on\(part{},part(\d*),bside\)'.format(part)
        match = re.search(next_pattern, pddl_state)
        part = match.group(1)
        parts.append(int(part))
    return parts
    

def get_jig_position(pddl_state, jig_index):
    pattern = r'in\(part0{},([^)]+)\);'.format(jig_index)
    match = re.search(pattern, pddl_state)
    assert(match)
    location = match.group(1)
    if location.startswith("beluga"):
        # Find its place on the beluga and index accordingly
        return 0
    elif location.startswith("rack"):
        pattern = r'rack([0-9]+)'
        match = re.match(pattern, location)
        assert(match)
        rack_num = int(match.group(1))
        return get_num_jigs(pddl_state)*1 + rack_num
    elif location.startswith("t"):
        pattern = r't([0-9]+)'
        match = re.match(pattern, location)
        assert(match)
        trailer_num = int(match.group(1)) - 1
        return get_num_jigs(pddl_state)*1 + get_num_racks(pddl_state) + trailer_num
    else:
       return get_num_jigs(pddl_state)*1 + get_num_racks(pddl_state) + 1 
    
def get_num_jigs(pddl_state):
    pattern = r'part(\d+)'
    return len(set(re.findall(pattern, pddl_state)))

def get_num_racks(pddl_state):
    pattern = r'rack(\d+)'
    return len(set(re.findall(pattern, pddl_state)))        

jani_state_variables = [
    "jig_0", "jig_1", "jig_2", "jig_3",
    "front_jig_0", "front_jig_1", "front_jig_2", "front_jig_3",
    "back_jig_0", "back_jig_1", "back_jig_2", "back_jig_3",
    "rack_load_0", "rack_load_1", "num_jigs_on_rack_0", "num_jigs_on_rack_1",
    "front_rack_0", "front_rack_1", "back_rack_0", "back_rack_1", 
    "empty_trailer_0", "beluga_0", "num_factory_delivered", "empty_truck"]

def pddl_state_to_jani_state_variable(pddl_state):
    state = '; '.join(pddl_state)
    num_racks = get_num_racks(state)
    jani_state = {}
    for rack in range(0, num_racks):
        num = num_jigs_on_rack(state, rack + 1)
        jani_state[f'num_jigs_on_rack_{rack}'] = num
        rl = rack_load(state, rack + 1)
        jani_state[f'rack_load_{rack}'] = rl
    jani_state["empty_trailer_0"] = 1 if is_trailer_empty(state) else 0
    jani_state["empty_truck"] = 1 if is_truck_empty(state) else 0
    jani_state["beluga_0"] = num_jigs_on_beluga(state)
    jani_state["num_factory_delivered"] = get_num_factory_delivered(state)
    beluga_orderings = get_jig_orderings_beluga(state)
    order = 0
    num_jigs = get_num_jigs(state)
    for jig in range(1, num_jigs + 1):
        jani_state[f"jig_{jig - 1}"] = get_jig_position(state, jig)
        jani_state[f"front_jig_{jig - 1}"] = -1
        jani_state[f"back_jig_{jig - 1}"] = -1
    for jig in beluga_orderings:
        jani_state[f"jig_{jig - 1}"] = order
        order = order + 1
        jani_state[f"front_jig_{jig - 1}"] = -1
        jani_state[f"back_jig_{jig - 1}"] = -1
    
    for rack in range(0, num_racks):
        rack_orderings = get_jig_orderings_rack(state, rack + 1)
        order = 0
        back_part = 0
        jani_state[f"front_rack_{rack}"] = rack_orderings[0] if len(rack_orderings) > 0 else -1
        jani_state[f"back_rack_{rack}"] = rack_orderings[len(rack_orderings) - 1] if len(rack_orderings) > 0 else -1
        for jig in rack_orderings:
            jani_state[f"front_jig_{jig - 1}"] = -1 if order == 0 else back_part
            back_part = rack_orderings[order + 1] if order < len(rack_orderings) - 1 else -1
            jani_state[f"back_jig_{jig - 1}"] = back_part
    return jani_state
    
if __name__ == "__main__":
    num_jigs = 4
    num_racks = 2
    with open(f"pddl_files/move_ignore_{num_jigs}_{num_racks}.json", 'r') as file:
        problem_plan = json.load(file)
        jani_states = []
        jani_actions = []
        for problem in problem_plan.keys():
            assert(problem.startswith(f"problem_{num_jigs}_{num_racks}"))
            plan = problem_plan[problem]
            print(problem, len(plan))
            for s in plan:
                pddl_state = s["state"]
                pddl_action = s["action"]
                jani_state = pddl_state_to_jani_state_variable(pddl_state)
                # print(jani_state.keys())
                assert(set(jani_state.keys()) == set(jani_state_variables))
                janistate = [jani_state[variable] for variable in jani_state_variables]
                jani_action = pddl_actions_to_jani_actions(pddl_action)
                janiaction = jani_action_labels.index(jani_action)
                jani_states.append(janistate)
                jani_actions.append(janiaction)
        print(jani_actions)
        with open(f"pickle_files/move_ignore_{num_jigs}_{num_racks}_states.pkl", 'wb') as f:
            pickle.dump(jani_states, f)
        with open(f"pickle_files/move_ignore_{num_jigs}_{num_racks}_actions.pkl", 'wb') as f:
            pickle.dump(jani_actions, f)