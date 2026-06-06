"""
Python port of domain_specific/manual_policies/beluga.cpp

Greedy hand-crafted heuristic for the Beluga domain.  Used as a shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++ BelugaActions and Python generator action_labels):
  0            : pop_beluga
  1 .. R       : push_rack_0 .. push_rack_{R-1}
  R+1 .. 2R    : pop_rack_0  .. pop_rack_{R-1}
  2R+1         : deliver
  2R+2         : send_back
  2R+3         : no_op

  where R = num_racks = int(obs[2])

State layout in the Python JANIEngine obs vector:
  Constant block (indices 0..4):
    obs[0] : max_swaps          (constant)
    obs[1] : num_jigs           (constant)
    obs[2] : num_racks          (constant)
    obs[3] : rack_max_capacity  (constant)
    obs[4] : num_production_lines (constant)

  Variable block (N_CONSTANTS = 5):
    obs[5 .. 5+R-1]             : num_jigs_on_rack_0 .. num_jigs_on_rack_{R-1}
    obs[5+R .. 5+R+R*C-1]       : rack_0_0, rack_0_1, ..., rack_{R-1}_{C-1}
                                  (C = rack_max_capacity, row-major over racks then slots)
    obs[5+R+R*C]                : trailer_0
    obs[5+R+R*C+1 .. 5+R+R*C+J]: beluga_1 .. beluga_J   (J = num_jigs)
    obs[5+R+R*C+J+1]            : next_line
    obs[5+R+R*C+J+2 .. +J+1+L] : line_1 .. line_L       (L = num_production_lines)
    obs[5+R+R*C+J+1+L+1]        : truck
    obs[5+R+R*C+J+1+L+2]        : num_swaps
"""

N_CONSTANTS = 5  # max_swaps, num_jigs, num_racks, rack_max_capacity, num_production_lines


def beluga_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    Returns:
        Integer action index.
    """
    obs = [int(x) for x in obs]

    num_jigs      = obs[1]
    num_racks     = obs[2]
    rack_capacity = obs[3]
    num_lines     = obs[4]

    # Action indices
    ACTION_POP_BELUGA = 0
    ACTION_PUSH_RACK  = [1 + r for r in range(num_racks)]
    ACTION_POP_RACK   = [1 + num_racks + r for r in range(num_racks)]
    ACTION_DELIVER    = 1 + 2 * num_racks
    ACTION_SEND_BACK  = 2 + 2 * num_racks
    ACTION_NO_OP      = 3 + 2 * num_racks

    # Variable block starts at index N_CONSTANTS
    rack_used_start  = N_CONSTANTS                              # num_jigs_on_rack_r
    rack_slots_start = rack_used_start + num_racks              # rack_r_s (row-major)
    trailer_idx      = rack_slots_start + num_racks * rack_capacity
    beluga_start     = trailer_idx + 1                          # beluga_1 .. beluga_J
    next_line_idx    = beluga_start + num_jigs
    lines_start      = next_line_idx + 1                        # line_1 .. line_L
    truck_idx        = lines_start + num_lines

    def rack_used(r: int) -> int:
        return obs[rack_used_start + r]

    def rack_slot(r: int, s: int) -> int:
        return obs[rack_slots_start + r * rack_capacity + s]

    trailer   = obs[trailer_idx]
    next_line = obs[next_line_idx]  # 1-based index of the next production line
    # line_l values are stored at lines_start + (l-1) for l in 1..num_lines
    line_val  = obs[lines_start + (next_line - 1)]
    truck     = obs[truck_idx]

    # Rule 1: next production line needs no jig
    if line_val == 0:
        return ACTION_NO_OP

    # Rule 2: truck carries the exact jig needed → deliver
    if truck > 0 and truck == line_val:
        return ACTION_DELIVER

    # Rule 3: truck carries a wrong jig → send it back
    if truck > 0 and truck != line_val:
        return ACTION_SEND_BACK

    # Rule 4: trailer has a jig → push it onto the rack with smallest used count
    if trailer > 0:
        best_rack = -1
        best_used = num_racks * rack_capacity + 1  # larger than any real value
        for r in range(num_racks):
            used = rack_used(r)
            if used < rack_capacity and used < best_used:
                best_used = used
                best_rack = r
        if best_rack >= 0:
            return ACTION_PUSH_RACK[best_rack]

    # Rule 5: truck is empty and the needed jig is on a rack → pop that rack
    if truck == 0:
        for r in range(num_racks):
            for s in range(rack_capacity):
                if rack_slot(r, s) == line_val:
                    return ACTION_POP_RACK[r]

    # Rule 6: fallback — unload the next jig from the beluga
    return ACTION_POP_BELUGA
