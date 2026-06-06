"""
Python port of domain_specific/manual_policies/transport.cpp / transport_feat.cpp

The policy is a greedy hand-crafted heuristic that works for both the plain
Transport and Transport_Feat state layouts.  It is used as a shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++ constants):
  0: pick up
  1: drop
  2: move forward
  3: move backward
"""

ACTION_PICKUP   = 0
ACTION_DROP     = 1
ACTION_FORWARD  = 2
ACTION_BACKWARD = 3


def transport_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    State layout in the Python JANIEngine obs vector:
      obs[0]                : num_locations  (constant – always first)
      obs[1]                : num_packages   (constant)
      obs[2..5]             : other constants (dropping_prob, tank_capacity, ...)
      obs[6 .. 6+N-1]       : location_load_0 .. location_load_{N-1}
      obs[6+N]              : truck_position
      obs[6+N+1]            : truck_load
      obs[6+N+2]            : last_capacity_diff  (Transport_Feat extra field)

    There are always 6 constants and 3 fixed tail variables (truck_pos, truck_load,
    last_capacity_diff), so N = len(obs) - 9, or equivalently int(obs[0]).
    """
    obs = [int(x) for x in obs]

    # Number of locations is stored directly in the first constant slot.
    N_CONSTANTS = 6
    number_of_locations = int(obs[0])   # obs[0] == num_locations constant
    loc_start           = N_CONSTANTS   # index of location_load_0 in obs

    truck_position = obs[loc_start + number_of_locations]      # obs[6 + N]
    truck_load     = obs[loc_start + number_of_locations + 1]  # obs[6 + N + 1]
    goal_position  = number_of_locations - 1
    bridge_start   = number_of_locations - 2

    packages_here = obs[loc_start + truck_position]  # obs[6 + truck_position]

    # Rule 1: drop excess load before the bridge
    if truck_load > 1 and truck_position == bridge_start:
        return ACTION_DROP

    # Rule 2: deliver at the goal
    if truck_load >= 1 and truck_position == goal_position:
        return ACTION_DROP

    # Rule 3: carrying something → drive toward goal
    if truck_load >= 1:
        return ACTION_FORWARD

    # Rule 4: empty, package here → pick it up
    if packages_here > 0 and truck_position != goal_position:
        return ACTION_PICKUP

    # Rule 5: empty, nothing here → go back if there's a package behind, else forward
    package_behind = any(obs[loc_start + loc] > 0 for loc in range(truck_position))
    return ACTION_BACKWARD if package_behind else ACTION_FORWARD
