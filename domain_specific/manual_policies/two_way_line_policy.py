"""
Python port of domain_specific/manual_policies/two_way_line.cpp

Greedy hand-crafted heuristic for the TwoWayLine domain.  Used as a shaping
signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++ constants):
  0: pick up
  1: drop
  2: accelerate
  3: decelerate
  4: move

Note: this policy is not necessarily safe for the Icy variant because packages
can be nondeterministically dropped during moving.
"""

ACTION_PICKUP     = 0
ACTION_DROP       = 1
ACTION_ACCELERATE = 2
ACTION_DECELERATE = 3
ACTION_MOVE       = 4


def two_way_line_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    State layout in the Python JANIEngine obs vector:
      obs[0]               : num_locations   (constant)
      obs[1]               : num_packages    (constant)
      obs[2]               : dropping_prob   (constant)
      obs[3]               : slipping_prob   (constant)
      obs[4]               : icy_prob        (constant)
      obs[5]               : tank_capacity   (constant)
      obs[6]               : add_parking     (constant)
      obs[7]               : fail_dec_on_ice (constant)
      obs[8]               : terminal-at-unsafe (constant)
      obs[9 .. 9+N-1]      : location_load_0 .. location_load_{N-1}
      obs[9+N]             : truck_0         (truck position)
      obs[9+N+1]           : truck_load_0    (truck load)
      obs[9+N+2]           : truck_vel_0     (truck velocity)
      obs[9+N+3]           : aux_vel

    N = int(obs[0]) = num_locations
    """
    obs = [int(x) for x in obs]

    N_CONSTANTS = 9
    number_of_locations = obs[0]   # num_locations constant
    loc_start           = N_CONSTANTS

    truck_position = obs[loc_start + number_of_locations]      # truck_0
    truck_load     = obs[loc_start + number_of_locations + 1]  # truck_load_0
    truck_velocity = obs[loc_start + number_of_locations + 2]  # truck_vel_0
    goal_position  = number_of_locations - 1

    packages_here = obs[loc_start + truck_position]

    packages_next = (
        obs[loc_start + truck_position + 1]
        if truck_position + 1 < number_of_locations
        else 0
    )

    packages_prev = obs[loc_start + truck_position]  # same as packages_here (mirrors C++)

    package_behind = any(
        obs[loc_start + loc] > 0
        for loc in range(truck_position)
    )

    # Rule: at goal with velocity 0 and carrying a package -> drop
    if truck_position == goal_position and truck_velocity == 0 and truck_load > 0:
        return ACTION_DROP

    # Rule: one step before goal at velocity 1 -> decelerate to arrive stopped
    if truck_position == goal_position - 1 and truck_velocity == 1:
        return ACTION_DECELERATE

    # Rule: moving forward and next cell has a package -> decelerate to stop and pick up
    if truck_velocity == 1 and packages_next > 0:
        return ACTION_DECELERATE

    # Rule: moving backward and current/prev cell has a package -> accelerate (slow down reverse)
    if truck_velocity == -1 and packages_prev > 0:
        return ACTION_ACCELERATE

    # Rule: stopped on top of a package (not at goal) -> pick it up
    if truck_velocity == 0 and packages_here > 0 and truck_position != goal_position:
        return ACTION_PICKUP

    # Rule: there is a package behind -> start reversing (decelerate from forward or zero)
    if truck_velocity >= 0 and package_behind:
        return ACTION_DECELERATE

    # Rule: moving (forward or backward) with nothing to stop for -> keep coasting
    if truck_velocity in (1, -1):
        return ACTION_MOVE

    # Rule: moving faster than 1 forward -> decelerate
    if truck_velocity > 1:
        return ACTION_DECELERATE

    # Rule: moving faster than 1 backward -> accelerate (slow down)
    if truck_velocity < -1:
        return ACTION_ACCELERATE

    # --- velocity == 0 from here on ---

    # Rule: velocity 0, nothing here -> accelerate
    return ACTION_ACCELERATE
