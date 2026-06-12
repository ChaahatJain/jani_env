"""
Python port of domain_specific/manual_policies/cartpole.cpp

Hand-crafted heuristic for the Cartpole domain.  Used as a shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++):
  0: push one way
  1: push the other way
  2: no-op / neutral
"""

ACTION_NEGATIVE = 0
ACTION_POSITIVE = 1
ACTION_NEUTRAL = 2


def cartpole_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    State layout in the Python JANIEngine obs vector should match the C++
    StateValues float access:
      obs[2] : angle
      obs[3] : angular_velocity

    In C++, Cartpole reads get_float(4) and get_float(5).  The Python
    observation omits the leading automaton/location slots, so these correspond
    to obs[2] and obs[3], matching inverted_pendulum_policy.py.
    """
    angle = float(obs[2])
    angular_velocity = float(obs[3])

    value = angle + 0.5 * angular_velocity
    if value > 0:
        return ACTION_POSITIVE
    if value < 0:
        return ACTION_NEGATIVE
    return ACTION_NEUTRAL
