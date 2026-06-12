"""
Python port of domain_specific/manual_policies/inverted_pendulum.cpp

Hand-crafted heuristic for the InvertedPendulum domain.  Used as a shaping
signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++):
  0: push one way
  1: push the other way
  2: no-op / neutral
"""

ACTION_POSITIVE = 0
ACTION_NEGATIVE = 1
ACTION_NEUTRAL = 2


def inverted_pendulum_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    State layout in the Python JANIEngine obs vector should match the C++
    StateValues float access:
      obs[2] : angle
      obs[3] : angular_velocity
    """
    angle = float(obs[2])
    angular_velocity = float(obs[3])

    value = angle + 0.5 * angular_velocity
    if value > 0:
        return ACTION_POSITIVE
    if value < 0:
        return ACTION_NEGATIVE
    return ACTION_NEUTRAL
