"""
Python port of domain_specific/manual_policies/follow_car.cpp

The current C++ FollowCar teacher is a constant policy.  It is used as a
shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches benchmarks_generator/generator/follow_car/generator.py):
  0: accelerate
  1: decelerate
"""

ACTION_ACCELERATE = 0
ACTION_DECELERATE = 1


def follow_car_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    State layout in the Python JANIEngine obs vector:
      obs[0] : terminal_at_unsafe constant
      obs[1] : timestep constant
      obs[2] : lead_velocity constant
      obs[3] : episode
      obs[4] : lead_position
      obs[5] : ego_position
      obs[6] : ego_velocity

    The C++ FollowCar policy currently ignores StateValues and returns action 1.
    """
    return ACTION_DECELERATE
