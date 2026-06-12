"""
Python port of domain_specific/manual_policies/bouncing_ball.cpp

Hand-crafted heuristic for the BouncingBall domain.  Used as a shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++):
  0: push the ball if possible
"""

ACTION_PUSH = 0


def bouncing_ball_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    The C++ policy ignores the state and always selects action 0.
    """
    return ACTION_PUSH
