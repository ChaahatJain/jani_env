import torch
import torch.nn as nn
import tensordict

import numpy as np

from tensordict import TensorDict
from torchrl.modules import MaskedCategorical
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from jani import JANIEnv


def collect_trajectory(env: JANIEnv, policy: nn.Module, idx: int, max_horizon: int = 1024) -> tuple[TensorDict, bool]:
    """Collect trajectories using the given policy."""
    policy.cpu() # Ensure policy is on CPU
    policy.eval() # Set policy to evaluation mode

    observations, next_observations, actions, action_masks, rewards = [], [], [], [], []
    safety, safe_actions = [], []
    obs_to_correct, corrected_actions, corrected_action_masks = [], [], []
    obs_to_keep, kept_actions, kept_action_masks = [], [], []
    safe_trajectory = True # Flag to indicate if the trajectory remains safe throughout

    obs, reset_info = env.reset(options={"idx": idx}) # Reset environment to specific initial state
    safety.append(reset_info["current_state_safety"])
    safe_actions.append(reset_info["current_safe_action"])
    for step in range(max_horizon):
        observations.append(obs) # Record current observation
        action_mask = env.unwrapped.action_mask() # Current action mask
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = policy(obs_tensor) # Compute logits from policy
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
        actions.append(action) # Record taken action
        action_masks.append(action_mask.astype(int)) # Record action mask
        obs, reward, done, _, info = env.step(action)

        assert "next_state_safety" in info, "Info dict missing 'next_state_safety'"
        assert "next_safe_action" in info, "Info dict missing 'next_safe_action'"
        # If entering into an unsafe state
        if not info["next_state_safety"]:
            safe_trajectory = False
            assert (reward == env.unwrapped._unsafe_reward) or (reward == -1.0), f"Inconsistent safety info: {reward} vs {env.unwrapped._unsafe_reward}"
            if safety[-1]:
                # If the previous state was safe but current state is unsafe, we need to correct the previous action
                obs_to_correct.append(observations[-1])
                safe_action = safe_actions[-1]
                assert safe_action != -1, f"Episode {idx} Step {step}: Safe action should be valid"
                assert safe_action != action, f"Episode {idx} Step {step}: Safe action should differ from the taken unsafe action {action}"
                assert action_mask[safe_action], f"Episode {idx} Step {step}: Safe action should be valid under action mask"
                corrected_actions.append(safe_action)
                corrected_action_masks.append(action_mask.astype(int))
                assert len(obs_to_correct) == len(corrected_action_masks) == len(corrected_actions), "Mismatch in correction lengths"
        else: # If the next state is still safe
            # TODO: Maybe should seperate transitions from an unsafe state to a safe state, it is also important
            obs_to_keep.append(observations[-1]) # Keep the action taken in the previous safe state unchanged
            kept_actions.append(action) # This is to restrict the policy not deviate too much
            kept_action_masks.append(action_mask.astype(int))
            assert len(obs_to_keep) == len(kept_action_masks) == len(kept_actions), f"Episode {idx} Step {step}: Mismatch in kept action lengths"

        rewards.append(reward)
        next_observations.append(obs)
        if done:
            break

        # Record safety info, ignore the last step if done
        safety.append(info["next_state_safety"])
        safe_actions.append(info["next_safe_action"])

    assert len(observations) == len(actions) == len(rewards) == len(next_observations), f"Mismatch in trajectory lengths, get {len(observations)}, {len(actions)}, {len(rewards)}, {len(next_observations)}"

    # Convert lists to numpy arrays for efficient tensor conversion
    observation_array = np.stack(observations, axis=0)
    next_observation_array = np.stack(next_observations, axis=0)
    obs_to_correct_array = np.stack(obs_to_correct, axis=0) if len(obs_to_correct) > 0 else np.empty((0, observation_array.shape[1]))
    mask_to_correct_array = np.stack(corrected_action_masks, axis=0) if len(corrected_action_masks) > 0 else np.empty((0, observation_array.shape[1]))
    obs_to_keep_array = np.stack(obs_to_keep, axis=0) if len(obs_to_keep) > 0 else np.empty((0, observation_array.shape[1]))
    mask_to_keep_array = np.stack(kept_action_masks, axis=0) if len(kept_action_masks) > 0 else np.empty((0, observation_array.shape[1]))

    trajectory = TensorDict({
        "observation": torch.tensor(observation_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "action": torch.tensor(actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "next_observation": torch.tensor(next_observation_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "reward": torch.tensor(rewards, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "safety": torch.tensor(safety, dtype=torch.bool).unsqueeze(0),  # Add batch dimension 1
        "safe_action": torch.tensor(safe_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "obs_to_correct": torch.tensor(obs_to_correct_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "corrected_action": torch.tensor(corrected_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "corrected_action_mask": torch.tensor(mask_to_correct_array, dtype=torch.bool).unsqueeze(0),  # Add batch dimension 1
        "obs_to_keep": torch.tensor(obs_to_keep_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "kept_action": torch.tensor(kept_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "kept_action_mask": torch.tensor(mask_to_keep_array, dtype=torch.bool).unsqueeze(0)  # Add batch dimension 1
    }, batch_size=())

    info = {
        "is_safe_trajectory": safe_trajectory,
        "final_reward": rewards[-1]
    }

    # For debugging
    # print(f"Safety info for trajectory {idx}: {safety}")

    return (trajectory, info)


def collect_trajectory_with_stricted_rule(
        env: JANIEnv, 
        policy: nn.Module, 
        idx: int, 
        max_horizon: int = 1024,
        no_safety_check: bool = False
    ) -> tuple[TensorDict, bool]:
    """Collect trajectories using the given policy, but subject to the stricted safety rule."""
    policy.cpu() # Ensure policy is on CPU
    policy.eval() # Set policy to evaluation mode

    assert not env.unwrapped._use_oracle, "Environment should not use oracle during trajectory collection with stricted rule."

    observations, next_observations, actions, action_masks, rewards = [], [], [], [], []
    obs_to_correct, corrected_actions, corrected_action_masks = [], [], []
    obs_to_keep, kept_actions, kept_action_masks = [], [], []
    safety, safe_actions = [], []
    safe_trajectory = True # Flag to indicate if the trajectory remains safe throughout

    if idx != -1:
        obs, reset_info = env.reset(options={"idx": idx, "no_safety_info": True}) # Reset environment to specific initial state
    else:
        obs, _ = env.reset()
    
    for step in range(max_horizon):
        observations.append(obs) # Record current observation
        action_mask = env.unwrapped.action_mask() # Current action mask
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = policy(obs_tensor) # Compute logits from policy
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
        actions.append(action) # Record taken action
        action_masks.append(action_mask.astype(int)) # Record action mask

        # Check whether the action is safe under the current state (before stepping)
        if not no_safety_check:
            is_state_safe, safe_action = env.unwrapped.current_state_safety_with_action(action)
        else:
            is_state_safe, safe_action = False, -1 # If safety check is disabled, we consider all states as safe and there is no safe action
        safety.append(is_state_safe)
        safe_actions.append(safe_action)
        if safe_action != -1 and safe_action != action:
            assert is_state_safe, f"State should be safe if there is a safe action exists"
            safe_trajectory = False # Mark the trajectory as unsafe because an unsafe action is taken
            # If there is a safe action differ from the taken action, we need to correct it
            obs_to_correct.append(obs)
            corrected_actions.append(safe_action)
            corrected_action_masks.append(action_mask.astype(int))
        else:
            if not is_state_safe:
                assert safe_action == -1, f"Safe action should be -1 if the state is unsafe"
                safe_trajectory = False # Mark the trajectory as unsafe because we are in an unsafe state
            if safe_action == action:
                obs_to_keep.append(obs) # Keep the action taken in the previous safe state unchanged
                kept_actions.append(action) # This is to restrict the policy not deviate too much
                kept_action_masks.append(action_mask.astype(int))

        # Step the environment    
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        next_observations.append(obs)
        if done:
            break

    assert len(observations) == len(actions) == len(rewards) == len(next_observations), f"Mismatch in trajectory lengths, get {len(observations)}, {len(actions)}, {len(rewards)}, {len(next_observations)}"

    # Convert lists to numpy arrays for efficient tensor conversion
    observation_array = np.stack(observations, axis=0)
    next_observation_array = np.stack(next_observations, axis=0)
    obs_to_correct_array = np.stack(obs_to_correct, axis=0) if len(obs_to_correct) > 0 else np.empty((0, observation_array.shape[1]))
    mask_to_correct_array = np.stack(corrected_action_masks, axis=0) if len(corrected_action_masks) > 0 else np.empty((0, observation_array.shape[1]))
    obs_to_keep_array = np.stack(obs_to_keep, axis=0) if len(obs_to_keep) > 0 else np.empty((0, observation_array.shape[1]))
    mask_to_keep_array = np.stack(kept_action_masks, axis=0) if len(kept_action_masks) > 0 else np.empty((0, observation_array.shape[1]))

    trajectory = TensorDict({
        "observation": torch.tensor(observation_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "action": torch.tensor(actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "next_observation": torch.tensor(next_observation_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "reward": torch.tensor(rewards, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "safety": torch.tensor(safety, dtype=torch.bool).unsqueeze(0),  # Add batch dimension 1
        "safe_action": torch.tensor(safe_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "obs_to_correct": torch.tensor(obs_to_correct_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "corrected_action": torch.tensor(corrected_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "corrected_action_mask": torch.tensor(mask_to_correct_array, dtype=torch.bool).unsqueeze(0),  # Add batch dimension 1
        "obs_to_keep": torch.tensor(obs_to_keep_array, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "kept_action": torch.tensor(kept_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "kept_action_mask": torch.tensor(mask_to_keep_array, dtype=torch.bool).unsqueeze(0)  # Add batch dimension 1
    }, batch_size=())

    info = {
        "is_safe_trajectory": safe_trajectory,
        "final_reward": rewards[-1]
    }

    return (trajectory, info)



class DAggerBuffer:
    """Replay buffer for DAgger algorithm."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # Initialize storages for replay buffer
        self.positive_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
        ) # Buffer for safe state-action pairs
        self.negative_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
        ) # Buffer for state-action pairs corrected by the oracle

    def add_samples(self, positive_samples: TensorDict, negative_samples: TensorDict):
        """Add samples to the respective buffers."""
        # print(f"{positive_samples.batch_size}, {negative_samples.batch_size}")
        # print(f"Shape info: {positive_samples['observation'].shape}, {negative_samples['observation'].shape}")
        if positive_samples.batch_size[0] != 0:
            self.positive_buffer.extend(positive_samples)
        if negative_samples.batch_size[0] != 0:
            self.negative_buffer.extend(negative_samples)

    def sample(self, batch_size: int):
        """Sample a batch of state-action pairs from both buffers."""
        #  TODO: handle the case when one buffer is empty or smaller than required batch size
        if len(self.positive_buffer) == 0 and len(self.negative_buffer) == 0:
            print("Both buffers are empty, training terminated.")
            exit(0)
        if len(self.positive_buffer) == 0:
            print("Positive buffer is empty, sampling only from negative buffer.")
            batch_data = self.negative_buffer.sample(batch_size)
            return batch_data
        if len(self.negative_buffer) == 0:
            print("Negative buffer is empty, sampling only from positive buffer.")
            batch_data = self.positive_buffer.sample(batch_size)
            return batch_data
        # batch_size_neg = min(batch_size // 2, len(self.negative_buffer))
        # batch_size_pos = batch_size - batch_size_neg
        sub_batch_size = batch_size // 2
        pos_batch = self.positive_buffer.sample(sub_batch_size)
        neg_batch = self.negative_buffer.sample(sub_batch_size)
        # Ususally shuffle is not necessary since we are doing behavior cloning
        batch_data = tensordict.cat([pos_batch, neg_batch], dim=0)
        return batch_data
    
    def add_rollouts(self, rollouts: list[tuple[TensorDict, bool]]):
        """Add multiple rollouts to the buffer."""
        for trajectory, _ in rollouts:
            # Extract safe state-action pairs
            obs_to_keep = trajectory["obs_to_keep"].squeeze(0)  # Remove batch dimension
            kept_actions = trajectory["kept_action"].squeeze(0)
            kept_action_masks = trajectory["kept_action_mask"].squeeze(0)

            positive_samples = TensorDict({
                "observation": obs_to_keep,
                "action": kept_actions,
                "action_mask": kept_action_masks
            }, batch_size=[obs_to_keep.shape[0]])
            # Extract corrected state-action pairs
            obs_to_correct = trajectory["obs_to_correct"].squeeze(0)  # Remove batch dimension
            corrected_actions = trajectory["corrected_action"].squeeze(0)
            corrected_action_masks = trajectory["corrected_action_mask"].squeeze(0)
            negative_samples = TensorDict({
                "observation": obs_to_correct,
                "action": corrected_actions,
                "action_mask": corrected_action_masks
            }, batch_size=[obs_to_correct.shape[0]])
            self.add_samples(positive_samples, negative_samples)

    def empty(self):
        """Clear both buffers."""
        self.positive_buffer.empty()
        self.negative_buffer.empty()