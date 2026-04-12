import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import FlatObsWrapper
from pipeline import evaluate_policy, load_policy_checkpoint

from custom_gym.lava import CustomLavaGapEnv
from callbacks import save_policy

from pathlib import Path

def create_lava_env(
    size:int = 7,
    num_init: int | None = None,
    n_envs: int = 1
):
    def make_env():
        env = CustomLavaGapEnv(size=size, num_init=num_init, max_steps=100)
        env = FlatObsWrapper(env)
        return env

    if n_envs == 1:
        env = make_env()
    else:
        env = make_vec_env(make_env, n_envs=n_envs)

    return env

env = create_lava_env(size=5, n_envs=4)
model = PPO("MlpPolicy", env, n_steps=256, verbose=1, device="cpu")
model.learn(total_timesteps=25000)

network_params = {
    'input_dim': env.observation_space.shape[0],
    'output_dim': env.action_space.n,
    'hidden_dims': model.policy.net_arch['pi'],
}

save_policy(model.policy, network_params, Path("test_path"), "actor")

actor = load_policy_checkpoint(Path("test_path") / "actor.pth", torch.device("cpu"))
eval_env = CustomLavaGapEnv(size=5, num_init=300, max_steps=100)
eval_env = FlatObsWrapper(eval_env)
eval_result = evaluate_policy(eval_env, actor, init_state_indices=list(range(10)), max_steps=100)
print("Evaluation result:", eval_result)