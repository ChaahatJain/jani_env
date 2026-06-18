import argparse
import json
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "benchmarks_generator" / "benchmarks" / "transport_" / "linetrack_15_10" / "model.jani"
DEFAULT_STARTS = ROOT / "benchmarks_generator" / "benchmarks" / "transport_" / "linetrack_15_10" / "pa_model_random_starts_100000.jani"
DEFAULT_POLICY = ROOT / "artifacts" / "learning" / "transport_linetrack_15_10" / "mask_ppo_best_params_seed0" / "models" / "final_actor.pth"


def add_container_jani_to_path() -> None:
    candidate = Path.cwd()
    if not (candidate / "jani").is_dir():
        return
    if not (candidate / "jani" / "engine" / "build").is_dir():
        return
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


add_container_jani_to_path()


@dataclass(frozen=True)
class TransportLayout:
    constant_names: list[str]
    variable_names: list[str]
    action_names: list[str]
    location_vars: list[str]
    truck_var: str | None
    truck_load_var: str | None
    capacity_diff_var: str | None
    num_locations: int
    num_packages: int

    @classmethod
    def from_model(cls, model_path: Path) -> "TransportLayout":
        with model_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        constant_names = [item["name"] for item in data.get("constants", [])]
        variable_names = [item["name"] for item in data.get("variables", [])]
        action_names = [item["name"] for item in data.get("actions", [])]
        constants = {item["name"]: item.get("value") for item in data.get("constants", [])}

        def location_index(name: str) -> int:
            return int(name.rsplit("_", 1)[1])

        location_vars = sorted(
            [name for name in variable_names if name.startswith("location_load_")],
            key=location_index,
        )
        truck_vars = [name for name in variable_names if name.startswith("truck_") and not name.startswith("truck_load_")]
        truck_load_vars = [name for name in variable_names if name.startswith("truck_load_")]
        capacity_diff_vars = [name for name in variable_names if name == "last_capacity_diff"]

        return cls(
            constant_names=constant_names,
            variable_names=variable_names,
            action_names=action_names,
            location_vars=location_vars,
            truck_var=truck_vars[0] if truck_vars else None,
            truck_load_var=truck_load_vars[0] if truck_load_vars else None,
            capacity_diff_var=capacity_diff_vars[0] if capacity_diff_vars else None,
            num_locations=int(constants.get("num_locations") or len(location_vars)),
            num_packages=int(constants.get("num_packages") or 0),
        )

    def value(self, obs: np.ndarray, name: str | None, default: int = 0) -> int:
        if name is None:
            return default
        try:
            offset = len(self.constant_names) + self.variable_names.index(name)
        except ValueError:
            return default
        return int(round(float(obs[offset])))

    def decode(self, obs: np.ndarray) -> dict:
        return {
            "location_loads": [self.value(obs, name) for name in self.location_vars],
            "truck_position": self.value(obs, self.truck_var),
            "truck_load": self.value(obs, self.truck_load_var),
            "last_capacity_diff": self.value(obs, self.capacity_diff_var),
        }


class RandomValidPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def predict(self, obs: np.ndarray, action_mask: np.ndarray) -> tuple[int, np.ndarray | None]:
        del obs
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            return 0, None
        return int(self.rng.choice(valid_actions.tolist())), None


class TorchActorPolicy:
    def __init__(self, policy_path: Path, device: str = "cpu", deterministic: bool = True) -> None:
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.model = self._load_actor(policy_path).to(self.device)
        self.model.eval()

    def predict(self, obs: np.ndarray, action_mask: np.ndarray) -> tuple[int, np.ndarray]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask.astype(bool), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(obs_tensor).squeeze(0)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(logits, dim=-1)
            if self.deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.distributions.Categorical(probs=probs).sample().item()
        return int(action), probs.detach().cpu().numpy()

    def _load_actor(self, policy_path: Path) -> torch.nn.Module:
        self._install_numpy_checkpoint_aliases()
        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        input_dim = int(checkpoint.get("input_dim"))
        output_dim = int(checkpoint.get("output_dim"))
        hidden_dims = list(checkpoint.get("hidden_dims", [64, 64]))

        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, int(hidden_dim)))
            layers.append(torch.nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(last_dim, output_dim))

        actor = torch.nn.Sequential(*layers)
        actor.load_state_dict(self._actor_state_dict(state_dict, hidden_dims), strict=True)
        return actor

    @staticmethod
    def _actor_state_dict(state_dict: dict, hidden_dims: list[int]) -> dict:
        if any(key.startswith("mlp_extractor.policy_net.") for key in state_dict):
            actor_state = {}
            for idx, _hidden_dim in enumerate(hidden_dims):
                src = f"mlp_extractor.policy_net.{idx * 2}"
                dst = f"{idx * 2}"
                actor_state[f"{dst}.weight"] = state_dict[f"{src}.weight"]
                actor_state[f"{dst}.bias"] = state_dict[f"{src}.bias"]
            out_idx = len(hidden_dims) * 2
            actor_state[f"{out_idx}.weight"] = state_dict["action_net.weight"]
            actor_state[f"{out_idx}.bias"] = state_dict["action_net.bias"]
            return actor_state

        if any(key.startswith("model.") for key in state_dict):
            return {key.removeprefix("model."): value for key, value in state_dict.items()}

        return state_dict

    @staticmethod
    def _install_numpy_checkpoint_aliases() -> None:
        try:
            import numpy
            import numpy.core

            sys.modules.setdefault("numpy._core", numpy.core)
            if hasattr(numpy.core, "multiarray"):
                sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
            if hasattr(numpy.core, "numeric"):
                sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
        except Exception:
            pass


class TransportPolicyGui:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("Transport Policy Simulator")
        self.root.geometry("1220x820")

        self.model_path = tk.StringVar(value=str(args.model))
        self.property_path = tk.StringVar(value=str(args.property or args.model))
        self.start_path = tk.StringVar(value=str(args.starts))
        self.policy_path = tk.StringVar(value=str(args.policy if args.policy else ""))
        self.repaired_policy_path = tk.StringVar(value=str(args.repaired_policy if args.repaired_policy else ""))
        self.seed = tk.IntVar(value=args.seed)
        self.max_steps = tk.IntVar(value=args.max_steps)
        self.delay_ms = tk.IntVar(value=args.delay_ms)
        self.unsafe_search_attempts = tk.IntVar(value=args.unsafe_search_attempts)
        self.deterministic = tk.BooleanVar(value=not args.sample)
        self.use_oracle = tk.BooleanVar(value=args.use_oracle)
        self.goal_reward = tk.DoubleVar(value=args.goal_reward)
        self.failure_reward = tk.DoubleVar(value=args.failure_reward)
        self.step_reward = tk.DoubleVar(value=args.step_reward)
        self.cycle_reward = tk.DoubleVar(value=args.cycle_reward)

        self.env = None
        self.policy: TorchActorPolicy | RandomValidPolicy | None = None
        self.repaired_policy: TorchActorPolicy | None = None
        self.loaded_repaired_policy_path: Path | None = None
        self.rollout_policy_override: TorchActorPolicy | None = None
        self.rollout_policy_label = "original"
        self.layout: TransportLayout | None = None
        self.obs: np.ndarray | None = None
        self.done = False
        self.truncated = False
        self.running = False
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.last_probs: np.ndarray | None = None
        self.last_info: dict = {}
        self.rollout_history: list[dict] = []
        self.last_fault_error: str | None = None
        self.searching_unsafe = False
        self.search_attempt = 0

        self._build_ui()
        self._set_status("Ready. Load the simulator to begin.")

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(2, 2))

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        paths = ttk.LabelFrame(left, text="Files", padding=10)
        paths.pack(fill=tk.X)
        self._path_row(paths, "Model", self.model_path, [("JANI files", "*.jani"), ("All files", "*.*")], 0)
        self._path_row(paths, "Property", self.property_path, [("JANI files", "*.jani"), ("All files", "*.*")], 1)
        self._path_row(paths, "Starts", self.start_path, [("JANI files", "*.jani"), ("All files", "*.*")], 2)
        self._path_row(paths, "Policy", self.policy_path, [("PyTorch policy", "*.pth"), ("All files", "*.*")], 3)
        self._path_row(paths, "Repaired", self.repaired_policy_path, [("PyTorch policy", "*.pth"), ("All files", "*.*")], 4)

        controls = ttk.Frame(left, padding=(0, 10, 0, 8))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Load", command=self.load).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=(8, 0))
        self.run_button = ttk.Button(controls, text="Run", command=self.toggle_run)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))
        self.find_unsafe_button = ttk.Button(controls, text="Find Unsafe Path", command=self.find_unsafe_path)
        self.find_unsafe_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Compare Repair", command=self.compare_repaired_policy).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Rollout Repair", command=self.rollout_repaired_policy).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Original From Here", command=self.rollout_original_from_here).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(controls, text="Deterministic", variable=self.deterministic).pack(side=tk.LEFT, padx=(18, 0))

        self.canvas = tk.Canvas(left, background="#f8fafc", highlightthickness=1, highlightbackground="#cbd5e1")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, pady=(8, 0))

        settings = ttk.LabelFrame(right, text="Run Settings", padding=10)
        settings.pack(fill=tk.X)
        self._number_row(settings, "Seed", self.seed, 0)
        self._number_row(settings, "Max steps", self.max_steps, 1)
        self._number_row(settings, "Delay ms", self.delay_ms, 2)
        self._number_row(settings, "Search tries", self.unsafe_search_attempts, 3)
        self._number_row(settings, "Goal reward", self.goal_reward, 4)
        self._number_row(settings, "Failure reward", self.failure_reward, 5)
        self._number_row(settings, "Step reward", self.step_reward, 6)
        self._number_row(settings, "Cycle reward", self.cycle_reward, 7)
        ttk.Checkbutton(settings, text="Use oracle", variable=self.use_oracle).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        stats = ttk.LabelFrame(right, text="Current Rollout", padding=10)
        stats.pack(fill=tk.X, pady=(12, 0))
        self.stats_text = tk.Text(stats, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)

        faults = ttk.LabelFrame(right, text="Path Faults", padding=10)
        faults.pack(fill=tk.X, pady=(12, 0))
        self.faults_text = tk.Text(faults, height=9, wrap=tk.WORD, state=tk.DISABLED)
        self.faults_text.pack(fill=tk.X)

        actions = ttk.LabelFrame(right, text="Actions", padding=10)
        actions.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.actions_text = tk.Text(actions, height=11, wrap=tk.NONE, state=tk.DISABLED)
        self.actions_text.pack(fill=tk.BOTH, expand=True)

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, filetypes: list[tuple[str, str]], row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky=tk.EW, padx=8, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(var, filetypes)).grid(row=row, column=2, sticky=tk.E, pady=2)
        parent.columnconfigure(1, weight=1)

    def _number_row(self, parent: ttk.Frame, label: str, var: tk.Variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky=tk.E, pady=2)
        parent.columnconfigure(1, weight=1)

    def _browse(self, var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        chosen = filedialog.askopenfilename(initialdir=str(ROOT), filetypes=filetypes)
        if chosen:
            var.set(chosen)

    def load(self) -> None:
        self.stop()
        try:
            model = Path(self.model_path.get())
            property_text = self.property_path.get().strip()
            starts_text = self.start_path.get().strip()
            property_file = Path(property_text) if property_text else model
            starts = Path(starts_text) if starts_text else None
            self.layout = TransportLayout.from_model(model)

            from jani.env import JANIEnv

            self.env = JANIEnv(
                jani_model_path=str(model),
                jani_property_path=str(property_file),
                start_states_path=str(starts) if starts is not None else "",
                objective_path="",
                failure_property_path="",
                seed=int(self.seed.get()),
                goal_reward=float(self.goal_reward.get()),
                failure_reward=float(self.failure_reward.get()),
                use_oracle=bool(self.use_oracle.get()),
                unsafe_reward=-0.5,
                step_reward=float(self.step_reward.get()),
                cycle_reward=float(self.cycle_reward.get()),
            )

            policy_file = Path(self.policy_path.get()) if self.policy_path.get() else None
            if policy_file and policy_file.exists():
                self.policy = TorchActorPolicy(policy_file, deterministic=bool(self.deterministic.get()))
            else:
                self.policy = RandomValidPolicy(seed=int(self.seed.get()))
                if policy_file:
                    self._set_status("Policy file not found; using a random valid-action policy.")

            self.repaired_policy = None
            self.loaded_repaired_policy_path = None
            self.rollout_policy_override = None
            self.rollout_policy_label = "original"

            self.reset()
            self._set_status("Loaded simulator.")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self._set_status(f"Load failed: {exc}")

    def reset(self) -> None:
        if self.env is None:
            self._set_status("Load the simulator first.")
            return
        try:
            self.stop(cancel_search=False)
            self.obs, _info = self.env.reset()
            self.done = False
            self.truncated = False
            self.step_count = 0
            self.total_reward = 0.0
            self.last_action = None
            self.last_reward = 0.0
            self.last_probs = None
            self.last_info = {}
            self.rollout_history = []
            self.last_fault_error = None
            self.rollout_policy_override = None
            self.rollout_policy_label = "original"
            self.render()
            self._set_status("Reset to a start state.")
        except Exception as exc:
            messagebox.showerror("Reset failed", str(exc))
            self._set_status(f"Reset failed: {exc}")

    def step_once(self) -> None:
        if self.env is None or self.policy is None or self.obs is None:
            self._set_status("Load the simulator first.")
            return
        if self.done or self.truncated or self.step_count >= int(self.max_steps.get()):
            self._set_status("Episode is finished. Press Reset to start another rollout.")
            self.stop()
            return

        try:
            active_policy = self.rollout_policy_override or self.policy
            if isinstance(active_policy, TorchActorPolicy):
                active_policy.deterministic = True if self.rollout_policy_override is not None else bool(self.deterministic.get())
            action_mask = self.env.action_mask().astype(bool)
            pre_obs = np.array(self.obs, copy=True)
            action, probs = active_policy.predict(self.obs, action_mask)
            is_fault, fault_error = self._classify_fault(pre_obs, action)
            self.obs, reward, self.done, self.truncated, info = self.env.step(action)
            self.step_count += 1
            self.total_reward += float(reward)
            self.last_action = action
            self.last_reward = float(reward)
            self.last_probs = probs
            self.last_info = info
            self.last_fault_error = fault_error
            self._record_step(pre_obs, action_mask, action, reward, info, is_fault)
            self.render()
        except Exception as exc:
            self.stop()
            messagebox.showerror("Step failed", str(exc))
            self._set_status(f"Step failed: {exc}")

    def _classify_fault(self, obs: np.ndarray, action: int) -> tuple[bool, str | None]:
        if self.env is None or not bool(self.use_oracle.get()):
            return False, None
        if not hasattr(self.env, "is_state_action_fault"):
            return False, "Environment does not expose state/action fault checks."
        try:
            return bool(self.env.is_state_action_fault(obs, action)), None
        except Exception as exc:
            return False, str(exc)

    def _record_step(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        info: dict,
        is_fault: bool,
    ) -> None:
        if self.layout is None:
            return
        decoded = self.layout.decode(obs)
        action_name = self.layout.action_names[action] if action < len(self.layout.action_names) else str(action)
        self.rollout_history.append(
            {
                "step": self.step_count,
                "observation": np.array(obs, copy=True),
                "action_mask": np.array(action_mask, copy=True),
                "action": int(action),
                "action_name": action_name,
                "reward": float(reward),
                "info": dict(info),
                "is_fault": bool(is_fault),
                "truck_position": decoded["truck_position"],
                "truck_load": decoded["truck_load"],
                "location_loads": decoded["location_loads"],
                "last_capacity_diff": decoded["last_capacity_diff"],
            }
        )

    def _get_repaired_policy(self) -> TorchActorPolicy:
        path_text = self.repaired_policy_path.get().strip()
        if not path_text:
            raise ValueError("Choose a repaired policy checkpoint first.")
        policy_path = Path(path_text)
        if not policy_path.exists():
            raise FileNotFoundError(f"Repaired policy file not found: {policy_path}")
        if self.repaired_policy is None or self.loaded_repaired_policy_path != policy_path:
            self.repaired_policy = TorchActorPolicy(policy_path, deterministic=True)
            self.loaded_repaired_policy_path = policy_path
        self.repaired_policy.deterministic = True
        return self.repaired_policy

    def compare_repaired_policy(self) -> None:
        if self.layout is None:
            self._set_status("Load the simulator first.")
            return

        fault_step = next((item for item in reversed(self.rollout_history) if item.get("is_fault")), None)
        if fault_step is None:
            self._set_status("Find or step into a fault before comparing the repaired policy.")
            return

        try:
            repaired_policy = self._get_repaired_policy()
            action, probs = repaired_policy.predict(
                fault_step["observation"],
                fault_step["action_mask"].astype(bool),
            )
            action_name = self.layout.action_names[action] if action < len(self.layout.action_names) else str(action)
            repaired_is_fault, fault_error = self._classify_fault(fault_step["observation"], action)
            fault_step["repaired_action"] = int(action)
            fault_step["repaired_action_name"] = action_name
            fault_step["repaired_action_prob"] = float(probs[action]) if probs is not None and action < len(probs) else None
            fault_step["repaired_is_fault"] = bool(repaired_is_fault)
            fault_step["repaired_fault_error"] = fault_error
            self.render()

            old_action = fault_step["action_name"]
            verdict = "still faulty" if repaired_is_fault else "not faulty"
            self._set_status(f"Repaired policy chose {action_name} instead of {old_action}; repaired action is {verdict}.")
        except Exception as exc:
            messagebox.showerror("Repair comparison failed", str(exc))
            self._set_status(f"Repair comparison failed: {exc}")

    def rollout_repaired_policy(self) -> None:
        if self.env is None or self.layout is None:
            self._set_status("Load the simulator first.")
            return

        fault_step = next((item for item in reversed(self.rollout_history) if item.get("is_fault")), None)
        if fault_step is None:
            self._set_status("Find or step into a fault before rolling out the repaired policy.")
            return

        try:
            repaired_policy = self._get_repaired_policy()
            self.stop()
            self.obs, _info = self.env.reset(options={"state": fault_step["observation"]})
            self.done = False
            self.truncated = False
            self.step_count = 0
            self.total_reward = 0.0
            self.last_action = None
            self.last_reward = 0.0
            self.last_probs = None
            self.last_info = {}
            self.rollout_history = []
            self.last_fault_error = None
            self.rollout_policy_override = repaired_policy
            self.rollout_policy_label = "repaired"
            self.render()
            self._set_status("Repaired policy loaded at the fault state. Press Step or Run to see what it does next.")
        except Exception as exc:
            messagebox.showerror("Repair rollout failed", str(exc))
            self._set_status(f"Repair rollout failed: {exc}")

    def rollout_original_from_here(self) -> None:
        if self.env is None or self.policy is None or self.obs is None:
            self._set_status("Load the simulator first.")
            return

        try:
            current_state = np.array(self.obs, copy=True)
            self.stop()
            self.obs, _info = self.env.reset(options={"state": current_state})
            self.done = False
            self.truncated = False
            self.step_count = 0
            self.total_reward = 0.0
            self.last_action = None
            self.last_reward = 0.0
            self.last_probs = None
            self.last_info = {}
            self.rollout_history = []
            self.last_fault_error = None
            self.rollout_policy_override = None
            self.rollout_policy_label = "original"
            self.render()
            self._set_status("Original policy loaded at the current state. Press Step or Run to see what it does next.")
        except Exception as exc:
            messagebox.showerror("Original rollout failed", str(exc))
            self._set_status(f"Original rollout failed: {exc}")

    def find_unsafe_path(self) -> None:
        if self.env is None or self.policy is None:
            self._set_status("Load the simulator first.")
            return
        if not bool(self.use_oracle.get()):
            messagebox.showinfo("Oracle required", "Enable Use oracle, press Load, then search again.")
            self._set_status("Unsafe-path search needs the oracle-enabled environment.")
            return
        if getattr(self.env, "_oracle", None) is None:
            messagebox.showinfo("Reload required", "Use oracle was not active when this simulator was loaded. Press Load, then search again.")
            self._set_status("Reload with Use oracle enabled before searching for faults.")
            return

        self.stop()
        self.searching_unsafe = True
        self.search_attempt = 0
        self.find_unsafe_button.configure(state=tk.DISABLED)
        self._set_status("Searching for an unsafe path...")
        self.root.after(1, self._start_unsafe_search_attempt)

    def _start_unsafe_search_attempt(self) -> None:
        if not self.searching_unsafe:
            return
        try:
            max_attempts = max(1, int(self.unsafe_search_attempts.get()))
        except Exception:
            max_attempts = 1
        if self.search_attempt >= max_attempts:
            self._finish_unsafe_search(f"No unsafe path found in {max_attempts} attempts.")
            return

        self.search_attempt += 1
        self.reset()
        self._set_status(f"Searching for unsafe path: attempt {self.search_attempt}/{max_attempts}")
        self.root.after(1, self._unsafe_search_step)

    def _unsafe_search_step(self) -> None:
        if not self.searching_unsafe:
            return
        if self.env is None or self.policy is None or self.obs is None:
            self._finish_unsafe_search("Search stopped because the simulator is not loaded.")
            return

        if self.done or self.truncated or self.step_count >= int(self.max_steps.get()):
            self.root.after(1, self._start_unsafe_search_attempt)
            return

        self.step_once()

        last_step = self.rollout_history[-1] if self.rollout_history else {}
        found_fault = bool(last_step.get("is_fault"))
        reached_failure = bool(self.last_info.get("reached_fail"))
        if found_fault or reached_failure:
            reason = "fault" if found_fault else "failure"
            self._finish_unsafe_search(
                f"Found unsafe path by {reason} on attempt {self.search_attempt}, step {self.step_count}."
            )
            return

        if self.done or self.truncated or self.step_count >= int(self.max_steps.get()):
            self.root.after(1, self._start_unsafe_search_attempt)
            return

        self.root.after(1, self._unsafe_search_step)

    def _finish_unsafe_search(self, status: str) -> None:
        self.searching_unsafe = False
        if hasattr(self, "find_unsafe_button"):
            self.find_unsafe_button.configure(state=tk.NORMAL)
        self._set_status(status)

    def toggle_run(self) -> None:
        if self.running:
            self.stop()
        else:
            self.running = True
            self.run_button.configure(text="Pause")
            self._run_loop()

    def _run_loop(self) -> None:
        if not self.running:
            return
        self.step_once()
        if self.running and not self.done and not self.truncated and self.step_count < int(self.max_steps.get()):
            self.root.after(max(1, int(self.delay_ms.get())), self._run_loop)
        else:
            self.stop()

    def stop(self, cancel_search: bool = True) -> None:
        self.running = False
        if cancel_search:
            self.searching_unsafe = False
            if hasattr(self, "find_unsafe_button"):
                self.find_unsafe_button.configure(state=tk.NORMAL)
        if hasattr(self, "run_button"):
            self.run_button.configure(text="Run")

    def render(self) -> None:
        self._render_canvas()
        self._render_stats()
        self._render_faults()
        self._render_actions()

    def _render_canvas(self) -> None:
        self.canvas.delete("all")
        if self.layout is None or self.obs is None:
            return

        decoded = self.layout.decode(self.obs)
        loads = decoded["location_loads"]
        truck_pos = decoded["truck_position"]
        truck_load = decoded["truck_load"]
        n_locations = max(self.layout.num_locations, len(loads), 1)

        width = max(self.canvas.winfo_width(), 900)
        height = max(self.canvas.winfo_height(), 420)
        left = 70
        right = width - 70
        y = height * 0.48
        spacing = (right - left) / max(n_locations - 1, 1)

        self.canvas.create_line(left, y, right, y, width=4, fill="#94a3b8")
        for idx in range(n_locations):
            x = left + idx * spacing
            fill = "#0f766e" if idx == n_locations - 1 else "#ffffff"
            outline = "#0f172a" if idx == truck_pos else "#475569"
            self.canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill=fill, outline=outline, width=3)
            self.canvas.create_text(x, y, text=str(idx), fill="#0f172a" if idx != n_locations - 1 else "#ffffff", font=("Segoe UI", 11, "bold"))
            load = loads[idx] if idx < len(loads) else 0
            self.canvas.create_text(x, y + 42, text=f"pkg {load}", fill="#334155", font=("Segoe UI", 10))

        fault_counts: dict[int, int] = {}
        for item in self.rollout_history:
            if item.get("is_fault"):
                loc = int(item.get("truck_position", 0))
                fault_counts[loc] = fault_counts.get(loc, 0) + 1
        for loc, count in fault_counts.items():
            if 0 <= loc < n_locations:
                x = left + loc * spacing
                self.canvas.create_oval(x - 26, y - 26, x + 26, y + 26, outline="#dc2626", width=4)
                label = "fault" if count == 1 else f"fault x{count}"
                self.canvas.create_text(x, y - 40, text=label, fill="#b91c1c", font=("Segoe UI", 10, "bold"))

        truck_x = left + min(max(truck_pos, 0), n_locations - 1) * spacing
        truck_y = y - 78
        self.canvas.create_rectangle(truck_x - 34, truck_y - 18, truck_x + 34, truck_y + 18, fill="#2563eb", outline="#1e3a8a", width=2)
        self.canvas.create_rectangle(truck_x - 22, truck_y - 36, truck_x + 22, truck_y - 18, fill="#3b82f6", outline="#1e3a8a", width=2)
        self.canvas.create_oval(truck_x - 25, truck_y + 12, truck_x - 9, truck_y + 28, fill="#0f172a")
        self.canvas.create_oval(truck_x + 9, truck_y + 12, truck_x + 25, truck_y + 28, fill="#0f172a")
        self.canvas.create_text(truck_x, truck_y - 52, text=f"truck load {truck_load}", fill="#0f172a", font=("Segoe UI", 11, "bold"))

        action_text = "Action: none yet"
        if self.last_action is not None and self.layout.action_names:
            action_name = self.layout.action_names[self.last_action]
            action_text = f"Action: {self.last_action}  {action_name}"
        last_fault = bool(self.rollout_history and self.rollout_history[-1].get("is_fault"))
        action_color = "#b91c1c" if last_fault else "#0f172a"
        if last_fault:
            action_text = f"FAULT  {action_text}"
        self.canvas.create_text(left, 34, text=action_text, anchor=tk.W, fill=action_color, font=("Segoe UI", 15, "bold"))
        self.canvas.create_text(left, 62, text=f"Step {self.step_count}   last reward {self.last_reward:.3f}   total reward {self.total_reward:.3f}", anchor=tk.W, fill="#475569", font=("Segoe UI", 11))

        self._render_fault_timeline(left, right, height)

        if self.done:
            banner = "DONE"
            color = "#15803d" if self.last_info.get("reached_goal") else "#b91c1c"
            self.canvas.create_text(right, 34, text=banner, anchor=tk.E, fill=color, font=("Segoe UI", 16, "bold"))
        elif self.truncated or self.step_count >= int(self.max_steps.get()):
            self.canvas.create_text(right, 34, text="TRUNCATED", anchor=tk.E, fill="#b45309", font=("Segoe UI", 16, "bold"))

    def _render_fault_timeline(self, left: int, right: int, height: int) -> None:
        if not self.rollout_history:
            return
        y = height - 48
        window = self.rollout_history[-80:]
        self.canvas.create_text(left, y - 24, text=f"Path steps {window[0]['step']}..{window[-1]['step']}", anchor=tk.W, fill="#475569", font=("Segoe UI", 10))
        self.canvas.create_line(left, y, right, y, width=2, fill="#cbd5e1")
        denom = max(len(window) - 1, 1)
        for idx, item in enumerate(window):
            x = left + (right - left) * idx / denom
            fill = "#64748b"
            radius = 4
            if item["info"].get("reached_goal"):
                fill = "#15803d"
                radius = 6
            if item["info"].get("reached_fail"):
                fill = "#991b1b"
                radius = 6
            if item.get("is_fault"):
                fill = "#dc2626"
                radius = 7
            if idx == len(window) - 1:
                self.canvas.create_oval(x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2, outline="#0f172a", width=2)
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="")

    def _render_stats(self) -> None:
        if self.layout is None or self.obs is None:
            self._set_text(self.stats_text, "")
            return
        decoded = self.layout.decode(self.obs)
        lines = [
            f"step: {self.step_count}",
            f"policy: {self.rollout_policy_label}",
            f"done: {self.done}",
            f"truncated: {self.truncated}",
            f"total reward: {self.total_reward:.4f}",
            f"faults found: {sum(1 for item in self.rollout_history if item.get('is_fault'))}",
            f"truck position: {decoded['truck_position']}",
            f"truck load: {decoded['truck_load']}",
            f"last capacity diff: {decoded['last_capacity_diff']}",
            f"location loads: {decoded['location_loads']}",
        ]
        if self.last_fault_error:
            lines.append(f"fault analysis: {self.last_fault_error}")
        if self.last_info:
            lines.append(f"info: {self.last_info}")
        self._set_text(self.stats_text, "\n".join(lines))

    def _render_faults(self) -> None:
        if self.env is None:
            self._set_text(self.faults_text, "")
            return
        if not bool(self.use_oracle.get()):
            self._set_text(self.faults_text, "Fault analysis unavailable. Reload with Use oracle enabled.")
            return
        if not self.rollout_history:
            self._set_text(self.faults_text, "No rollout steps yet.")
            return

        faults = [item for item in self.rollout_history if item.get("is_fault")]
        lines = [
            f"path length: {len(self.rollout_history)}",
            f"fault steps: {len(faults)}",
        ]
        if self.last_info.get("reached_fail"):
            lines.append("termination: failure")
        elif self.last_info.get("reached_goal"):
            lines.append("termination: goal")
        elif self.truncated or self.step_count >= int(self.max_steps.get()):
            lines.append("termination: truncated")

        if faults:
            lines.append("")
            lines.append("fault details:")
            for item in faults[-6:]:
                line = (
                    f"step {item['step']}: loc {item['truck_position']}, "
                    f"load {item['truck_load']}, action {item['action']} {item['action_name']}"
                )
                if "repaired_action" in item:
                    prob = item.get("repaired_action_prob")
                    prob_text = f" p={prob:.3f}" if prob is not None else ""
                    status = "FAULT" if item.get("repaired_is_fault") else "safe"
                    line += (
                        f" | repaired -> {item['repaired_action']} "
                        f"{item['repaired_action_name']}{prob_text} [{status}]"
                    )
                lines.append(line)

        lines.append("")
        lines.append("recent path:")
        for item in self.rollout_history[-8:]:
            tags = []
            if item.get("is_fault"):
                tags.append("FAULT")
            if item["info"].get("reached_fail"):
                tags.append("FAIL")
            if item["info"].get("reached_goal"):
                tags.append("GOAL")
            suffix = f" [{' '.join(tags)}]" if tags else ""
            lines.append(
                f"{item['step']}: loc {item['truck_position']} -> "
                f"{item['action_name']} r={item['reward']:.3f}{suffix}"
            )
        self._set_text(self.faults_text, "\n".join(lines))

    def _render_actions(self) -> None:
        if self.env is None or self.layout is None:
            self._set_text(self.actions_text, "")
            return
        try:
            mask = self.env.action_mask().astype(bool)
        except Exception:
            mask = np.zeros(len(self.layout.action_names), dtype=bool)

        lines = []
        last_step = self.rollout_history[-1] if self.rollout_history else {}
        for idx, name in enumerate(self.layout.action_names):
            legal = "legal" if idx < len(mask) and mask[idx] else "blocked"
            prob = ""
            if self.last_probs is not None and idx < len(self.last_probs):
                prob = f"  p={self.last_probs[idx]:.3f}"
            chosen = " <- last" if self.last_action == idx else ""
            if chosen and last_step.get("is_fault"):
                chosen += " FAULT"
            lines.append(f"{idx}: {name:<18} {legal:<7}{prob}{chosen}")
        self._set_text(self.actions_text, "\n".join(lines))

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status.set(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple GUI simulator for a trained transport Mask PPO actor.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--property", type=Path, default=None)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY if DEFAULT_POLICY.exists() else None)
    parser.add_argument("--repaired-policy", "--repaired_policy", dest="repaired_policy", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--delay_ms", type=int, default=350)
    parser.add_argument("--unsafe-search-attempts", "--unsafe_search_attempts", dest="unsafe_search_attempts", type=int, default=200)
    parser.add_argument("--sample", action="store_true", help="Sample from the policy distribution instead of taking argmax.")
    parser.add_argument("--use_oracle", action="store_true")
    parser.add_argument("--goal_reward", type=float, default=1.0)
    parser.add_argument("--failure_reward", type=float, default=-10.0)
    parser.add_argument("--step_reward", type=float, default=-0.005)
    parser.add_argument("--cycle_reward", type=float, default=-0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    TransportPolicyGui(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
