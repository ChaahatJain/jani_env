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
        self.root.geometry("1180x760")

        self.model_path = tk.StringVar(value=str(args.model))
        self.property_path = tk.StringVar(value=str(args.property or args.model))
        self.start_path = tk.StringVar(value=str(args.starts))
        self.policy_path = tk.StringVar(value=str(args.policy if args.policy else ""))
        self.seed = tk.IntVar(value=args.seed)
        self.max_steps = tk.IntVar(value=args.max_steps)
        self.delay_ms = tk.IntVar(value=args.delay_ms)
        self.deterministic = tk.BooleanVar(value=not args.sample)
        self.use_oracle = tk.BooleanVar(value=args.use_oracle)
        self.goal_reward = tk.DoubleVar(value=args.goal_reward)
        self.failure_reward = tk.DoubleVar(value=args.failure_reward)
        self.step_reward = tk.DoubleVar(value=args.step_reward)
        self.cycle_reward = tk.DoubleVar(value=args.cycle_reward)

        self.env = None
        self.policy: TorchActorPolicy | RandomValidPolicy | None = None
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
        right = ttk.Frame(main, width=330)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        paths = ttk.LabelFrame(left, text="Files", padding=10)
        paths.pack(fill=tk.X)
        self._path_row(paths, "Model", self.model_path, [("JANI files", "*.jani"), ("All files", "*.*")], 0)
        self._path_row(paths, "Property", self.property_path, [("JANI files", "*.jani"), ("All files", "*.*")], 1)
        self._path_row(paths, "Starts", self.start_path, [("JANI files", "*.jani"), ("All files", "*.*")], 2)
        self._path_row(paths, "Policy", self.policy_path, [("PyTorch policy", "*.pth"), ("All files", "*.*")], 3)

        controls = ttk.Frame(left, padding=(0, 10, 0, 8))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Load", command=self.load).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=(8, 0))
        self.run_button = ttk.Button(controls, text="Run", command=self.toggle_run)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))
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
        self._number_row(settings, "Goal reward", self.goal_reward, 3)
        self._number_row(settings, "Failure reward", self.failure_reward, 4)
        self._number_row(settings, "Step reward", self.step_reward, 5)
        self._number_row(settings, "Cycle reward", self.cycle_reward, 6)
        ttk.Checkbutton(settings, text="Use oracle", variable=self.use_oracle).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        stats = ttk.LabelFrame(right, text="Current Rollout", padding=10)
        stats.pack(fill=tk.X, pady=(12, 0))
        self.stats_text = tk.Text(stats, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)

        actions = ttk.LabelFrame(right, text="Actions", padding=10)
        actions.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.actions_text = tk.Text(actions, height=16, wrap=tk.NONE, state=tk.DISABLED)
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
            self.stop()
            self.obs, _info = self.env.reset()
            self.done = False
            self.truncated = False
            self.step_count = 0
            self.total_reward = 0.0
            self.last_action = None
            self.last_reward = 0.0
            self.last_probs = None
            self.last_info = {}
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
            if isinstance(self.policy, TorchActorPolicy):
                self.policy.deterministic = bool(self.deterministic.get())
            action_mask = self.env.action_mask().astype(bool)
            action, probs = self.policy.predict(self.obs, action_mask)
            self.obs, reward, self.done, self.truncated, info = self.env.step(action)
            self.step_count += 1
            self.total_reward += float(reward)
            self.last_action = action
            self.last_reward = float(reward)
            self.last_probs = probs
            self.last_info = info
            self.render()
        except Exception as exc:
            self.stop()
            messagebox.showerror("Step failed", str(exc))
            self._set_status(f"Step failed: {exc}")

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

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "run_button"):
            self.run_button.configure(text="Run")

    def render(self) -> None:
        self._render_canvas()
        self._render_stats()
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
        self.canvas.create_text(left, 34, text=action_text, anchor=tk.W, fill="#0f172a", font=("Segoe UI", 15, "bold"))
        self.canvas.create_text(left, 62, text=f"Step {self.step_count}   last reward {self.last_reward:.3f}   total reward {self.total_reward:.3f}", anchor=tk.W, fill="#475569", font=("Segoe UI", 11))

        if self.done:
            banner = "DONE"
            color = "#15803d" if self.last_info.get("reached_goal") else "#b91c1c"
            self.canvas.create_text(right, 34, text=banner, anchor=tk.E, fill=color, font=("Segoe UI", 16, "bold"))
        elif self.truncated or self.step_count >= int(self.max_steps.get()):
            self.canvas.create_text(right, 34, text="TRUNCATED", anchor=tk.E, fill="#b45309", font=("Segoe UI", 16, "bold"))

    def _render_stats(self) -> None:
        if self.layout is None or self.obs is None:
            self._set_text(self.stats_text, "")
            return
        decoded = self.layout.decode(self.obs)
        lines = [
            f"step: {self.step_count}",
            f"done: {self.done}",
            f"truncated: {self.truncated}",
            f"total reward: {self.total_reward:.4f}",
            f"truck position: {decoded['truck_position']}",
            f"truck load: {decoded['truck_load']}",
            f"last capacity diff: {decoded['last_capacity_diff']}",
            f"location loads: {decoded['location_loads']}",
        ]
        if self.last_info:
            lines.append(f"info: {self.last_info}")
        self._set_text(self.stats_text, "\n".join(lines))

    def _render_actions(self) -> None:
        if self.env is None or self.layout is None:
            self._set_text(self.actions_text, "")
            return
        try:
            mask = self.env.action_mask().astype(bool)
        except Exception:
            mask = np.zeros(len(self.layout.action_names), dtype=bool)

        lines = []
        for idx, name in enumerate(self.layout.action_names):
            legal = "legal" if idx < len(mask) and mask[idx] else "blocked"
            prob = ""
            if self.last_probs is not None and idx < len(self.last_probs):
                prob = f"  p={self.last_probs[idx]:.3f}"
            chosen = " <- last" if self.last_action == idx else ""
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--delay_ms", type=int, default=350)
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
