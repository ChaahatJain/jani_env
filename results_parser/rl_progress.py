import argparse
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
	"Timestep",
	"Elapsed(s)",
	"MeanReward",
	"GoalFrac",
	"AvoidFrac",
	"Episodes",
}


def slugify(value: str) -> str:
	slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
	return slug or "training_progress"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot RL training progress from a checkpoint CSV log."
	)
	parser.add_argument(
		"csv_path",
		type=Path,
		help="Path to checkpoint CSV log (for example results_parser/test/checkpoint_log.csv).",
	)
	parser.add_argument(
		"--title",
		default="RL training progress",
		help="Title shown at the top of the plot.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output image filename (for example plots/rl_progress_run42.png).",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Also display the plot window after saving.",
	)
	return parser.parse_args()


def default_output_path(title: str) -> Path:
	return Path(f"rl_progress_{slugify(title)}.png")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists() or not csv_path.is_file():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
	if missing:
		raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
	return df


def main() -> None:
	args = parse_args()
	df = load_dataframe(args.csv_path)
	output_path = args.output if args.output is not None else default_output_path(args.title)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	episodes = df["Episodes"].cumsum()

	fig = plt.figure(figsize=(14, 8))
	fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.98)
	gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

	ax0 = fig.add_subplot(gs[0, 0])
	ax0.plot(episodes, df["GoalFrac"], marker="o", color="#3266ad", linewidth=2, label="Goal")
	ax0.plot(episodes, df["AvoidFrac"], marker="o", color="#d9534f", linewidth=2, label="Fail")
	ax0.set_xlabel("Episodes")
	ax0.set_ylabel("Fraction")
	ax0.set_ylim(-0.05, 1.1)
	ax0.set_title("Goal and fail fraction", fontsize=11)
	ax0.legend(fontsize=9)
	ax0.grid(True, alpha=0.3)

	ax1 = fig.add_subplot(gs[0, 1])
	ax1.plot(episodes, df["MeanReward"], marker="o", color="#3b6d11", linewidth=2)
	ax1.set_xlabel("Episodes")
	ax1.set_ylabel("Average reward")
	ax1.set_title("Average reward", fontsize=11)
	ax1.grid(True, alpha=0.3)

	ax2 = fig.add_subplot(gs[1, 0])
	ax2.plot(episodes, df["Elapsed(s)"], marker="o", color="#ba7517", linewidth=2)
	ax2.set_xlabel("Episodes")
	ax2.set_ylabel("Seconds")
	ax2.set_title("Wall-clock time", fontsize=11)
	ax2.grid(True, alpha=0.3)

	ax3 = fig.add_subplot(gs[1, 1])
	ax3.plot(episodes, df["Timestep"], marker="o", color="#73726c", linewidth=2)
	ax3.set_xlabel("Episodes")
	ax3.set_ylabel("Timesteps")
	ax3.set_title("Timesteps used", fontsize=11)
	ax3.grid(True, alpha=0.3)

	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Saved plot to: {output_path}")



if __name__ == "__main__":
	main()
