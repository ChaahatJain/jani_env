import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSV_PATH = "test/checkpoint_log.csv"

df = pd.read_csv(CSV_PATH)
# Expected columns: checkpoint, timestep, elapsed_seconds, avg_reward,
#                   goal_fraction, failure_fraction, episodes

episodes = df["episodes"].cumsum()  # cumulative episodes as x axis

fig = plt.figure(figsize=(14, 8))
fig.suptitle("RL training progress over episodes - 2 way line (20, 10) det", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(episodes, df["goal_fraction"],    marker="o", color="#3266ad", linewidth=2, label="Goal")
ax0.plot(episodes, df["failure_fraction"], marker="o", color="#d9534f", linewidth=2, label="Fail")
ax0.set_xlabel("Episodes")
ax0.set_ylabel("Fraction")
ax0.set_ylim(-0.05, 1.1)
ax0.set_title("Goal and fail fraction", fontsize=11)
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.3)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(episodes, df["avg_reward"], marker="o", color="#3b6d11", linewidth=2)
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average reward")
ax1.set_title("Average reward", fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(episodes, df["elapsed_seconds"], marker="o", color="#ba7517", linewidth=2)
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Seconds")
ax2.set_title("Wall-clock time", fontsize=11)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(episodes, df["timestep"], marker="o", color="#73726c", linewidth=2)
ax3.set_xlabel("Episodes")
ax3.set_ylabel("Timesteps")
ax3.set_title("Timesteps used", fontsize=11)
ax3.grid(True, alpha=0.3)

plt.savefig("rl_progress.png", dpi=150, bbox_inches="tight")
