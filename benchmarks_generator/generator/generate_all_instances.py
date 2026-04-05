import os
import subprocess
benchmarks = ["beluga", "blocksworld", "bouncing_ball", "cart_pole", "inverted_pendulum", "stopping_car", "one_way_line", "two_way_line", "transport"]
for benchmark in benchmarks:
    print("Benchmark", benchmark)
    result = subprocess.run(['python', f'{benchmark}/generate_all_instances.py'], cwd=os.path.dirname(__file__), capture_output=True, text=True)
    print(result.stdout)