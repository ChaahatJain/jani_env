import os
import subprocess

result = subprocess.run(['./generate_model.sh'], cwd=os.path.dirname(__file__), capture_output=True, text=True)