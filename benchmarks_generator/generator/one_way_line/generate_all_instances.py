import subprocess
import os
import sys
from typing import List, Tuple

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)

def run_command(command: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per command
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def check_script_exists(script_name: str) -> bool:
    """Check if a script exists and is executable."""
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found")
        return False
    if not os.access(script_name, os.X_OK):
        print(f"Error: {script_name} is not executable")
        return False
    return True

def main():
    """Main function to generate instances and run processing scripts."""
    
    # List of tuples: (locations, packages, icy_locations)
    instances = [
        (10, 15, 2, 3),
        (10, 17, 2, 3),
        (10, 20, 3, 3),
        (20, 30, 4, 5), 
        (25, 40, 5, 5),
        (25, 50, 7, 8),
        (30, 60, 12, 9),
        (35, 70, 15, 10),
        (40, 80, 25, 12),
        # Add more tuples as needed
        # (locations, packages, icy_locations)
    ]
    
    # Check if required scripts exist
    if not check_script_exists("./instance_generator.py"):
        return 1
    
    all_instance_success = True
    generated_instances = []
    
    # Generate all instances
    print("Starting instance generation...")
    for i, (locations, packages, icy, speed) in enumerate(instances, 1):
        print(f"\n{'#'*70}")
        print(f"Generating instance {i}/{len(instances)}: locations={locations}, packages={packages}, icy={icy}, speed={speed}")
        print(f"{'#'*70}")
        
        cmd = [
            "python3", "instance_generator.py",
            "-l", str(locations),
            "-p", str(packages),
            "-i", str(icy),
            "-s", str(speed)
        ]
        
        success = run_command(cmd, f"Instance generation {i} (l={locations}, p={packages}, i={icy}, speed={speed})")
        
        if success:
            # Store the generated instance filename pattern
            instance_file = f"description_files/one_way_line_{locations}_{packages}.json"
            generated_instances.append(instance_file)
            print(f"Generated instance: {instance_file}")
        else:
            all_instance_success = False
            print(f"Failed to generate instance: locations={locations}, packages={packages}, icy={icy}")

if __name__ == "__main__":
    exit(main())