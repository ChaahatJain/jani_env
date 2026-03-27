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
        (4, 2),
        (4, 3),
        (6, 4),
        (6, 5),
        (8, 4),
        (8, 5),
        (8, 7),
        (10, 5),
        (10, 7), 
        (12, 4), 
        (12, 6),
        (15, 5),
        (15, 8), 
        (20, 6),        # Add more tuples as needed
        # (blocks, table_limit)
    ]
    
    # Check if required scripts exist
    if not check_script_exists("./instance_generator.py"):
        return 1
    if not check_script_exists("./generate_networks.sh"):
        return 1
    if not check_script_exists("./generate_learning.sh"):
        return 1
    if not check_script_exists("./generate_verification.sh"):
        return 1
    
    all_instance_success = True
    generated_instances = []
    
    # Generate all instances
    print("Starting instance generation...")
    for i, (blocks, table_limit) in enumerate(instances, 1):
        print(f"\n{'#'*70}")
        print(f"Generating instance {i}/{len(instances)}: blocks={blocks}, table limit={table_limit}")
        print(f"{'#'*70}")
        
        cmd = [
            "python3", "instance_generator.py",
            "-b", str(blocks),
            "-t", str(table_limit),
        ]
        
        success = run_command(cmd, f"Instance generation {i} (blocks={blocks}, table limit={table_limit})")
        
        if success:
            # Store the generated instance filename pattern
            instance_file = f"Blocksworld_{blocks}_{table_limit}.json"
            generated_instances.append(instance_file)
            print(f"Generated instance: {instance_file}")
        else:
            all_instance_success = False
            print(f"Failed to generate instance: blocks={blocks}, table_limit={table_limit}")
    
    # If any instance generation failed, ask whether to continue
    if not all_instance_success:
        print("Some instances failed to generate!")
        response = input("Do you want to continue with network generation and verification? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Exiting...")
            return 1
    
    # Print final summary
    print(f"\n{'#'*70}")
    print("FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"Instances to generate: {len(instances)}")
    print(f"Instances successfully generated: {len(generated_instances)}")
    
    if generated_instances:
        print("\n Generated instances:")
        for instance in generated_instances:
            print(f"  - {instance}")
    
    if all_instance_success:
        print("\n All processes completed successfully!")
        return 0
    else:
        print("\n⚠️  Some processes failed!")
        return 1

if __name__ == "__main__":
    exit(main())