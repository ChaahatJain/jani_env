import json
import random
import argparse
from typing import List, Dict, Any
import os
import subprocess
def generate_instance(
    num_locations: int,
    num_packages: int,
) -> Dict[str, Any]:
    """
    Generate a transportation problem instance with the given parameters.
    
    Args:
        num_locations: Number of locations in the linear track
        num_packages: Number of packages to deliver    
    Returns:
        Dictionary representing the problem instance
    """
    if num_locations < 2:
        raise ValueError("Number of locations must be at least 2")
        
    # Generate instance name if not provided
    instance_name = f"linetrack_{num_packages}_{num_locations}"
    
    truck_capacity = num_packages
    
    # Generate locations with linear connections
    locations = []
    for i in range(num_locations):
        roads = []
        if i < num_locations - 1:
            roads.append({"to": i + 1, "capacity": truck_capacity if i != num_locations - 2 else 1, "label": "drive_forward"})
        if i != 0:
            roads.append({"to": i - 1, "capacity": 0, "label": "drive_backward"})
        locations.append({"id": i, "roads": roads})
    
    # Goal is always the last location
    goal = num_locations - 1
    
    # Generate trucks
    trucks = [{"start": 0, "end": goal, "capacity": truck_capacity}]
    
    # Generate packages (all starting at 0, ending at goal)
    packages = []
    for _ in range(num_packages):
        packages.append({"start": 0, "end": goal, "goal": goal})
        
    return {
        "name": instance_name,
        "drive-actions": ["drive_forward", "drive_backward"],
        "locations": locations,
        "goal": goal,
        "trucks": trucks,
        "packages": packages,
    }

def save_instance(instance: Dict[str, Any], filename: str):
    """Save the instance to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(instance, f, indent=2)

def call_generate_model_commands(instance_path: str):
    """Call the generate_model.sh script with different parameter combinations."""
    
    # Define the command configurations with descriptive names
    commands = [
        {
            "name": "Deterministic variant",
            "args": ["./generate_model.sh", instance_path]
        },
    ]
    
    results = []
    
    try:
        # Check if the script exists and is executable
        if not os.path.exists('./generate_model.sh'):
            print("Error: generate_model.sh script not found")
            return False
        
        if not os.access('./generate_model.sh', os.X_OK):
            print("Error: generate_model.sh is not executable")
            return False
        
        # Run each command
        for command in commands:
            print(f"Running {command['name']} model generation...")
            print(f"Command: {' '.join(command['args'])}")
            
            result = subprocess.run(
                command['args'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per command
            )
            
            success = result.returncode == 0
            results.append({
                "name": command['name'],
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            if success:
                print(f"✓ {command['name']} completed successfully")
            else:
                print(f"✗ {command['name']} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
            
            print("-" * 50)
        
        # Print summary
        print("\n=== MODEL GENERATION SUMMARY ===")
        all_successful = True
        for result in results:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"{result['name']}: {status}")
            if not result['success']:
                all_successful = False
        
        return all_successful
            
    except subprocess.TimeoutExpired:
        print("Error: One of the model generation commands timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error calling generate_model.sh: {e}")
        return False

def main():
    """Command-line interface for generating instances."""
    parser = argparse.ArgumentParser(description='Generate transportation problem instances')
    
    parser.add_argument('--locations', '-l', type=int, required=True,
                       help='Number of locations')
    parser.add_argument('--packages', '-p', type=int, required=True,
                       help='Number of packages')
    parser.add_argument('--output-dir', '-d', type=str, default='description_files',
                       help='Output directory (default: description_files)')
    parser.add_argument('--seed', '-r', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        instance = generate_instance(
            num_locations=args.locations,
            num_packages=args.packages
        )
        
        output_filename = os.path.join(args.output_dir, f"linetrack_{args.packages}_{args.locations}.json")
        
        save_instance(instance, output_filename)
        print(f"Instance successfully generated and saved to {output_filename}")
        call_generate_model_commands(output_filename)
        print(f"Successfully generated the model")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())