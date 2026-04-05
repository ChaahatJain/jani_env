import json
import random
import argparse
from typing import List, Dict, Any
import os
import subprocess

def generate_instance(
    num_locations: int,
    num_packages: int,
    num_icy_locations: int,
    max_speed: int
) -> Dict[str, Any]:
    """
    Generate a transportation problem instance with the given parameters.
    
    Args:
        num_locations: Number of locations in the linear track
        num_packages: Number of packages to deliver
        num_icy_locations: Number of icy locations (placed in the middle)
        max_speed: Maximum speed of the truck in any direction
    
    Returns:
        Dictionary representing the problem instance
    """
    if num_locations < 2:
        raise ValueError("Number of locations must be at least 2")
    
    if num_icy_locations > num_locations - 2:  # Exclude first and last locations
        raise ValueError("Too many icy locations for the given number of locations")
    
    # Generate instance name if not provided
    instance_name = f"one_way_line_{num_packages}_{num_locations}"
    
    truck_capacity = num_packages
    
    # Generate locations with linear connections
    locations = []
    for i in range(num_locations):
        roads = [{"to": i + 1}] if i < num_locations - 1 else []
        locations.append({"id": i, "roads": roads})
    
    # Goal is always the last location
    goal = num_locations - 1
    
    # Generate bounds for the last few locations
    bounds = []
    for i in range(max(0, num_locations - max_speed), num_locations):
        bound_value = num_locations - i
        bounds.append({"id": i, "bound": bound_value})
    
    # Generate trucks
    trucks = [{"start": 0, "end": goal, "capacity": truck_capacity}]
    
    # Generate packages (all starting at 0, ending at goal)
    packages = []
    for _ in range(num_packages):
        packages.append({"start": 0, "end": goal, "goal": goal})
    
    # Generate icy locations (randomly in the middle, excluding first and last)
    possible_icy_locations = list(range(1, num_locations - 1))
    icy = random.sample(possible_icy_locations, min(num_icy_locations, len(possible_icy_locations)))
    
    return {
        "name": instance_name,
        "locations": locations,
        "goal": goal,
        "bounds": bounds,
        "trucks": trucks,
        "packages": packages,
        "icy": sorted(icy),
        "speed": max_speed
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
            "args": ["./generate_model.sh", "det", instance_path, "0", "0", "0", "0"]
        },
        {
            "name": "Non-deterministic variant (icy and package dropping) without parking",
            "args": ["./generate_model.sh", "non_det_no_park", instance_path, "0.1", "0.1", "0", "1"]
        },
        {
            "name": "Non-deterministic variant (icy and package dropping) with parking",
            "args": ["./generate_model.sh", "non_det_with_park", instance_path, "0.1", "0.1", "1", "1"]
        }
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
    parser.add_argument('--icy', '-i', type=int, required=True,
                       help='Number of icy locations')
    parser.add_argument('--speed', '-s', type=int, required=True,
                        help="Max speed of truck")
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
            num_packages=args.packages,
            num_icy_locations=args.icy,
            max_speed = args.speed
        )
        
        output_filename = os.path.join(args.output_dir, f"one_way_line_{args.packages}_{args.locations}.json")
        
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