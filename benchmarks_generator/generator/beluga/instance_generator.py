import json
import random
import argparse
from typing import List, Dict, Any
import os
import subprocess

def generate_instance(
    num_jigs: int,
    num_racks: int,
) -> Dict[str, Any]:
    """
    Generate a beluga problem instance with the given parameters.
    
    Args:
        num_jigs: Number of locations in the linear track
        num_racks: Number of packages to deliver    
    Returns:
        Dictionary representing the problem instance
    """
    if num_jigs < 2:
        raise ValueError("Number of locations must be at least 2")
    
    if num_racks > num_jigs:  # Exclude first and last locations
        raise ValueError("Too many racks for the given number of jigs")
    
    # Generate instance name if not provided
    instance_name = f"beluga_{num_jigs}_{num_racks}"

    config = {
        "name": instance_name,
        "racks": num_racks,
        "trailers": 1,
        "hangars": 1,
        "jigs": num_jigs,
        "belugas": 1,
        "cost-aware": False,
        "move-aware": False,
        "load-aware": False,
        "production-lines": 2
    }
        
     # Create pairs of jigs that should be balanced across lines
    jig_pairs = []
    for i in range(num_jigs // 2):
        jig_pairs.append((i, num_jigs - 1 - i))
    
    # If odd number of jigs, add the middle one separately
    if num_jigs % 2 == 1:
        jig_pairs.append((num_jigs // 2,))
    
    # Distribute pairs between production lines
    pl0 = []
    pl1 = []
    
    for pair in jig_pairs:
        if len(pair) == 2:
            # For pairs, put one in each production line
            if random.choice([True, False]):
                pl0.append(pair[0])
                pl1.append(pair[1])
            else:
                pl0.append(pair[1])
                pl1.append(pair[0])
        else:
            # For single jig (odd case), assign randomly
            if random.choice([True, False]):
                pl0.append(pair[0])
            else:
                pl1.append(pair[0])
    
    config["pl0"] = sorted(pl0)
    config["pl1"] = sorted(pl1)
    
    return config

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
    
    parser.add_argument('--jigs', '-j', type=int, required=True,
                       help='Number of Jigs')
    parser.add_argument('--racks', '-r', type=int, required=True,
                       help='Number of racks')
    parser.add_argument('--output-dir', '-d', type=str, default='description_files',
                       help='Output directory (default: description_files)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        instance = generate_instance(
            num_jigs=args.jigs,
            num_racks=args.racks
        )
        
        output_filename = os.path.join(args.output_dir, f"beluga_{args.jigs}_{args.racks}.json")
        
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