import json
import random
import argparse
from typing import List, Dict, Any
import os
import subprocess

def generate_instance(
    num_blocks: int,
    table_limit: int
):
    """
    Generate a blocksworld problem instance with the given parameters.
    
    Args:
        num_blocks: Number of blocks
        table_limit: Maximum number of blocks that can be on the table safely    
    Returns:
        Dictionary representing the problem instance
    """
    if num_blocks < 2:
        raise ValueError("Number of locations must be at least 2")
    
    if table_limit >= num_blocks:  # Exclude first and last locations
        raise ValueError("No unsafe region since table can have all blocks")
    
    name = f"blocksworld_{num_blocks}_{table_limit}"
    print(name)
    result = subprocess.run(
                ["./generate_model.sh", str(num_blocks), str(table_limit)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per command
            )
    print("Here")
    success = result.returncode == 0
    if success:
        print(f"✓ {name} completed successfully")
    else:
        print(f"✗ {name} failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")
    
    result = subprocess.run(
        ["./generate_networks.sh", "cost_ignore", "--cost-ignoring-nn"],
                capture_output=True,
                text=True,
                timeout=300
    )
    success = result.returncode == 0
    if success:
        print(f"✓ {name} cost ignoring network interface completed successfully")
    else:
        print(f"✗ {name} cost ignoring network interface failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")
            
    result = subprocess.run(
        ["./generate_networks.sh", "cost_aware"],
        capture_output=True,
        text=True,
        timeout=300
    )
    success = result.returncode == 0
    if success:
        print(f"✓ {name} cost aware network interface completed successfully")
    else:
        print(f"✗ {name} cost aware network interface failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")

    result = subprocess.run(
        ["./generate_learning.sh", "cost_ignore", "--cost-ignoring-nn"],
                capture_output=True,
                text=True,
                timeout=300
    )
    success = result.returncode == 0
    if success:
        print(f"✓ {name} random state generation completed successfully")
    else:
        print(f"✗ {name} random state generation failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")
    return 

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
    parser = argparse.ArgumentParser(description='Generate blocksworld problem instances')
    
    parser.add_argument('--blocks', '-b', type=int, required=True,
                       help='Number of blocks')
    parser.add_argument('--table', '-t', type=int, required=True,
                       help='Table limit')
    parser.add_argument('--seed', '-s', type=int, default=2020, 
                       help='Seed')    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        print("Here")
        instance = generate_instance(
            num_blocks=args.blocks,
            table_limit=args.table
        )        
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