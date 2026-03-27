import json

def extract_start_from_jani(file_path):
    try:
        # Read and parse the JANI file
        with open(file_path, 'r') as file:
            jani_data = json.load(file)
        
        # Look for properties that contain "start"
        if 'properties' in jani_data:
            for property in jani_data['properties']:
                if 'expression' in property:
                    expression = property['expression']
                    if 'start' in expression:
                        start = expression['start']
                        # Recursively search for "start" in the expression
                        start_info = find_key_in_structure(expression, 'values')
                        if start_info:
                            print("Found 'start' in property expression:")
                            # print(json.dumps(start_info, indent=2))
                            return start_info
        
        print("No 'start' found in properties")
        return None
        
    except Exception as e:
        print(f"Error reading JANI file: {e}")
        return None

def find_key_in_structure(data, target_key):
    """Recursively search for a key in nested JSON structure"""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            result = find_key_in_structure(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_in_structure(item, target_key)
            if result is not None:
                return result
    return None

# Usage example
file_path = "/home/chaahat/Desktop/PhD_projects/code/PlaJABenchmarks/benchmarks//blocksworld/additional_properties/repair/random_starts_1000/blocksworld_4_2/pa_blocksworld_4_2_random_starts_1000.jani"
start_data = extract_start_from_jani(file_path)
print(f"\nTotal elements: {len(start_data)}")