import os
import json
import torch

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def load_json(json_file):
    # Load the JSON file into a variable
    with open(json_file) as f:
        json_data = json.load(f)

    # Return the data as a dictionary
    return json_data

def save_json(json_file, data):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    # Save the data to a JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f)