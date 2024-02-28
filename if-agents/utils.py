import json

def write_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def read_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
