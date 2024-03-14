import json


def create_map(initial_location, map_filename):
    """
    Create a map from the initial location and save it to a file.
    @param initial_location: the initial location of the map (string)
    @param map_filename: the name of the file to save the map to (string)
    """
    # lowercase everything for consistency
    initial_location = initial_location.lower()

    _map = {initial_location: {}}
    with open(map_filename, 'w') as f:
        json.dump(_map, f)


def add_location_to_map(from_location, direction, to_location, map_filename):
    """
    Add a location to the map and save it to a file.
    @param location: the location to add to the map (string)
    @param map_filename: the name of the file to save the map to (string)
    """
    # lowercase everything for consistency
    from_location = from_location.lower()
    direction = direction.lower()
    to_location = to_location.lower()
    
    with open(map_filename, 'r') as f:
        _map = json.load(f)
    if from_location not in _map:
        _map[from_location] = {direction: to_location}
    elif direction not in _map[from_location]:
        _map[from_location][direction] = to_location
    with open(map_filename, 'w') as f:
        json.dump(_map, f)