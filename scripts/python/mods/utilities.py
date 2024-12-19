import os
import json
import numpy as np
import time


def convert_to_serializable(obj):
    """
    Convert various types of objects to a serializable format.

    This function handles conversion of numpy arrays, numpy scalar types,
    dictionaries, and lists to formats that can be easily serialized to JSON.

    Parameters:
    obj (any): The object to be converted to a serializable format.

    Returns:
    any: The converted object in a serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, type(lambda: None)):
        return obj.__name__
    else:
        return None

def save_kwargs(kwargs, path):
    """
    Save keyword arguments to a JSON file.

    This function takes a dictionary of keyword arguments and saves it to a specified
    file path in JSON format. If the directory does not exist, it will be created.

    Args:
        kwargs (dict): The keyword arguments to save.
        path (str): The file path where the JSON file will be saved (without the .json extension).

    Returns:
        None

    Raises:
        OSError: If the file cannot be created or written to.
        TypeError: If the keyword arguments contain non-serializable data.

    Example:
        save_kwargs({'param1': 10, 'param2': 'value'}, '/path/to/file')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.json', 'w') as f:
        serializable_kwargs = convert_to_serializable(kwargs)
        json.dump(serializable_kwargs, f, indent=4)
    print('Kwargs saved.')


def get_max_depth(d, level=1):
    """
    Recursively finds the maximum depth of a nested dictionary.

    Args:
        d (dict): The dictionary to check.
        level (int, optional): The current level of depth. Defaults to 1.

    Returns:
        int: The maximum depth of the dictionary.
    """
    if not isinstance(d, dict) or not d:
        return level
    return max(get_max_depth(value, level + 1) for value in d.values())


def print_nested_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with indentation.

    Args:
        d (dict): The dictionary to print.
        indent (int, optional): The current level of indentation. Defaults to 0.

    Returns:
        None
    """
    dict_depth = get_max_depth(d)
    for key, value in d.items():
        if dict_depth > 2:
            print('')
            print('------- ' + str(key), end=' -------')
        else:
            print(' ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            print_nested_dict(value, indent + 4)
        else:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, type(lambda: None)):
                value = value.__name__
            print(value)
    time.sleep(0.5)


def scientific_notation_parser(float_str):
    if 'e' in float_str:
        base, exponent = float_str.split('e')
        return float(base) * 10**int(exponent)
    return float(float_str)
