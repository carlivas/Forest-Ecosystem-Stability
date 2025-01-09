import os
import json
import numpy as np
import time
from mods.plant import Plant
from mods.fields import DensityFieldSPH
from mods.buffers import DataBuffer, StateBuffer, FieldBuffer
from scipy.spatial import KDTree

def convert_to_serializable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        val = obj
        if isinstance(obj, Plant):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, KDTree):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, DensityFieldSPH):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, DataBuffer):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, StateBuffer):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, FieldBuffer):
            val = convert_to_serializable(obj.__dict__)
        elif isinstance(obj, np.ndarray):
            val = obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            val = int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            val = float(obj)
        elif isinstance(obj, np.bool_):
            val = bool(obj)
        elif isinstance(obj, dict):
            val = {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            val = [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, type(lambda: None)):
            val = obj.__name__
        return val
    return val

def save_kwargs(kwargs, path, exclude=None):
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
    
    if exclude is not None:
        kwargs = {k: v for k, v in kwargs.items() if k not in exclude}
    kwargs = dict(sorted(kwargs.items()))
    
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


def print_nested_dict(d, indent=0, exclude=None):
    """
    Recursively prints a nested dictionary with indentation.

    Args:
        d (dict): The dictionary to print.
        indent (int, optional): The current level of indentation. Defaults to 0.

    Returns:
        None
    """
    if exclude is not None:
        d = {k: v for k, v in d.items() if k not in exclude
                and not isinstance(v, type(lambda: None))}
    d = dict(sorted(d.items()))
    
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
