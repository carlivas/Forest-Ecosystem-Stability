import os
import json
import numpy as np
import time
from scipy.spatial import KDTree
from datetime import datetime

from mods.plant import *
from mods.fields import *
from mods.buffers import *

def linear_regression(x, y, advanced=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.vstack([np.ones_like(x), x]).T

    # Calculate theta using the equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept, slope = theta

    if advanced:
        # Calculate the regression line using the calculated theta
        regression_line = X @ theta

        residuals = y - regression_line
        # Calculate the sum of squared residuals
        sum_squared_residuals = np.sum(residuals**2)
        # regression_line = slope * X[:, 1] + intercept
        # residuals = y - regression_line
        return intercept, slope, regression_line, residuals, sum_squared_residuals
    else:
        return intercept, slope
    
def convergence_check(x, y, trend_threshold=1e-3):
    dx = x[1] - x[0]
    # Calculate the linear regression parameters
    intercept, slope = linear_regression(x, y)
    # Calculate the regression line
    regression_line = intercept + slope * x

    # Calculate the convergence factor
    if np.std(y) != 0:
        convergence_factor = np.abs(
            (regression_line[-1] - np.mean(y)) / np.std(y))
    else:
        convergence_factor = np.inf

    # Check if the convergence factor is below the threshold
    return convergence_factor < trend_threshold

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
        elif isinstance(obj, DensityFieldCustom):
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

def save_dict(d, path, exclude=None):
    if exclude is not None:
        d = {k: v for k, v in d.items() if k not in exclude}
    d = dict(sorted(d.items()))

    if not path.endswith('.json'):
            path = path + '.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        serializable_dictionary = convert_to_serializable(d)
        json.dump(serializable_dictionary, f, indent=4)
    print(f'utilities.save_dict: Kwargs saved at path {path}')


def dbh_to_crown_radius(dbh):
    # diameter at breast height to crown radius
    # everything in m
    d = 1.42 + 28.17*dbh - 11.26*dbh**2
    return d/2

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


def print_dict(d, indent=0, exclude=None):
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
            print_dict(value, indent + 4)
        else:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, type(lambda: None)):
                value = value.__name__
            print(f'{value}')
    time.sleep(0.5)


def scientific_notation_parser(float_str):
    if 'e' in float_str:
        base, exponent = float_str.split('e')
        return float(base) * 10**int(exponent)
    return float(float_str)


def convert_dict(d, conversion_factors, reverse=False):
        if reverse:
            converted_dict = {key: (value * conversion_factors[key] if key in conversion_factors.keys() else value)
                              for key, value in d.items()}
        else:
            converted_dict = {key: (value / conversion_factors[key] if key in conversion_factors.keys() else value)
                              for key, value in d.items()}
        return converted_dict
    
def format_float(val):
    if val == 0:
        val_str = '0'
    else:
        split_e = str(val).split('e')
        exponent = int(split_e[1]) if len(split_e) > 1 else 0
        split_dot = split_e[0].split('.')
        precision = exponent - len(split_dot[1]) if len(split_dot) > 1 else exponent
        val_decimal = str(split_e[0]).replace(".", "")
        
        trailing_zeros = len(val_decimal) - len(val_decimal.rstrip("0"))
        
        val_decimal = val_decimal.strip("0")
        precision += trailing_zeros
        
        val_str = f'{val_decimal}e{precision}' if precision != 0 else val_decimal
    return val_str

def format_alias(alias):
    alias = alias.replace(' ', '_').replace('-', '_').replace('.', '_')
    return alias

def generate_alias(id, keys, abrevs = None, time=False, **kwargs):
    alias = f'{id}_'
    for i, key in enumerate(keys):
        if key in kwargs:
            if isinstance(abrevs, dict) and key in abrevs:
                abrev = abrevs[key]
            elif isinstance(abrevs, list) and i < len(abrevs):
                abrev = abrevs[i]
            else:
                abrev = [s[0].upper() for s in key.split('_') if len(s) > 0]
                abrev = ''.join(abrev)                  
            if isinstance(kwargs[key], float):
                alias += f'{abrev}{format_float(kwargs[key])}_'
            else:
                alias += f'{abrev}{kwargs[key]}_'            
    if time:
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        alias += f'{current_time}_'
    alias = alias.rstrip('_')
    alias = format_alias(alias)
    
    return alias

def get_unique_aliases(folder, includes=None, excludes=None):        
    files = os.listdir(folder)
    aliases = [f.split('-')[-1].split('.')[0] for f in files if f.endswith('.json')]
    if includes is not None:
        aliases = [alias for alias in aliases if includes in alias]
    if excludes is not None:
        aliases = [alias for alias in aliases if excludes not in alias]
        
    aliases = list(set(aliases))
    aliases.sort()
    return aliases
    