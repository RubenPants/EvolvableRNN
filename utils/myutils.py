"""
myutils.py

Share utils used across the project.

Utils overview:
 * DICT: load_dict, update_dict
 * PICKLE: load_pickle, store_pickle
 * SYSTEM: get_subfolder
 * TIMING: drop, prep, status_out, total_time
 * LOGGING: append_log
"""
import json
import os
import pickle
import sys
from glob import glob
from timeit import default_timer as timer


# ------------------------------------------------------> DICT <------------------------------------------------------ #

def load_dict(full_path):
    """Load a stored dictionary."""
    with open(full_path, 'r') as f:
        return json.load(f)


def update_dict(full_path, new_dict):
    """
    Update existing dictionary if exists, otherwise create new.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_dict: The JSON file that must be stored
    """
    files = glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        original.update(new_dict)
        with open(full_path, 'w') as f:
            json.dump(original, f, indent=2)
    else:
        # Create new file to save json in
        with open(full_path, 'w') as f:
            json.dump(new_dict, f, indent=2)


# -----------------------------------------------------> LOGGING <---------------------------------------------------- #


def append_log(inp: str, full_path: str):
    """Append the log-file with the given string. Creates the file if not yet exist."""
    with open(full_path, 'a', encoding="utf-8") as file:
        file.write("%s\n" % inp)


# -----------------------------------------------------> PICKLE <----------------------------------------------------- #


def load_pickle(full_path):
    """Load pickled object."""
    with open(full_path, 'rb') as f:
        g = pickle.load(f)
    return g


def store_pickle(obj, full_path):
    """Store object as pickle."""
    with open(full_path, 'wb') as f:
        pickle.dump(obj, f)


# -----------------------------------------------------> SYSTEM <----------------------------------------------------- #

def get_subfolder(path, subfolder, init=True):
    """
    Check if subfolder already exists in given directory, if not, create one.
    
    :param path: Path in which subfolder should be located (String)
    :param subfolder: Name of the subfolder that must be created (String)
    :param init: Initialize folder with __init__.py file (Bool)
    :return: Path name if exists or possible to create, raise exception otherwise
    """
    if subfolder and subfolder[-1] not in ['/', '\\']:
        subfolder += '/'
    
    # Path exists
    if os.path.isdir(path) or path == '':
        if not os.path.isdir(path + subfolder):
            # Folder does not exist, create new one
            os.mkdir(path + subfolder)
            if init:
                with open(path + subfolder + '__init__.py', 'w') as f:
                    f.write('')
        return path + subfolder
    
    # Given path does not exist, raise Exception
    raise FileNotFoundError(f"Path '{path}' does not exist")


# -----------------------------------------------------> TIMING <----------------------------------------------------- #

time_dict = dict()


def drop(key=None, silent=False):
    """
    Stop timing, print out duration since last preparation and append total duration.
    """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    time_dict[key]['end'] = timer()
    time_dict[None]['end'] = timer()
    
    # Determine difference
    start = time_dict[key]['start']
    end = time_dict[key]['end']
    diff = end - start
    
    # Fancy print diff
    if not silent:
        status_out(' done, ' + get_fancy_time(diff) + '\n')
    
    # Save total time
    if key is not None:
        if 'sum' not in time_dict[key]:
            time_dict[key]['sum'] = diff
        else:
            time_dict[key]['sum'] += diff
    
    return diff


def get_fancy_time(sec):
    """
    Convert a time measured in seconds to a fancy-printed time.
    
    :param sec: Float
    :return: String
    """
    h = int(sec) // 3600
    m = (int(sec) // 60) % 60
    s = sec % 60
    if h > 0:
        return '{h} hours, {m} minutes, and {s} seconds.'.format(h=h, m=m, s=round(s, 2))
    elif m > 0:
        return '{m} minutes, and {s} seconds.'.format(m=m, s=round(s, 2))
    else:
        return '{s} seconds.'.format(s=round(s, 2))


def intermediate(key=None):
    """
    Get the time that already has passed.
    """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    
    # Determine difference
    start = time_dict[key]['start']
    end = timer()
    return end - start


def prep(msg="Start timing...", key=None, silent=False):
    """
    Prepare timing, print out the given message.
    """
    global time_dict
    if key not in time_dict:
        time_dict[key] = dict()
    if not silent:
        status_out(msg)
    time_dict[key]['start'] = timer()
    
    # Also create a None-instance (in case drop() is incorrect)
    if key:
        if None not in time_dict:
            time_dict[None] = dict()
        time_dict[None]['start'] = timer()


def print_all_stats():
    """
    Print out each key and its total (cumulative) time.
    """
    global time_dict
    if time_dict:
        if None in time_dict: del time_dict[None]  # Remove None-instance first
        print("\n\n\n---------> OVERVIEW OF CALCULATION TIME <---------\n")
        keys_space = max(map(lambda x: len(x), time_dict.keys()))
        line = ' {0:^' + str(keys_space) + 's} - {1:^s}'
        line = line.format('Keys', 'Total time')
        print(line)
        print("-" * (len(line) + 3))
        line = '>{0:^' + str(keys_space) + 's} - {1:^s}'
        t = 0
        for k, v in sorted(time_dict.items()):
            try:
                t += v['sum']
                print(line.format(k, get_fancy_time(v['sum'])))
            except KeyError:
                raise KeyError(f"KeyError: Probably you forgot to place a 'drop()' in the {k} section")
        end_line = line.format('Total time', get_fancy_time(t))
        print("-" * (len(end_line)))
        print(end_line)


def remove_time_key(key: str):
    """
    Remove a key from the timing-dictionary.
    
    :param key: Key that must be removed
    """
    global time_dict
    if key in time_dict:
        del time_dict[key]


def status_out(msg):
    """
    Write the given message.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
