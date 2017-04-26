"""Read data from each CSV file, find the appropriate log file, and group 
data into sets of the same parameters but varying driving frequency. Once 
this is done, calculate the response curve for each set."""

import pandas as pd
import re
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

def find_files(directory, check_type='.csv'):
    """Find all the files, non-recursively, in the directory, of the 
    specified type."""
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    if check_type is not None:
        onlyfiles = [f for f in onlyfiles if f[-len(check_type):] == check_type]
    return onlyfiles


def get_filename_roots(directory):
    onlyfiles = find_files(directory)
    filename_roots = []
    for csv_file in onlyfiles:
        if re.search(re.compile(r'displacements'), csv_file):
            if csv_file[:-17] + 'measured-vals.csv' in onlyfiles:
                filename_roots.append(csv_file[:-17])
    return filename_roots


def get_configs(directory, filename_root):
    log_files = find_files(directory + 'logs/', check_type='.txt')
    root = filename_root
    look_for = root + 'log.txt'
    assert look_for in log_files
    with open('{}logs/{}'.format(directory, look_for), 'r') as f:
        log_contents = f.read()
        config_string = re.findall('(?<={)[^}]*(?=})', log_contents)
        assert len(config_string) == 1
        config_string = config_string[0].replace('array', 'np.array')
        configs = config_string.split(',')
        config_dict = {}
        for key_string in configs:
            key = eval(key_string.split(':')[0])
            value = eval(key_string.split(':')[1])
            config_dict[key] = value
        # NOTE this will split the string into individual keys as long as the
        # numpy array only have 1 element each!
    return config_dict


def main():
    directory = '../../../Tests/ExperimentClasses/NRRegimesPython/'
    filename_roots = get_filename_roots(directory)
    outputs = []
    for root in filename_roots:
        one_set = list([])
        one_set.append(get_configs(directory, root))
        disps = pd.read_csv('{}{}displacements.csv'.format(directory, root),
                            index_col=0)
        torques = pd.read_csv('{}{}measured-vals.csv'.format(directory, root),
                              index_col=0)
        one_set.append(disps)
        one_set.append(torques)
        print(one_set)
        outputs.append(one_set)
        plt.plot(disps['t'], disps['theta'])
        plt.plot(torques['t'], torques['theta_sim'])
        plt.show()
        input()


if __name__ == '__main__':
    main()

# todo calculate analytic torque by reading the torque and log files.
