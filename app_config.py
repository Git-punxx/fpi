import configparser
import os
from itertools import product

CONFIG_FILE = 'config.ini'

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), CONFIG_FILE))

def base_dir():
    return config['Paths']['DataBaseDir']

def set_base_dir(path):
    print('Setting path...')
    config['Paths']['DataBaseDir'] = path
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)
        print('Changes saved...')


def animal_lines():
    return [line.upper() for line in config['Categories']['MouseLines'].split()]

def genotypes():
    return [line.upper() for line in config['Categories']['Genotypes'].split()]

def treatments():
    return [line.upper() for line in config['Categories']['Treatment'].split()]

def stimulations():
    return [line.upper() for line in config['Categories']['Stimulations'].split()]

def name_pattern():
    return config['Metadata']['ExperimentName']

def categories():
    res = list(config['Categories'].keys())
    return res


def folder_structure():
    base = base_dir()
    structure = []
    for option, values in config.items('Categories'):
        structure.append(values.split())
    paths = [os.path.join(base, *folders) for folders in product(*structure)]
    return paths


if __name__ == '__main__':
    import re
    print(animal_lines())
    print(genotypes())
    print(stimulations())
    print(base_dir())
    print(categories())

    text = 'datastore_20200404_6789_response_0.csv'
    patt = name_pattern()
    print(patt)
    m = re.search(patt, text)
    if m is not None:
        print(m.group(0)[1:])
    else:
        print('No result')


    print(folder_structure())
