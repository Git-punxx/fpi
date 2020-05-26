import configparser
import os
from itertools import product
from pathlib import Path
import json

CONFIG_FILE = 'config.ini'
CONFIG_JSON = 'config.json'

class ConfigManager:
    def __init__(self, config_file):
        self._file = config_file

    @property
    def base_dir(self):
        raise NotImplementedError

    @base_dir.setter
    def base_dir(self):
        raise NotImplementedError

    @property
    def animal_lines(self):
        raise NotImplementedError

    @property
    def genotypes(self):
        raise NotImplementedError

    @property
    def treatments(self):
        raise NotImplementedError

    @property
    def stimulations(self):
        raise NotImplementedError

    @property
    def name_pattern(self):
        raise NotImplementedError

    @property
    def categories(self):
        raise NotImplementedError

    def folder_structure(self):
        raise NotImplementedError

    def create_folders(self):
        raise NotImplementedError

class JSONConfigManager(ConfigManager):
    def __init__(self, json_file):
        ConfigManager.__init__(self, json)
        with open(self._file, 'r+') as _file:
            self._json = json.loads(_file.read())

    @property
    def base_dir(self):
        return self._json.paths.databasedir

    @base_dir.setter
    def base_dir(self):
        raise NotImplementedError

    @property
    def animal_lines(self):
        raise NotImplementedError

    @property
    def genotypes(self):
        raise NotImplementedError

    @property
    def treatments(self):
        raise NotImplementedError

    @property
    def stimulations(self):
        raise NotImplementedError

    @property
    def name_pattern(self):
        raise NotImplementedError

    @property
    def categories(self):
        raise NotImplementedError

    def folder_structure(self):
        raise NotImplementedError

    def create_folders(self):
        raise NotImplementedError



config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), CONFIG_FILE))

def base_dir():
    return config['Paths']['DataBaseDir']

def set_base_dir(path):
    print('Setting path...')
    config['Paths']['DataBaseDir'] = path
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)


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
    structure = [['Data']]
    for option, values in config.items('Categories'):
        structure.append(values.split())
    paths = [os.path.join(base, *folders) for folders in product(*structure)]
    return paths

def create_folders():
    paths = folder_structure()
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def test_module():
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

def test_json():
    cfg_manager = JSONConfigManager(CONFIG_JSON)
    print(cfg_manager.base_dir)


if __name__ == '__main__':
    test_json()
