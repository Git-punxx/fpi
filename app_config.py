import configparser
import os
from itertools import product
from pathlib import Path
import json

CONFIG_FILE = 'config.ini'
FPI_CONFIG_JSON = os.path.join(os.path.dirname(__file__), 'fpi_config.json')


class ConfigManager:
    def __init__(self, configuration_file):
        self._file = configuration_file

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
    def __init__(self, configuration_file):
        ConfigManager.__init__(self, configuration_file)
        with open(self._file, 'r') as f:
            self._json = json.load(f)

    @property
    def base_dir(self):
        return self._json['paths']['databasedir']

    @base_dir.setter
    def base_dir(self, path):
        self._json['paths']['databasedir'] = path
        with open(self._file, 'w') as f:
            json.dump(self._json, f)


    @property
    def animal_lines(self):
        return self._json['categories']['mouselines']

    @property
    def genotypes(self):
        return self._json['categories']['genotypes']

    @property
    def treatments(self):
        return self._json['categories']['treatment']

    @property
    def stimulations(self):
        return self._json['categories']['stimulations']

    @property
    def name_pattern(self):
        return self._json['metadata']['experimentnames']

    @property
    def categories(self):
        return self._json['categories']

    def folder_structure(self):
        base = self.base_dir
        structure = [['Data']]
        for option, values in self.categories.items():
            structure.append(values)
        paths = [os.path.join(base, *folders) for folders in product(*structure)]
        return paths

    def create_folders(self):
        raise NotImplementedError

config_manager = JSONConfigManager(FPI_CONFIG_JSON)

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
    cfg_manager = JSONConfigManager(FPI_CONFIG_JSON)
    print(cfg_manager.base_dir)
    cfg_manager.base_dir = 'Some other folder'
    print(cfg_manager.base_dir)
    print(cfg_manager.categories)
    print(cfg_manager.folder_structure())


if __name__ == '__main__':
    test_json()

