import configparser
import os
from sys import platform
from itertools import product
from pathlib import Path
import json
from enum import Enum

CONFIG_FILE = 'config.ini'
FPI_CONFIG_JSON = os.path.join(os.path.dirname(__file__), 'fpi_config.json')


class ConfigManager:
    def __init__(self, configuration_file):
        self._file = configuration_file

    def is_linux(self):
        return True if platform.startswith('linux') else False

    def is_mac(self):
        return True if platform.startswith('darwin') else False

    def is_win(self):
        return True if platform.startswith('win32') else False

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

    def file_explorer(self):
        if self.is_mac():
            return ['open']
        elif self.is_win():
            return ['explorer.exe']
        else:
            return None

class JSONConfigManager(ConfigManager):
    def __init__(self, configuration_file):
        ConfigManager.__init__(self, configuration_file)
        with open(self._file, 'r') as f:
            self._json = json.load(f)
        self.update_env()

    def update_env(self):
        os.environ['FPI_PATH'] = self.base_dir

    @property
    def base_dir(self):
        return self._json['paths']['databasedir']

    @base_dir.setter
    def base_dir(self, path):
        self._json['paths']['databasedir'] = path
        with open(self._file, 'w') as f:
            json.dump(self._json, f)


    @property
    def raw_dir(self):
        return self._json['paths']['rawdir']

    @raw_dir.setter
    def raw_dir(self, path):
        self._json['paths']['rawdir'] = path
        with open(self._file, 'w') as f:
            json.dump(self._json, f)

    @property
    def csv_dir(self):
        return self._json['paths']['csv']

    @property
    def animal_lines(self):
        return self._json['categories']['animalline']

    @property
    def genotypes(self):
        return self._json['categories']['genotype']

    @property
    def treatments(self):
        return self._json['categories']['treatment']

    @property
    def stimulations(self):
        return self._json['categories']['stimulation']

    @property
    def name_pattern(self):
        return self._json['metadata']['experimentnames']

    @property
    def categories(self):
        return self._json['categories']

    @property
    def data_export_dir(self):
        base_dir = os.path.dirname(self.base_dir)
        return base_dir + '/data_exports'


    def folder_structure(self):
        base = self.base_dir
        structure = [['Data']]
        for option, values in self.categories.items():
            structure.append(values)
        paths = [os.path.join(base, *folders) for folders in product(*structure)]
        return paths

    def create_folders(self):
        print('Creating new folder structure')
        folders = self.folder_structure()
        [os.makedirs(path, exist_ok=True) for path in folders]

config_manager = JSONConfigManager(FPI_CONFIG_JSON)

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


AnimalLine = Enum('AnimalLine', list(map(lambda x: x.upper(), config_manager.animal_lines)))
Treatment = Enum('Treatment', list(map(lambda x: x.upper(), config_manager.treatments)))
Stimulation = Enum('Stimulation', list(map(lambda x: x.upper(), config_manager.stimulations)))
Genotype = Enum('Genotype', list(map(lambda x: x.upper(), config_manager.genotypes)))


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

def check_trials_folder(path):
    if not os.path.exists(path):
        return False
    else:
        return any(map(lambda x: 'Trial' in x, os.listdir(path)))

def test_module():
    import re

    text = 'datastore_20200404_6789_response_0.csv'
    patt = name_pattern()
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
    print(AnimalLine)
    for line in AnimalLine:
        print(f'{line.name}: {line.value}')


if __name__ == '__main__':
    test_json()

