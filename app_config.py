import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__),'config.ini'))

def base_dir():
    return config['Paths']['DataBaseDir']

def animal_lines():
    return [line.upper() for line in config['Categories']['MouseLines'].split()]

def genotypes():
    return [line.upper() for line in config['Categories']['Genotypes'].split()]

def stimulations():
    return [line.upper() for line in config['Categories']['Stimulations'].split()]

def name_pattern():
    return config['Metadata']['ExperimentName']

def categories():
    res = list(config['Categories'].keys())
    print(res)
    return res

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
