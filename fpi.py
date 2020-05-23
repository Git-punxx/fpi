import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from collections import namedtuple
from abc import abstractmethod
import app_config
from pathlib import Path
import re
from itertools import takewhile



'''
The point here is that we design a class hirearchy that will be able to handle different types
of files (csv, h5py) providing a united interface.
The data files we want to analyze are categorized based on:
1. Type of animal_line (Shank, PTEN etc)
2. Type of animal? (ko, wt)
3. Type of file (h5, csv)
4. Type of metadata they contain (response, timecourse, all_pixels)


So how we do that. 
First of all we look at the directory and find what kinds of files it contains.
Datastore files represent a full experiment. OUr parser can extract all_pixels, response and timecourse from them
For csv files:
csv files are produced from a datastore file that represents an experiment.
So we have to divide them based on their filename. That means that there should be triads od csv files for every experiment.

'''
fpi_meta = namedtuple('fpi_meta', 'name line stimulus genotype')
###### Parsers ######
class FPIParser:
    '''
    The parser we will be using to parse our data files. It accepts the name of the experiment and then
    it reconstructs its path.
    Depending on the file format we will use a factory method to return the appropriate parser.
    '''
    def __init__(self, experiment, path):
        self.experiment = experiment
        self._path = path
        self._file_type = None


    @abstractmethod
    def parser_type(self):
        raise NotImplementedError

    @abstractmethod
    def response(self):
        raise NotImplementedError

    @abstractmethod
    def timecourse(self):
        raise NotImplementedError

    @abstractmethod
    def all_pixel(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        raise NotImplementedError

class CSVParser(FPIParser):
    def __init__(self, experiment, path):
        FPIParser.__init__(self, experiment, path)

    def parser_type(self):
        return 'csv'

    def response(self):
        data = []
        fname = [f for f in self._path if 'response' in f][0]
        with open(fname, 'r') as datafile:
            datafile.readline()
            for line in datafile:
                if ',' not in line:
                    continue
                else:
                    row = float(line.split(',')[1].rstrip())
                    data.append(row)
        return np.array(data)

    def timecourse(self):
        y = []
        fname = [f for f in self._path if 'response' in f][0]
        with open(fname, 'r') as f:
            f.readline()
            for line in f:
                if ',' not in line:
                    continue
                else:
                    y.append(float(line.split(',')[1]))
        return np.array(y)

    def all_pixel(self):
        data = []
        fname = [f for f in self._path if 'response' in f][0]
        with open(fname, 'r') as f:
            f.readline()
            for line in f:
                if ',' not in line:
                    continue
                else:
                    data.append(float(line.split(',')[1]))
        return np.array(data)

class HD5Parser(FPIParser):
    def __init__(self, experiment, path):
        FPIParser.__init__(self, experiment, path)

    def parser_type(self):
        return 'hdf5'

    def response(self):
        print('HD5 parser parses response')
        store = h5py.File(self._path, 'r')
        store.close()

    def timecourse(self):
        pass

    def all_pixel(self):
        pass

#### Utility functions #######
def debug(func):
    def wrapper(*args, **kwargs):
        print(f'Running {func.__name__}')
        res = func(*args, **kwargs)
        print(res)
        return res
    return wrapper

def extract_name(path):
    '''
    Given a pathm, this function returns the name of the experiment which is the numeric value contained in
    the basename of the path. This name will be used to instantiate FPIExperiment instances that are able to reconstruct
    their path.
    :return: the name of the experiment as a string
    '''
    name = re.search(app_config.name_pattern(), str(path))
    return name.group(0)[1:]

def fpiparser(path):
    '''
    A factory method that tries to understand if we are dealing with a list of csv files or a single datastore file
    :param path: A list of filenames
    :return: A FPIParser object
    '''
    # Check if there is only one file and that it ends with h5. Then we return an HD%Parser
    if len(path) == 1 and os.path.basename(path[0]).endswith('h5'):
        return HD5Parser(extract_name(path[0]), path[0])
    # else if there are more files with the same experiment name, assume there are three csv files, one for response,
    # timecourse and all_pixels. This should be enforced in the analysis step
    elif len(path) == 3:
        if len(list(takewhile(lambda x: x.endswith('.csv'), path))) == 3:
            return CSVParser(extract_name(path), path) # and pray that they are response, timecourse and all pixels
        else:
            raise ValueError(f'{path} could not be matched against a parser')
    else:
        raise ValueError(f'{path} could not be matched against a parser')

#### Model #####
class FPIGatherer:
    def __init__(self):
        self.path = app_config.base_dir()
        self._children = None
        self.experiments = None
        self.working_list = self.experiment_list()


    def __getitem__(self, item):
        for animal_line in self.children.keys():
            print(animal_line.path)
            if item in animal_line.path:
                return animal_line

    def experiment_list(self, filtered_list = None):
        result = []
        for exp in self.get_experiments():
            result.append(fpi_meta._make((exp.name, exp.animal_line, exp.stimulation, exp.genotype)))
        return result



    def find(self, exp_number):
        print('Searching for ', exp_number)
        for item in self.working_list:
            if exp_number == item.name:
                return item

    @property
    def children(self):
        '''
        Scan the base directory and return the animal line folders.
        Create a dict in which the directory is the key and an AnimalLine is the value.
        :return:
        '''
        if self._children is not None:
            return self._children
        else:
            animal_lines = app_config.animal_lines()
            base_path = Path(self.path)
            self._children = {AnimalLine(str(d)): d for d in base_path.iterdir() if os.path.basename(d).upper() in animal_lines}
            return self._children

    def gather(self):
        for line in self.children.keys():
            line.gather()

    def get_experiments(self):
        if self.experiments is not None:
            return self.experiments
        else:
            result = []
            for child in self.children.keys():
                result += child.get_experiments()
            self.experiments = result
            return self.experiments

    def clear(self):
        self.working_list = self.experiment_list()

    def filterLine(self, line):
        exps = self.working_list
        print(exps)
        self.working_list = [exp for exp in exps if exp.line.upper() == line.upper()]
        return self.working_list

    def filterGenotype(self, gen):
        exps = self.working_list
        print(exps)
        self.working_list = [exp for exp in exps if exp.genotype.upper() == gen.upper()]
        return self.working_list

    def filterStimulus(self, stim):
        exps = self.working_list
        print(exps)
        self.working_list = [exp for exp in exps if exp.stimulus.upper() == stim.upper()]
        return self.working_list

    def get_active(self):
        '''
        Check whci experiments are contained in the working list (after the filtering performed from the gui)
        :return: A list of FPIExperiment object
        '''
        working_names = [exp.name for exp in self.working_list]
        res = []
        for exp in self.experiments:
            if exp.name in working_names:
                res.append(exp)
            else:
                print(f'{exp.name } not in the list')
        return res


    def get_response_peak(self):
        experiments = self.get_active()
        return [exp.peak_latency() for exp in experiments]

    def get_response_latency(self):
        experiments = self.get_active()
        return [exp.response_latency() for exp in experiments]

    def __str__(self):
        return f'FPIGatherer@{self.path}'

class AnimalLine:
    def __init__(self, path):
        self.path = path
        self._children = None

    def __getitem__(self, item):
        for stim in self._children.keys():
            print(stim._path)
            if item in stim._path.upper():
                return stim

    def items(self):
        return self.children.items()

    @property
    def children(self):
        '''
        Scan the animal line directory for animal folders
        :return:
        '''
        if self._children is not None:
            return self._children
        else:
            stims = app_config.stimulations()
            base_path = Path(self.path)
            children = [str(d) for d in base_path.iterdir() ]
            self._children = {Stimulation(str(d)): d for d in base_path.iterdir() if
                              os.path.basename(d).upper() in stims}
            return self._children

    def gather(self):
        for stim in self.children.keys():
            stim.gather()

    def get_experiments(self):
        result = []
        for child in self.children.keys():
            result += child.get_experiments()
        return result

    def __str__(self):
        return f'AnimalLine {os.path.basename(str(self.path))}'

class Stimulation:
    def __init__(self, path):
        self._path = path
        self._children = None

    def __getitem__(self, item):
        for genotype in self._children.keys():
            if item in genotype.path.upper():
                return genotype
    @property
    def children(self):
        '''
        Scan the animal line directory for animal folders
        :return:
        '''
        genotypes = app_config.genotypes()
        base_path = Path(self._path)
        self._children = {Genotype(str(d)): d for d in base_path.iterdir() if
                          os.path.basename(d) in genotypes}
        return self._children

    def gather(self):
        for genotype in self.children.keys():
            genotype.gather()

    def get_experiments(self):
        result = []
        for child in self.children.keys():
            exp = child.get_experiments()
            result += exp
        return result

    def __str__(self):
        return f'Stimulation {os.path.basename(str(self._path))}'

class Genotype:
    def __init__(self, path):
        self.path = path
        self._children = None

    def __getitem__(self, item):
        return self.children.get(item)

    @property
    def children(self):
        '''
        Scan the animal line directory for animal folders
        :return:
        '''
        base_path = Path(self.path)
        genotype = os.path.basename(self.path)
        stimulation = os.path.basename(str(base_path.parent))
        animal_line = os.path.basename(str(base_path.parent.parent))
        self._children = {extract_name(str(d)): FPIExperiment(name=extract_name(d), animal_line = animal_line, stimulation=stimulation, genotype=genotype)
                          for d in base_path.iterdir()}
        return self._children


    def gather(self):
        for line in self.children.values():
            line.gather()

    def get_experiments(self):
        return self.children.values()

    def __str__(self):
        return f'Genotype {os.path.basename(str(self.path))}'

class FPIExperiment:
    '''
    This class represents the results of an FPI experiment.
    It holds the dataframes of the result of the experiment we need to analyze
    The dataframes are read from a file produced by the imaging module. It could be an .h5 file or it could be
    3 csv files, each for each dataframe (resonse, timecourse, all_pixel)
    We should pass the name of the experiment only, NOT THE PATH, because if we need to read from three files, this
    operations should be handled by the parser.
    By providing the name of the experiment we could build the path using the config options
    '''
    def __init__(self, name, animal_line, stimulation, genotype):
        self.animal_line = animal_line
        self.stimulation = stimulation
        self.genotype = genotype
        self.name = name

        self._path = None
        self._parser = None
        self._all_pixel = None
        self._response = None
        self._timecourse = None

        self.materialize()

    def materialize(self):
        '''
        Create the appropriate parser for the specific file (csv, h5 etc) and read the dataframes we need
        :return:
        '''
        self.build_path()
        self._parser = fpiparser(self._path)
        self._all_pixel = self._parser.all_pixel()
        self._response = self._parser.response()
        self._timecourse = self._parser.timecourse()

    def build_path(self):
        base_dir = app_config.base_dir()
        base_path = os.path.join(base_dir, self.animal_line, self.stimulation, self.genotype)
        data_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) if self.name in f]
        self._path = data_paths

    def gather(self):
        pass

    @property
    def all_pixel(self):
        return self._all_pixel

    @property
    def response(self):
        return self._all_pixel

    @property
    def timecourse(self):
        return self._all_pixel

    def baseline_mean(self, n_baseline = 30):
        data = self.response
        # Compute the mean of the baseline
        baseline = np.array(data[:n_baseline])
        baseline_mean = np.mean(baseline)
        return baseline_mean

    def peak_latency(self):
        data = self.response
        if data is None:
            return
        response_region = data[:]
        peak = np.argmax(response_region)
        peak_value = np.max(response_region)
        return (peak, peak_value)

    def response_latency(self, ratio = 0.3, n_baseline = 30):
        data = self.response
        if data is None:
            return
        mean_baseline = self.baseline_mean(n_baseline)
        latency = [(index, val) for index, val in enumerate(data[31:], n_baseline +1) if val > abs(1 + ratio) * mean_baseline]
        return latency


    def __str__(self):
        return f'{self.name}: {self.stimulation} -> {self.genotype}'

    def check(self):
        result = []
        if self.response is not None:
            result.append('response')
        else:
            result.append('No response')
        if self.timecourse is not None:
            result.append('timecourse')
        else:
            result.append('No timecourse')
        if self.all_pixel is not None:
            result.append('all_pixel')
        else:
            result.append('No all_pixelsj')
        return result

if __name__ == '__main__':
    gatherer = FPIGatherer()
    gatherer.gather()
    all_exp = gatherer.get_experiments()

    l = gatherer.experiment_list()
    print(l)
