from typing import Type
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import h5py
from collections import namedtuple
from abc import abstractmethod
from app_config import config_manager as app_config
import re
from pubsub import pub
from pub_messages import ANALYSIS_UPDATE
from intrinsic.imaging import check_datastore


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
fpi_meta = namedtuple('fpi_meta', 'name line stimulus treatment genotype')

CHOICES_CHANGED = 'choices.changed'
EXPERIMENT_LIST_CHANGED = 'experiments.list.changed'
EXPERIMENT_SCANNING = 'experiment.scanning'


#### Utility functions #######
def debug(func):
    def wrapper(*args, **kwargs):
        print(f'In {func.__name__}')
        res = func(*args, **kwargs)
        return res

    return wrapper


def extract_name(path):
    '''
    Given a pathm, this function returns the name of the experiment which is the numeric value contained in
    the basename of the path. This name will be used to instantiate FPIExperiment instances that are able to reconstruct
    their path.
    :return: the name of the experiment as a string
    '''
    name = re.search(app_config.name_pattern, str(path))
    try:
        res = name.group(0)[1:]
    except Exception as e:
        res = 'Default name'
    return res


def fpiparser(path):
    '''
    A factory method that tries to understand if we are dealing with a list of csv files or a single datastore file
    :param path: A list of filenames
    :return: A FPIParser object
    '''
    # Check if there is only one file and that it ends with h5. Then we return an HD%Parser
    if os.path.basename(path).endswith('h5'):
        return HD5Parser(extract_name(path), path)
    # timecourse and all_pixels. This should be enforced in the analysis step
    else:
        # raise ValueError(f'{path} could not be matched against a parser')
        print('No parser for ', path)
        pass


def normalize_stack(stack, n_baseline=30):
    # Global average
    y = np.nanmean(stack, (0, 1))[1:n_baseline]
    y_min, y_max = y.min(), y.max()
    # Exponential fit during baseline
    t = np.arange(1, n_baseline)
    z = 1 + (y - y_min) / (y_max - y_min)
    p = np.polyfit(t, np.log(z), 1)
    # Modeled decay
    full_t = np.arange(stack.shape[2])
    decay = np.exp(p[1]) * np.exp(full_t * p[0])
    # Renormalized
    decay = (decay - 1) * (y_max - y_min) + y_min
    norm_stack = stack - decay

    return norm_stack


#
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
        name = extract_name(self._path)
        try:
            candidates = [os.path.join(os.path.dirname(self._path), f) for f in os.scandir(self._path[0])]
        except Exception:
            return
        data = []
        fname = [f for f in candidates if 'response' in f and name in f]
        if not fname:
            return
        if not os.path.exists(fname[0]):
            return
        with open(fname[0], 'r') as datafile:
            datafile.readline()
            for line in datafile:
                if ',' not in line:
                    continue
                else:
                    row = float(line.split(',')[1].rstrip())
                    data.append(row)
            res = np.array(data)
            return res

    def timecourse(self):
        name = extract_name(self._path)
        try:
            candidates = [os.path.join(os.path.dirname(self._path), f) for f in os.scandir(self._path)[0]]
        except Exception:
            return
        y = []
        fname = [f for f in candidates if 'timecourse' in f and name in f]
        if not fname:
            return
        if not os.path.exists(fname[0]):
            return
        with open(fname[0], 'r') as f:
            f.readline()
            for line in f:
                if ',' not in line:
                    continue
                else:
                    y.append(float(line.split(',')[1]))
        return np.array(y)

    def all_pixel(self):
        name = extract_name(self._path)
        try:
            candidates = [os.path.join(os.path.dirname(self._path), f) for f in os.scandir(self._path[0])]
        except Exception:
            return
        fname = [f for f in candidates if 'all_pixel' in f and name in f]
        data = []
        if not fname:
            return
        if not os.path.exists(fname[0]):
            return
        with open(fname[0], 'r') as f:
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
        with h5py.File(self._path, 'r') as datastore:
            # here we need to see if we will use 'response' or 'resp_map'
            try:
                return datastore['df']['avg_df'][()]
            except Exception as e:
                print(e)
                return None

    def timecourse(self):
        with h5py.File(self._path, 'r') as datastore:
            # Get the normalized stack
            # We need to get the normalized stack
            # normalize_stack(self.stack, self.n_baseline
            # self.stack is returned on avg_stack()
            # which is df['stack'] in the datastore
            try:
                avg_stack = datastore['df']['stack']
                df = normalize_stack(avg_stack)
                # Compute the mean
                df_avg = df.std((0, 1))
                df_std = df.mean((0, 1))
                # Compute the average
                timecourse = np.vstack((np.arange(1, df_avg.shape[0] + 1, dtype=np.intp), df_avg, df_std)).T
                return timecourse
            except Exception as e:
                return

    def all_pixel(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                area = datastore['df']['area'][()]
                return area
            except Exception as e:
                print(e)
                return


### Model #####

class ExperimentManager:
    def __init__(self, root):
        """
        This is where we manage our experiments.
        It is also the interface through which the gui gets its data

        :param root: The root folder of the experiments. The directory tree must conform a specific structure that
        is specified in the fpi_config.json
        """
        self.root = root  # The root folder
        self._exp_paths = set()  # A set that contains the paths of the datastores
        self._experiments = {}  # A mapping between the name of an experiment and its path
        self.filtered = []  # A list that contains the filtered names of the experiments

        self._filters = []

        self.scan()

        pub.subscribe(self.filterAll, CHOICES_CHANGED)


    def scan(self):
        for path, dirs, files in os.walk(self.root):
            file_paths = [os.path.join(path, file) for file in files if file.endswith('h5')]
            [self._exp_paths.add(file) for file in file_paths]

        total = len(self._exp_paths)
        futures = []
        with ProcessPoolExecutor() as executor:
            for exp in self._exp_paths:
                res = executor.submit(self.check_if_valid, exp)
                futures.append(res)
        for fut in as_completed(futures):
            if fut.result() is not None:
                name = extract_name(os.path.basename(exp))

                self._experiments[name] = fut.result()
                val = 100 * (1 / total)
                pub.sendMessage(ANALYSIS_UPDATE, val = val)

        self.filtered = list(self._experiments.keys())
        pub.sendMessage(EXPERIMENT_LIST_CHANGED, choices=self.to_tuple())

    def check_if_valid(self, experiment_path):
        if check_datastore(experiment_path):
            return experiment_path
        else:
            return None

    def get_experiment(self, name: str) -> object:
        """
        This function takes a name of an experiment as it was extracted from its path. It uses it to get the path
        of the experiment datastore file, and then use this path to create an FPIExperiment object
        :param name: The name of the experiment as it was extracted from its path
        :return: An FPIExperiment object
        """
        experiment = self[name]
        animal_line, stimulus, treatment, genotype, filename = experiment.split(os.sep)[-5:]
        return FPIExperiment(name=name, path=experiment, animal_line=animal_line, stimulation=stimulus,
                             treatment=treatment, genotype=genotype)

    def filterLine(self, line):
        if line != '':
            self.filtered = [experiment for experiment in self.filtered if
                             self.get_experiment(experiment).animal_line == line]

    def filterTreatment(self, treatment):
        if treatment != '':
            self.filtered = [experiment for experiment in self.filtered if
                             self.get_experiment(experiment).treatment == treatment]

    def filterStimulus(self, stim):
        if stim != '':
            self.filtered = [experiment for experiment in self.filtered if
                             self.get_experiment(experiment).stimulation == stim]

    def filterGenotype(self, gen):
        if gen != '':
            self.filtered = [experiment for experiment in self.filtered if self.get_experiment(experiment).genotype == gen]

    def filterAll(self, selections):
        self.clear_filters()
        line, stimulation, treatment, genotype = selections
        self.filterLine(line)
        self.filterStimulus(stimulation)
        self.filterTreatment(treatment)
        self.filterGenotype(genotype)
        return self.to_tuple()



    def filterSelected(self, selected):
        self.filtered = list(self._experiments.keys())
        self.filtered = [filtered for filtered in self.filtered if filtered in selected]
        return self.to_tuple()

    def clear_filters(self):
        self.filtered = list(self._experiments.keys())
        return self.to_tuple()

    def to_tuple(self):
        res = []
        for exp in self.filtered:
            live = self.get_experiment(exp)
            print(live)
            res.append(fpi_meta._make((live.name, live.animal_line, live.stimulation, live.treatment, live.genotype)))
            print(res)
        return res

    def __getitem__(self, name):
        return self._experiments[name]

    def __iter__(self):
        return iter(self._experiments.keys())


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

    def __init__(self, name, path, animal_line, stimulation, treatment, genotype):
        self.animal_line = animal_line
        self.stimulation = stimulation
        self.treatment = treatment
        self.genotype = genotype
        self.name = name
        self._path = path

        self._parser = fpiparser(self._path)
        self._all_pixel = None
        self._response = None
        self._timecourse = None

    @property
    def all_pixel(self):
        if self._all_pixel is None:
            self._all_pixel = self._parser.all_pixel()
        return self._all_pixel

    @property
    def response(self):
        if self._response is None:
            self._response = self._parser.response()
        return self._response

    @property
    def timecourse(self):
        if self._timecourse is None:
            self._timecourse = self._parser.timecourse()
        return self._timecourse

    def baseline_mean(self, n_baseline=30):
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

    def response_latency(self, ratio=0.3, n_baseline=30):
        data = self.response
        if data is None:
            return
        mean_baseline = self.baseline_mean(n_baseline)
        latency = [(index, val) for index, val in enumerate(data[31:], n_baseline + 1) if
                   val > abs(1 + ratio) * mean_baseline]
        return latency

    def plot(self, ax, type):
        if type == 'response':
            self.plot_response(ax)
        elif type == 'latency':
            self.plot_response_latency(ax)
        elif type == 'timecourse':
            self.plot_timecourse(ax)
        else:
            raise ValueError('Unsupported plot type')

    def plot_response(self, ax):
        data = self._parser.response()
        x = range(len(data))
        ax.set_title(f'Response: {self.name}')
        ax.plot(x, data)

    def plot_response_latency(self, ax):
        data = self.response_latency()
        if data is None:
            return
        x = range(len(data))
        ax.plot(x, data, 'k-')

    def plot_timecourse(self, ax):
        data = self.timecourse
        if data is None:
            return
        x = range(len(data))
        ax.plot(x, data, 'k-')

    def __str__(self):
        return f'{self.name}: {self.animal_line} {self.stimulation} {self.treatment} {self.genotype}'

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
    root = app_config.base_dir
    manager = ExperimentManager(root)
    manager.filterLine('PTEN')
    manager.clear_filters()
    for item in manager.to_tuple():
        print(item)
