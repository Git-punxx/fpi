from typing import Type
from fpi_util import explain
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import sys
import h5py
from collections import namedtuple
from abc import abstractmethod
from app_config import config_manager as app_config
import re
from pubsub import pub
from pub_messages import ANALYSIS_UPDATE


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

def peak_latency(data):
    # it must return a frame
        #TODO Add 5 before and 5 after and find the mean frame integral
    response_region = data[30:75]
    peak = np.argmax(response_region)
    return peak + 30

def onset_threshold(mean_baseline, ratio = 0.3):
    if mean_baseline > 0:
        threshold = mean_baseline * (1 + ratio)
    elif mean_baseline < 0:
        threshold = mean_baseline * (1 - ratio)
    else:
        # TODO If mean baseline is zero Think about it
        raise ValueError('Mean baseline error')
    return threshold

def onset_latency(threshold, response, ratio=0.3, n_baseline=30):
    # returns frames. Find 5 syneomena pou na plhroun th sun8hkh 1 + 0.3 * basekube <  frmaw
    if response is None:
        return None
    try:
        latency = np.array([index for index, val in enumerate(response[n_baseline+1:], n_baseline + 1) if
                            val > threshold])
        if latency is None or len(latency) == 0:
            raise ValueError('Error on onset latency. Zero or none result')
        return latency[0]
    except Exception as e:
        print('Exception in onset_latency (fpi)')
        return None


def peak_latency(response, start = 30, end = 75):
    # it must return a frame
    if response is None:
        return
    response_region = response[start:end]
    peak = np.argmax(response_region)
    peak_value = np.max(response_region)
    return peak + 30


def fpiparser(path, root = 'df'):
    '''
    A factory method that tries to understand if we are dealing with a list of csv files or a single datastore file
    :param path: A list of filenames
    :return: A FPIParser object
    '''
    # Check if there is only one file and that it ends with h5. Then we return an HD%Parser
    if os.path.basename(path).endswith('h5'):
        return HD5Parser(extract_name(path), path, root)
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
        fname = [f for f in candidates if 'response_area' in f and name in f]
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
    def __init__(self, experiment, path, root = 'df'):
        FPIParser.__init__(self, experiment, path)
        self.root = root

    def has_roi(self):
        with h5py.File(self._path, 'r') as datastore:
            if 'roi/resp_map' in datastore:
                return True
            else:
                return False

    def is_analysis_complete(self):
        with h5py.File(self._path, 'r') as datastore:
            if 'df/resp_map' in datastore:
                return True
            else:
                return False



    def parser_type(self):
        return 'hdf5'

    def range(self):
        with h5py.File(self._path, 'r') as datastore:
            # here we need to see if we will use 'response' or 'resp_map'
            try:
                ys, ye, xs, xe = datastore['roi']['roi_range']
                return (slice(xs, xe), slice(ys, ye))
            except Exception as e:
                return (slice(None), slice(None))

    def response(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            # here we need to see if we will use 'response' or 'resp_map'
            try:
                if not roi:
                    data = datastore['df']['avg_df'][()]
                    return data
                else:
                    data = datastore['roi']['avg_df'][()]
                    return data
            except Exception as e:
                print(f'Exception in response method. {self._path} needs analysis')
                print(e)
                return None

    def stack(self, roi = False):
        '''
        Returns the average stack from the datastore. If a roi_range is in datastore too it will return a slice
        of the stack
        :return: A 3d np.array
        '''
        with h5py.File(self._path, 'r') as datastore:
            # here we need to see if we will use 'response' or 'resp_map'
            try:
                x_slice, y_slice = (None, None)
                if not roi:
                    data = datastore['df']['stack'][()] # This is the stack of average frames. It is a 3D array
                else:
                    x_slice, y_slice = self.range()
                    data = datastore['roi']['stack'][x_slice, y_slice, :] # This is the stack of average frames. It is a 3D array
            except Exception as e:
                print('Exception in stack method')
                data = datastore[self.root]['stack'][()]
            return data

    def norm_stack(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            try:
                x_slice, y_slice = (None, None)
                if roi:
                    x_slice, y_slice = self.range()
                    data = datastore['roi']['norm_stack'][x_slice, y_slice, :] # This is the stack of average frames. It is a 3D array
                else:
                    data = datastore['df']['norm_stack'][()]
            except Exception as e:
                print('Exception in norm_stack method')
                print(e)
                return None
            return data

    def timecourse(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            # Get the normalized stack
            # We need to get the normalized stack
            # normalize_stack(self.stack, self.n_baseline
            # self.stack is returned on avg_stack()
            # which is df['stack'] in the datastore
            x_slice, y_slice = (None, None)
            if roi:
                x_slice, y_slice = self.range()
                avg_stack = datastore['roi']['stack'][()]
            else:
                avg_stack = datastore['df']['stack'][()]

            try:
                df = normalize_stack(avg_stack)
                # Compute the mean
                df_avg = df.std((0, 1))
                df_std = df.mean((0, 1))
                # Compute the average
                timecourse = np.vstack((np.arange(1, df_avg.shape[0] + 1, dtype=np.intp), df_avg, df_std)).T
                return timecourse
            except Exception as e:
                return

    def area(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            if roi:
                root = 'roi'
            else:
                root = 'df'
            x_slice, y_slice = self.range()
            try:
                area = datastore[root]['area'][()]
                print(area)
                return area
            except Exception as e:
                print('Exception in all_pixel method')
                print(e)
                return

    def max_df(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            if roi:
                try:
                    data = datastore['roi']['max_df'][()]
                    return data
                except Exception as e:
                    print('Exception on max_df method')
                    print(e)
            else:
                try:
                    data = datastore['df']['max_df'][()]
                    return data
                except Exception as e:
                    print('Exception on max_df method')
                    print(e)
                return None

    def avg_df(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            if roi:
                try:
                    data = datastore['roi']['avg_df'][()]
                    return data
                except Exception as e:
                    print('Exception on max_df method')
                    print(e)
            else:
                try:
                    data = datastore['df']['avg_df'][()]
                    return data
                except Exception as e:
                    print('Exception on avg_df method')
                    print(e)
                return None

    def max_project(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                avg_df = datastore['df']['max_project'][()]
                return avg_df
            except Exception as e:
                print('Exception on avg_df method')
                print(e)
                return None

    def no_baseline(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                no_baseline = datastore['n_baseline'][()]
                return no_baseline
            except Exception as e:
                print('Exception on no_baseline method')
                print(e)
                return None

    def no_trials(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                no_trials = len(list(datastore['trials'].keys()))
                return no_trials
            except Exception as e:
                print('Exception on no_trials method')
                print(e)
                return None

    def anat(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                anat = datastore['anat'][()]
                return anat
            except Exception as e:
                print('Exception on anat method')
                print(e)
                return None

    def resp_map(self, roi = False):
        with h5py.File(self._path, 'r') as datastore:
            if roi:
                try:
                    data = datastore['roi']['resp_map'][()]
                    print('Fetching ROI data with shape {}', data.shape)
                    sys.stdout.flush()
                    return data
                except Exception as e:
                    print('Exception on resp_map method')
                    print(e)
            else:
                try:
                    data = datastore['df']['resp_map'][()]
                    print('Fetching normal data with shape {}', data.shape)
                    return data
                except Exception as e:
                    print('Exception on resp_map method')
                    print(e)
                return None


    def roi_range(self):
        with h5py.File(self._path, 'r') as datastore:
            try:
                roi = datastore['roi']['roi_range'][()]
                return roi
            except Exception as e:
                return None



class HDF5RoiWriter:
    def __init__(self, path):
        self._path = path

    def insert_into_group(self, data_dict):
        """
        :param data_dict: A dictionary that contains names of datasets and dataset to write into group
        :return: Nada
        """
        with h5py.File(self._path, 'r+') as datastore:
            if 'roi' not in datastore:
                datastore.create_group('roi')
            roi_grp = datastore['roi']
            print('Inserting ', data_dict.keys())
            for key, dataset in data_dict.items():
                if key in roi_grp.keys():
                    del roi_grp[key]
                roi_grp.create_dataset(key, data = dataset)

    def delete_roi(self):
        print('Deleting roi')
        with h5py.File(self._path, 'r+') as datastore:
            if 'roi' not in datastore:
                print(datastore.keys())
                return
            roi_grp = datastore['roi']
            for key, dataset in roi_grp.items():
                del roi_grp[key]

    def write_roi(self, roi):
        print(f'received {roi}')
        with h5py.File(self._path, 'r+') as datastore:
            if 'roi' not in datastore:
                print(datastore.keys())
                datastore.create_group('roi')
            roi_grp = datastore['roi']
            if 'roi_range' in roi_grp:
                del roi_grp['roi_range']
            roi_grp.create_dataset(name = 'roi_range', data =np.array(roi))


### Model #####

class ExperimentManager:
    def __init__(self, root):
        """
        This is where we manage our experiments.
        It is also the interface through which the gui gets its data

        :param root: The root folder of the experiments. The directory tree must conform a specific structure that
        is specified in the fpi_config.json
        """
        self.use_roi = False
        self.root = root  # The root folder
        self._exp_paths = set()  # A set that contains the paths of the datastores
        self._experiments = {}  # A mapping between the name of an experiment and its path
        self.filtered = []  # A list that contains the filtered names of the experiments

        self._filters = []

        self.scan()
        pub.subscribe(self.filterAll, CHOICES_CHANGED)



    def scan(self):
        print(f'Scanning folder {self.root}')
        for path, dirs, files in os.walk(self.root):
            file_paths = [os.path.join(path, file) for file in files if file.endswith('h5')]
            [self._exp_paths.add(file) for file in file_paths]

        futures = []
        #TODO Check if we are on linux, mac or windows and enable or disable the mulitprocessing
        #if not app_config.is_linux() and not app_config.is_mac():
        #    with ProcessPoolExecutor() as executor:
        #        for exp in self._exp_paths:
        #            res = executor.submit(self.check_if_valid, exp)
        #            futures.append(res)
        #    for fut in as_completed(futures):
        #        if fut.result() is not None:
        #            name = extract_name(os.path.basename(fut.result()))
        #            self._experiments[name] = fut.result()
        #else:
        for exp in self._exp_paths:
            if self.check_if_valid(exp):
                name = extract_name(os.path.basename(exp))
                self._experiments[name] = exp
            else:
                pass

        self.filtered = list(self._experiments.keys())
        pub.sendMessage(EXPERIMENT_LIST_CHANGED, choices=self.to_tuple())

    def check_if_valid(self, experiment_path):
        try:
            h5py.File(experiment_path, 'r')
            return True
        except OSError:
            return False

    def get_experiment(self, name: str) -> object:
        """
        This function takes a name of an experiment as it was extracted from its path. It uses it to get the path
        of the experiment datastore file, and then use this path to create an FPIExperiment object
        :param name: The name of the experiment as it was extracted from its path
        :return: An FPIExperiment object
        """
        experiment = self[name]
        animal_line, stimulus, treatment, genotype, filename = experiment.split(os.sep)[-5:]
        exp = FPIExperiment(name=name, path=experiment, animalline=animal_line, stimulation=stimulus,
                             treatment=treatment, genotype=genotype)
        exp._use_roi = self.use_roi
        return exp

    def filterLine(self, line):
        if line != '':
            self.filtered = [experiment for experiment in self.filtered if
                             self.get_experiment(experiment).animalline == line]

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
        print(len(selected))
        self.filtered = list(self._experiments.keys())
        self.filtered = [filtered for filtered in self.filtered if filtered in selected]
        print(self.to_tuple())
        return self.to_tuple()

    def clear_filters(self):
        print('Clearing selection')
        self.filtered = list(self._experiments.keys())
        return self.to_tuple()

    def to_tuple(self):
        res = []
        for exp in self.filtered:
            live = self.get_experiment(exp)
            try:
                res.append(fpi_meta._make((live.name, live.animalline, live.stimulation, live.treatment, live.genotype)))
            except Exception as e:
                print('Exception in as_tuple')
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
    3 csv files, each for each dataframe (resonse, timecourse, response_area)
    We should pass the name of the experiment only, NOT THE PATH, because if we need to read from three files, this
    operations should be handled by the parser.
    By providing the name of the experiment we could build the path using the config options
    '''

    def __init__(self, name, path, animalline, stimulation, treatment, genotype):
        self.animalline = animalline
        self.stimulation = stimulation
        self.treatment = treatment
        self.genotype = genotype
        self.name = name
        self._path = path


        self._response = None
        self._timecourse = None
        self._resp_map = None

        self._no_trials = None
        self._no_baseline = None
        self._response_area = None
        self._max_df = None
        self._avg_df = None
        self._mean_baseline = None
        self._peak_latency = None
        self._anat = None
        self._stack = None
        self._norm_stack = None
        self._roi = None
        self._area = None
        self._max_project = None

        self._use_roi = False

    def roi_slice(self):
        parser = fpiparser(self._path)
        return parser.range()

    def roi_area(self):
        if self.has_roi():
            parser = fpiparser(self._path)
            area = parser.area(roi = True)

    def get_root(self):
        return 'roi' if self._use_roi else 'df'

    @property
    def roi_range(self):
        parser = fpiparser(self._path)
        if self._roi is None:
            self._roi = parser.roi_range()
        return self._roi

    @property
    def area(self, roi = False):
        parser = fpiparser(self._path, self.get_root())
        if self._area is None:
            self._area = parser.area()
        return self._area

    def roi_area(self):
        parser = fpiparser(self._path, self.get_root())
        return parser.area(roi = True)


    @property
    def response(self, roi = False):
        parser = fpiparser(self._path, self.get_root())
        if self._response is None:
            self._response = parser.response(roi)
        return self._response

    @property
    def resp_map(self):
        parser = fpiparser(self._path, self.get_root())
        if self._resp_map is None:
            self._resp_map = parser.resp_map()
        return self._resp_map

    @property
    def norm_stack(self):
        parser = fpiparser(self._path, self.get_root())
        if self._norm_stack is None:
            self._norm_stack = parser.norm_stack()
        return self._norm_stack

    @property
    def timecourse(self):
        parser = fpiparser(self._path, self.get_root())
        if self._timecourse is None:
            self._timecourse = parser.timecourse()
        return self._timecourse

    @property
    def mean_baseline(self, n_baseline=30):
        if self._mean_baseline is None:
            data = self.response
            # Compute the mean of the baseline
            baseline = np.array(data[:n_baseline])
            self._mean_baseline = np.mean(baseline)
        return self._mean_baseline

    @property
    def max_project(self):
        parser = fpiparser(self._path, self.get_root())
        if self._max_project is None:
            self._max_project = parser.max_project()
        return self._max_project

    @property
    def peak_latency(self):
        # it must return a frame
        if self._peak_latency is None:
            self._peak_latency = peak_latency(self.response)
        return self._peak_latency

    @property
    def onset_threshold(self, ratio = 0.3, n_baseline = 30):
        data = self.response
        if data is None:
            return
        return onset_threshold(self.mean_baseline, ratio)

    @property
    def onset_latency(self, ratio=0.3, n_baseline=30):
        # returns frames. Find 5 syneomena pou na plhroun th sun8hkh 1 + 0.3 * basekube <  frmaw
        threshold = onset_threshold(self.mean_baseline)
        res = onset_latency(threshold, self.response)
        return res


    @property
    def no_trials(self):
        parser = fpiparser(self._path)
        if self._no_trials is None:
            self._no_trials = parser.no_trials()
        return self._no_trials

    @property
    def no_baseline(self):
        parser = fpiparser(self._path)
        if self._no_baseline is None:
            self._no_baseline = parser.no_baseline()
        return self._no_baseline

    @property
    def max_df(self):
        parser = fpiparser(self._path, self.get_root())
        if self._max_df is None:
            self._max_df = parser.max_df()
        return self._max_df

    @property
    def avg_df(self):
        parser = fpiparser(self._path, self.get_root())
        if self._avg_df is None:
            self._avg_df = parser.avg_df()
        return self._avg_df

    @property
    def area_df(self, percent = 0.5):
        # returns the pixels  (area) that are above the given percent * max_df/f
        # TODO add a button and a input field to enter the percentage
        area_df = np.sum(self.resp_map[self.resp_map > percent * self.max_df])
        print(area_df)
        return area_df

    @property
    def anat(self):
        parser = fpiparser(self._path)
        if self._anat is None:
            self._anat = parser.anat()
        return self._anat

    @property
    def stack(self):
        parser = fpiparser(self._path)
        if self._stack is None:
            self._stack = parser.stack()
        return self._stack

    @property
    def roi(self):
        parser = fpiparser(self._path)
        if self._roi is None:
            self._roi = parser.range()
        return self._roi

    def has_stack(self):
        return self.stack is not None

    def has_roi(self):
        parser = fpiparser(self._path)
        return parser.has_roi()


    def clear(self):
        self._response = None
        self._timecourse = None
        self._no_trials = None
        self._no_baseline = None
        self._response_area = None
        self._max_df = None
        self._avg_df = None
        self._mean_baseline = None
        self._peak_latency = None
        self._anat = None
        self._stack = None

    def __str__(self):
        return f'{self.name}: {self.animalline} {self.stimulation} {self.treatment} {self.genotype}'

    def is_roi_analyzed(self):
        parser = fpiparser(self._path)
        return parser.has_roi()

    def is_analysis_complete(self):
        parser = fpiparser(self._path)
        return parser.is_analysis_complete()

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
        if self.area is not None:
            result.append('response_area')
        else:
            result.append('No all_pixelsj')
        return result

    def peak_response(self):
        '''
        Compute the frame and the value of the max df/f
        We use the avg_df of the datastore
        :return: (int, float)
        '''
        df = self.response
        frame = np.argmax(df)
        val = np.max(df)
        return frame, val

    def baseline_value(self, no_baseline = 30):
        '''
        :param no_baseline: Where does the baseline stops
        :return: int, float
        '''
        return no_baseline, self.response[no_baseline]

    def halfwidth(self, no_baseline = 30):
        response_curve = self.response

        base_frame, baseline_val = self.baseline_value()
        peak_frame, peak_val = self.peak_response()

        half_val = (peak_val - baseline_val)/2

        med_line = np.zeros_like(response_curve[no_baseline:])
        med_line[()] = half_val

        idx = np.argwhere(np.diff(np.sign(response_curve[no_baseline:] - med_line))).flatten()
        if len(idx) < 2:
            return (0, 0), 0
        halfwidth_start, *_, halfwidth_end = idx
        print(f'Response value at {halfwidth_start + no_baseline} to {halfwidth_end} = {response_curve[idx + no_baseline]}')
        return halfwidth_end - halfwidth_start


    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        ids = (np.abs(array - value)).argmin()
        return ids

    @staticmethod
    def log_tuple(msg, tup):
        print(f'{msg}: {tup[0]}: {tup[1]}')





if __name__ == '__main__':
    root = app_config.base_dir
    manager = ExperimentManager(root)
    exp = '20190924_1613'
    live = manager.get_experiment(exp)
    print(live.halfwidth())
