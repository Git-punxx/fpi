import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from collections import namedtuple
from abc import abstractmethod

'''
The point here is that we design a class hiearchy that will be able to handle different types
of files (csv, h5py) providing a united interface.

'''
fpi_meta = namedtuple('fpi_meta', 'mean std')


class FPIParser:
    def __init__(self, path):
        self._path = path
        self._data = None

    @abstractmethod
    def _parse_response(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_timecourse(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    def metadata(self):
        mean = self._data.mean()
        std = self._data.std()
        return fpi_meta(mean, std)


class CSVParser(FPIParser):
    def __init__(self, path):
        FPIParser.__init__(self, path)

    def _parse_response(self):
        data = []
        with open(self._path, 'r') as datafile:
            datafile.readline()
        for line in datafile:
            if ',' not in line:
                continue
            else:
                row = float(line.split(',')[1].rstrip())
                data.append(row)
        self._data = np.array(data)

    def _parse_timecourse(self):
        y = []
        with open(self._path) as f:
            f.readline()
            for line in f:
                if ',' not in line:
                    continue
                else:
                    y.append(float(line.split(',')[1]))
        self._data = np.array(y)

    def parse(self):
        '''
        see what kind of file we handle.
        if it is a response file use the _parse_response etc
        :return:
        '''
        if 'response' in self._path:
            self._parse_response()
        elif 'timecourse' in self._path:
            self._parse_response()
        else:
            raise ValueError
        return self._data


class HD5Parser(FPIParser):
    def __init__(self, path):
        FPIParser.__init__(self, path)

    def _parse_response(self):
        store = h5py.File(self._path, 'r')
        print(store.name)
        print(list(store.keys()))
        store.close()

    def _parse_timecourse(self):
        raise NotImplementedError

    def parse(self):
        print(f'Parsing {self._path}')
        self._parse_response()


def fpiparser(fname):
    print(fname)
    if fname.endswith('.csv'):
        return CSVParser(fname)
    elif fname.endswith('.h5'):
        return HD5Parser(fname)


class FPIHandler:
    def __init__(self, path):
        print('Initializing handler')
        self._path = path
        self.setup()

    def setup(self):
        print('Setting up handler')
        self._parser = fpiparser(self._path)
        self._data = self._parser.parse()

    def plot(self):
        plt.plot(self._data)
        plt.show()



if __name__ == '__main__':
    path = 'C:\\Users\\spyros\\Documents\\PythonProjects\\bi\\data\\'
    files = [f for f in os.listdir(path) if not os.path.isdir(f)]
    print(files)
    f = FPIHandler(files[0])
