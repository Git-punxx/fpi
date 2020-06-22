from app_config import config_manager, Genotype, AnimalLine, Stimulation, Treatment
import fpi_util
from numpy import arange
from collections import defaultdict
from pandas import DataFrame

from gui.dialogs import DataPathDialog
import pickle
import os
from pandas import DataFrame
'''
experiment_data = [gatherer.get_experiment(exp.name) for exp in experiment_list]
plotter = FPIPlotter(ax, experiment_data)
plotter.plot(plot_type)
'''
plot_registry = {}

def register(plot_type):
    def deco(func):
        plot_registry[plot_type] = func
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return deco


class FPIPlotter:
    def __init__(self, figure, experiments):
        self.figure = figure
        self.experiments = experiments


    def sanitize(self, data):
        for key, val in data.items():
            if val is None:
                del data[key]

    def plot(self, plot_type, choice = None):
        plot_registry[plot_type](self, self.experiments, choice)


    @register('response')
    def plot_response(self, experiments, choice):
        ax = self.figure.subplots()
        ax.grid(True, alpha = 0.5)
        values = [exp.response[:-1] for exp in experiments]
        for d in values:
            ax.plot(d)

    @register('baseline')
    def plot_baseline(self, experiments, choice):
        """

        :param experiments: FPIExperiment object list
        :param choice: A string returned from the util.BoxPlotChoices panel
        :return:
        """
        # Get the options for the current category
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.mean_baseline for exp in exp_list if exp.mean_baseline is not None]

        '''
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)
        print(genotype_dict.keys())
        print(f'No axes: {len(axes)}')

        for ax, gen in zip(axes, genotype_dict.keys()):
            if len(genotype_dict[gen].values()) == 0:
                continue
            ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
            ax.set_xlabel(gen.name)
            ax.grid(True, alpha = 0.5)
        '''
        DataFrame(genotype_dict).to_csv('baseline.csv')




    @register('peak_latency')
    def plot_peak_latency(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.peak_latency[1] for exp in exp_list if exp.peak_latency is not None]
        '''
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)

        for ax, gen in zip(axes, genotype_dict.keys()):
            if len(genotype_dict[gen].values()) == 0:
                continue
            ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
            ax.set_xlabel(gen.name)
            ax.grid(True, alpha = 0.5)
        '''
        DataFrame(genotype_dict).to_csv('peak_latency.csv')



    @register('onset_latency')
    def plot_onset_latency(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [30 + exp.onset_latency for exp in exp_list if exp.onset_latency is not None]

        '''
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)

        for ax, gen in zip(axes, genotype_dict.keys()):
            if len(genotype_dict[gen].values()) == 0:
                continue
            ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
            ax.set_xlabel(gen.name)
            ax.grid(True, alpha = 0.5)
        '''
        DataFrame(genotype_dict).to_csv('onset_latency.csv')

    @register('peak_value')
    def plot_peak_value(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.peak_latency[0] for exp in exp_list if exp.peak_latency is not None]

        '''
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)

        for ax, gen in zip(axes, genotype_dict.keys()):
            if len(genotype_dict[gen].values()) == 0:
                continue
            ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
            ax.set_xlabel(gen.name)
            ax.grid(True, alpha = 0.5)
        '''
        DataFrame(genotype_dict).to_csv('peak_value.csv')

    @register('anat')
    def plot_anat(self, experiment, choice):
        ax = self.figure.subplots()
        data = experiment[0].anat
        ax.pcolor(data)

    @register('area')
    def plot_area(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.area for exp in exp_list if exp.area is not None]
        '''
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)

        for ax, gen in zip(axes, genotype_dict.keys()):
            if len(genotype_dict[gen].values()) == 0:
                continue
            ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
            ax.set_xlabel(gen.name)
            ax.grid(True, alpha = 0.5)
        '''
        DataFrame(genotype_dict).to_csv('area.csv')
