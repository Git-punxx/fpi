from app_config import config_manager, Genotype, AnimalLine, Stimulation, Treatment
import fpi_util
from numpy import arange
from collections import defaultdict
from itertools import cycle
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
boxoplot_colors = cycle(['cyan', 'khakki'])

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


    def _plot_dict(self, genotype_dict):
        no_subplots = len(genotype_dict.keys())
        print(f'Creating {no_subplots} subplots')
        axes = self.figure.subplots(1, len(genotype_dict.keys()), sharey = True)
        if no_subplots > 1:
            for ax, gen in zip(axes, genotype_dict.keys()):
                if len(genotype_dict[gen].values()) == 0:
                    continue
                ax.boxplot(genotype_dict[gen].values(), labels = [gen.name for gen in genotype_dict[gen].keys()], patch_artist = True)
                ax.set_xlabel(gen.name)
                ax.grid(True, alpha = 0.1)
        else:
            for gen in genotype_dict.keys():
                axes.boxplot(genotype_dict[gen].values(), labels=[gen.name for gen in genotype_dict[gen].keys()],
                             patch_artist=True)
                axes.set_xlabel(gen.name)
                axes.grid(True, alpha=0.1)

    @register('response')
    def plot_response(self, experiments, choice):
        ax = self.figure.subplots()
        ax.grid(True, alpha = 0.1)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Response')
        values = [(exp.response[2:-1], exp.name) for exp in experiments]
        for data, name in values:
            ax.plot(range(3, 81), data, label = name, linewidth = 2)
        ax.legend()

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
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.mean_baseline for exp in exp_list if exp.mean_baseline is not None]

        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        self._plot_dict(genotype_dict)
        path = config_manager.csv_dir
        [DataFrame(item).to_csv(f'{path}/{key}_baseline.csv') for key, item in genotype_dict.items()]




    @register('peak_latency')
    def plot_peak_latency(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])

        # Loading the data into the genotype dict
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.peak_latency[1] for exp in exp_list if exp.peak_latency is not None]

        fpi_util.clear_data(genotype_dict)
        self._plot_dict(genotype_dict)
        path = config_manager.csv_dir
        [DataFrame(item).to_csv(f'{path}/{key}_peak_latency.csv') for key, item in genotype_dict.items()]



    @register('response_latency')
    def plot_onset_latency(self, experiments, choice):
        #TODO Check this functions because it causes an exception
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.response_latency for exp in exp_list if exp.response_latency is not None]

        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        print(genotype_dict)
        self._plot_dict(genotype_dict)
        [DataFrame(item).to_csv(f'../csv/{key}_response_latency.csv') for key, item in genotype_dict.items()]

    @register('peak_value')
    def plot_peak_value(self, experiments, choice):
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.peak_latency[0] for exp in exp_list if exp.peak_latency is not None]
        fpi_util.clear_data(genotype_dict)
        self._plot_dict(genotype_dict)
        path = config_manager.csv_dir
        [DataFrame(item).to_csv(f'{path}/{key}_peak_value.csv') for key, item in genotype_dict.items()]

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
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter] = [exp.response_area for exp in exp_list if exp.response_area is not None]
        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        self._plot_dict(genotype_dict)
        path = config_manager.csv_dir
        [DataFrame(item).to_csv(f'{path}/{key}_area.csv') for key, item in genotype_dict.items()]
