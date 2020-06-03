from app_config import config_manager, Genotype, AnimalLine, Stimulation, Treatment
import fpi_util
from numpy import arange
from collections import defaultdict

from gui.dialogs import DataPathDialog
import pickle
import os
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
    def __init__(self, axes, experiments):
        self.axes = axes
        self.experiments = experiments


    def sanitize(self, data):
        for key, val in data.items():
            if val is None:
                del data[key]

    def plot(self, plot_type, choice = None):
        plot_registry[plot_type](self, self.experiments, choice)


    @register('response')
    def plot_response(self, experiments, choice):
        values = [exp.response for exp in experiments]
        for d in values:
            self.axes.plot(d)

    @register('baseline')
    def plot_baseline(self, experiments, choice):
        """

        :param experiments: FPIExperiment object list
        :param choice: A string returned from the util.BoxPlotChoices panel
        :return:
        """
        self.axes.set_axisbelow(True)
        self.axes.set_title('Mean Baseline')
        self.axes.set_xlabel('Distribution')
        self.axes.set_ylabel('.... ()')

        # Get the options for the current category
        filter_dict = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment and assign them to the genotype categories
        genotype_dict = defaultdict(dict)
        genotype_dict.update((k, {}) for k in [item for item in Genotype])
        print(genotype_dict)
        for base_filter, genotypes in filter_dict.items():
            for genotype, exp_list in genotypes.items():
                genotype_dict[genotype][base_filter] = []
                genotype_dict[genotype][base_filter].append([exp.mean_baseline for exp in exp_list if exp.mean_baseline is not None])

        fpi_util.clear_data(genotype_dict)
        # Compute the positions of the boxplots
        for gen, filter_list in genotype_dict.items():
            for key, val in filter_list.items():
                print(f'Plotting {key} with label {key.name}')
                bp = self.axes.boxplot(val, labels = [key.name], patch_artist = True)

        self.axes.legend()

        self.axes.set_xticklabels([item.name for item in Genotype])
        self.axes.set_xticks([1, 4, 7])



    @register('peak_latency')
    def plot_peak_latency(self, experiments, choice):

        self.axes.set_axisbelow(True)
        self.axes.set_title('Peak latency')
        self.axes.set_xlabel('Distribution')
        self.axes.set_ylabel('Latency ()')

        # Get the options for the current category
        genotypes = config_manager.genotypes
        data = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment
        for base_filter, genotypes in data.items():
            for genotype, exp_list in genotypes.items():
                data[base_filter][genotype] = [exp.peak_latency for exp in exp_list]

        # Compute the positions of the boxplots
        no_genotypes = len(genotypes)
        no_filters = len(data.keys())

        positions = arange(no_filters * no_genotypes).reshape(no_genotypes, no_filters).T
        colors = ['pink', 'lightblue', 'lightgreen']
        for (filter, genotypes_list), position in zip(data.items(), positions):
            gen_data = [data[filter][gen] for gen in genotypes_list]
            plot = self.axes.boxplot(gen_data, positions = position, widths = 0.5, patch_artist = True)
            for color, patch in enumerate(plot['boxes']):
                patch.set(facecolor= colors[color])

        self.axes.set_xticklabels(genotypes)
        self.axes.set_xticks([1.5, 4.5, 7.5])

    @register('response_latency')
    def plot_response_latency(self, experiments, choice):
        data = [exp.response_latency() for exp in experiments]
        for item in data:
            d = [p[1] for p in item]
            self.axes.plot(d)

    @register('anat')
    def plot_anat(self, experiment, choice):
        data = experiment[0].anat
        self.axes.pcolor(data)

