from app_config import config_manager
import fpi_util
from numpy import arange
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
        self.axes.set_axisbelow(True)
        self.axes.set_title('Mean Baseline')
        self.axes.set_xlabel('Distribution')
        self.axes.set_ylabel('.... ()')

        # Get the options for the current category
        genotypes = config_manager.genotypes
        data = fpi_util.categorize(experiments, choice)

        # Get the actual data from the fpiexperiment
        for base_filter, genotypes in data.items():
            for genotype, exp_list in genotypes.items():
                data[base_filter][genotype] = [exp.mean_baseline for exp in exp_list]

        # Compute the positions of the boxplots
        no_genotypes = len(genotypes)
        no_filters = len(data.keys())
        filter_positions = {val.lower(): key for key, val in dict(enumerate(data.keys(), 1)).items()}
        genotype_positions = {val.lower(): key for key, val in dict(enumerate(genotypes, 1)).items()}

        positions = arange(no_filters * no_genotypes).reshape(no_genotypes, no_filters).T
        colors = ['pink', 'lightblue', 'lightgreen', 'khaki']
        for (filter, genotypes_list), position in zip(data.items(), positions):
            for gen in genotypes_list:
                d = data[filter][gen]
                plot = self.axes.boxplot(d, positions = [filter_positions[filter] * genotype_positions[gen] + genotype_positions[gen]], widths = 0.5, patch_artist = True)
                plot['boxes'][0].set(facecolor= colors[filter_positions[filter]])

        self.axes.set_xticklabels(genotypes)
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

