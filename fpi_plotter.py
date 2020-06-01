import matplotlib as mpl
from app_config import config_manager
from collections import Counter, defaultdict
import fpi_util
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
        # Get the options for the current category

        data = fpi_util.categorize(experiments, choice)
        for base_filter, genotypes in data.items():
            for genotype, exp_list in genotypes.items():
                data[base_filter][genotype] = [exp.mean_baseline for exp in exp_list]


        for index, line in enumerate(data.keys()):
            for genotype, exps in data[line].items():
                self.axes.boxplot(exps, positions = [index], widths = 0.6)

        self.axes.set_xticklabels(genotypes)
        self.axes.set_xticks([1.5, 4.5, 7.5])

    @register('peak_latency')
    def plot_peak_latency(self, experiments, choice):
        self.axes.set_axisbelow(True)
        self.axes.set_title('Peak latency')
        self.axes.set_xlabel('Distribution')
        self.axes.set_ylabel('Latency ()')

        data = fpi_util.categorize(experiments, choice)
        for base_filter, genotypes in data.items():
            for genotype, exp_list in genotypes.items():
                data[base_filter][genotype] = [exp.peak_latency[1] for exp in exp_list]


        for index, line in enumerate(data.keys()):
            for genotype, exps in data[line].items():
                self.axes.boxplot(exps, positions = [index], widths = 0.6)

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

