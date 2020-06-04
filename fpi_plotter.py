from collections import namedtuple
from pandas import DataFrame
import seaborn as sns
'''
experiment_data = [gatherer.get_experiment(exp.name) for exp in experiment_list]
plotter = FPIPlotter(ax, experiment_data)
plotter.plot(plot_type)
'''
df_tuple = namedtuple('df_tuple', 'name path animal_line stimulation treatment genotype filter')
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


    def plot(self, plot_type):
        plot_registry[plot_type](self, self.experiments)


    @register('response')
    def plot_response(self, experiments):
        data = [df_tuple._make((exp.name, exp._path, exp.animal_line, exp.stimulation, exp.treatment, exp.genotype, exp.response)) for exp in experiments]
        df = DataFrame(data)
        ax = self.figure.subplots()
        for resp in df['filter']:
            ax.plot(resp)


    @register('baseline')
    def plot_baseline(self, experiments):
        data = [df_tuple._make((exp.name, exp._path, exp.animal_line, exp.stimulation, exp.treatment, exp.genotype, exp.mean_baseline)) for exp in experiments]
        df = DataFrame(data)
        ax = self.figure.subplots()
        print(df['filter'])
        sns.boxplot('filter', len(df['filter']), ax = ax, data = df)

    @register('peak_latency')
    def plot_peak_latency(self, experiments):
        data = [df_tuple._make((exp.name, exp._path, exp.animal_line, exp.stimulation, exp.treatment, exp.genotype, exp.peak_latency)) for exp in experiments]
        df = DataFrame(data)
        print(df)


    @register('response_latency')
    def plot_response_latency(self, experiments):
        data = [df_tuple._make((exp.name, exp._path, exp.animal_line, exp.stimulation, exp.treatment, exp.genotype, exp.response_latency)) for exp in experiments]
        df = DataFrame(data)
        print(df)

    @register('anat')
    def plot_anat(self, experiment):
        data = experiment[0].anat
        self.axes.pcolor(data)

