import matplotlib as mpl

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


    def plot(self, plot_type):
        plot_registry[plot_type](self, self.experiments)


    @register('response')
    def plot_response(self, experiments):
        data = [exp.response for exp in experiments]
        for d in data:
            self.axes.plot(d)

    @register('baseline')
    def plot_baseline(self, experiments):
        self.axes.set_title('Mean baseline')
        self.axes.set_xlabel('something here')
        self.axes.set_ylabel('And something here ()')
        vals = [exp.mean_baseline for exp in experiments]
        self.axes.boxplot(vals )

    @register('peak_latency')
    def plot_peak_latency(self, experiments):
        self.axes.set_axisbelow(True)
        self.axes.set_title('Peak latency')
        self.axes.set_xlabel('Distribution')
        self.axes.set_ylabel('Latency ()')
        data_dict = [exp.peak_latency[1] for exp in experiments]
        self.axes.boxplot(data_dict)


    @register('response_latency')
    def plot_response_latency(self, experiments):
        data = [exp.response_latency() for exp in experiments]
        for item in data:
            d = [p[1] for p in item]
            self.axes.plot(d)

    @register('anat')
    def plot_anat(self, experiment):
        data = experiment[0].anat
        self.axes.pcolor(data)

