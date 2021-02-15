import wx
import subprocess
from gui.dialogs import *
from app_config import config_manager as app_config
import intrinsic
from gui.dialogs import Preferences
import matplotlib.pyplot as plt
import numpy as np
from fpi import HD5Parser
import fpi



# Create a bunch of Id's to use
ID_CHECK_FOLDERS = wx.NewId()
ID_CREATE_FOLDERS = wx.NewId()
ID_CONFIG_RESPONSE_PLOT = wx.NewId()
ID_SET_DATABASE_DIR = wx.NewId()
ID_PREFERENCES = wx.NewId()
ID_STATS = wx.NewId()

ID_EXPORT_PEAK_VALUES = wx.NewId()
ID_EXPORT_ONSET_LATENCY = wx.NewId()
ID_EXPORT_ONSET_THRESHOLD = wx.NewId()
ID_EXPORT_PEAK_LATENCY = wx.NewId()

ID_ROI_EXPORT_PEAK_VALUES = wx.NewId()
ID_ROI_EXPORT_ONSET_LATENCY = wx.NewId()
ID_ROI_EXPORT_ONSET_THRESHOLD = wx.NewId()
ID_ROI_EXPORT_PEAK_LATENCY = wx.NewId()

ID_EXPORT_ROI_ATTRIBUTES = wx.NewId()

ID_ROI_PLOT_RESPONSE = wx.NewId()

# Event ids
ID_PLOT_MEAN_BASELINE = wx.NewId()

ID_MEAN_RESPONSE = wx.NewId()

# Analysis ids
ID_INTRINSIC_ANALYSIS = wx.NewId()




# A dict that associates wx.Ids with functions
# We will use a decorator to populate it
command_registry = {}

def register(wx_id):
    def deco(func):
        global command_registry
        command_registry[wx_id] = func
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return deco

file_menu = [(wx.ID_OPEN, 'Open\tCtrl+o'),
             (wx.ID_CLOSE, 'Close\tCtrl+x')
             ]
edit_menu = [(wx.ID_COPY, 'Copy\tCtrl+c'),
             (wx.ID_CUT, 'Cut\tCtrl+x'),
             (wx.ID_PASTE, 'Paste\tCtrl+v')
             ]

options_menu = [(ID_CREATE_FOLDERS, 'Create folder structure'),
                (ID_CHECK_FOLDERS, 'Check folder structure'),
                (ID_CONFIG_RESPONSE_PLOT, 'Configure response plot'),
                (ID_SET_DATABASE_DIR, 'Set experiments folder'),
                (ID_PREFERENCES, 'Preferences..')
                 ]

plot_menu = [(ID_STATS, 'Plot total stats'),
             (ID_MEAN_RESPONSE, 'Plot Mean Response')]

roi_plot_menu = [(ID_ROI_PLOT_RESPONSE, 'Plot Response')]

intrincic_menu = [(ID_INTRINSIC_ANALYSIS, 'Intrinsic analysis')]

export_menu = [(ID_EXPORT_PEAK_VALUES, 'Peak Values'),
               (ID_EXPORT_PEAK_LATENCY, 'Peak Latency'),
               (ID_EXPORT_ONSET_LATENCY, 'Onset Latency'),
               (ID_EXPORT_ONSET_THRESHOLD, 'Onset Threshold')]

roi_export_menu = [(ID_ROI_EXPORT_PEAK_VALUES, 'ROI Peak Values'),
                  (ID_ROI_EXPORT_PEAK_LATENCY, 'ROI Peak Latency'),
                  (ID_ROI_EXPORT_ONSET_LATENCY, 'ROI Onset latency'),
                  (ID_ROI_EXPORT_ONSET_THRESHOLD, 'ROI Onset Threshold'),
                  (ID_EXPORT_ROI_ATTRIBUTES, 'ROI Attributes')]

class FPIMenuBar(wx.MenuBar):
    def __init__(self):
        wx.MenuBar.__init__(self)
        self.setup()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def setup(self):
        global file_menu
        global edit_menu
        global options_menu
        global intrincic_menu
        global plot_menu
        global export_menu
        self.FileMenu(file_menu)
        self.EditMenu(edit_menu)
        self.OptionsMenu(options_menu)
        self.AnalysisMenu(intrincic_menu)
        self.PlotMenu(plot_menu)
        self.ExportMenu(export_menu)

    def OnMenu(self, event):
        try:
            command_registry[event.GetId()](self)
        except KeyError as e:
            ErrorDialog('This action is not yet implemented')
            print(f'Error {e}')

    def FileMenu(self, items):
        filem = wx.Menu()
        for item_id, description in items:
            filem.Append(item_id, description)
        self.Append(filem, '&File')

    def EditMenu(self, items):
        editm = wx.Menu()
        for item_id, description in items:
            editm.Append(item_id, description)
        self.Append(editm, '&Edit')

    def OptionsMenu(self, items):
        optionsm = wx.Menu()
        for item_id, description in items:
            optionsm.Append(item_id, description)
        self.Append(optionsm, 'Options')

    def PlotMenu(self, items):
        plotm = wx.Menu()
        for item_id, description in items:
            plotm.Append(item_id, description)
        self.Append(plotm, 'Plot')

    def AnalysisMenu(self, items):
        analysism = wx.Menu()
        for item_id, description in items:
            analysism.Append(item_id, description)
        self.Append(analysism, 'Intrinsic Analysis')

    def ExportMenu(self, items):
        export_menu = wx.Menu()
        for item_id, description in items:
            export_menu.Append(item_id, description)
        self.Append(export_menu, 'Export')



class FPIImageMenu(wx.MenuBar):
    def __init__(self):
        super().__init__()
        self.setup()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def setup(self):
        global roi_plot_menu
        global roi_export_menu
        self.PlotMenu(roi_plot_menu)
        self.ExportMenu(roi_export_menu)



    def OnMenu(self, event):
        try:
            command_registry[event.GetId()](self)
        except KeyError as e:
            ErrorDialog('This action is not yet implemented')
            print(f'Error {e}')

    def PlotMenu(self, items):
        plotm = wx.Menu()
        for item_id, description in items:
            plotm.Append(item_id, description)
        self.Append(plotm, 'Plot')

    def ExportMenu(self, items):
        export_menu = wx.Menu()
        for item_id, description in items:
            export_menu.Append(item_id, description)
        self.Append(export_menu, 'Export')


class FPIIslandMenu(wx.MenuBar):
    pass



@register(ID_SET_DATABASE_DIR)
def SetDataPath(parent):
    path = DataPathDialog(parent, 'Select experiments folder ')
    if path is not None:
        # here we should check if the directory contains the proper data structure
        # if not we should offer to create it
        app_config.base_dir = path
        print(parent)
        parent = parent.GetParent()
        parent.gatherer.scan()
        parent.exp_list.clear()
        parent.exp_list.add_rows(parent.gatherer.to_tuple())
    else:
        ErrorDialog('Could not set requested path')

@register(ID_CREATE_FOLDERS)
def CreateFolderStructure(parent):
    app_config.create_folders()

@register(ID_INTRINSIC_ANALYSIS)
def RunIntrinsic(parent):
    print('Running intrinsic')
    subprocess.run(['python', '../intrinsic/explorer.py'])

@register(ID_PREFERENCES)
def LaunchPreferences(parent):
    with Preferences(None, 'Preferences') as dlg:
        dlg.ShowModal()



@register(ID_STATS)
def TotalStats(parent):
    gatherer = parent.GetTopLevelParent().gatherer
    attributes = ['max_df', 'mean_baseline', 'peak_latency']
    experiments = [exp for exp in gatherer]
    boxplot_dict = {attr: [] for attr in attributes}
    for exp in experiments:
        for attr in attributes:
            boxplot_dict[attr].append(getattr(gatherer.get_experiment(exp), attr))

    fig, axes = plt.subplots(1, len(attributes))
    for ax, (attr, value_list) in zip(axes, boxplot_dict.items()):
        ax.boxplot(value_list)
        ax.set_title(attr)

    plt.show()
    '''    
    axes.boxplot(genotype_dict[gen].values(), labels=[gen.name for gen in genotype_dict[gen].keys()],
                 patch_artist=True)
    axes.set_xlabel(gen.name)
    axes.grid(True, alpha=0.1)
    '''

@register(ID_EXPORT_PEAK_VALUES)
def ExportPeakValue(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    peak_value = {exp.name: exp.max_df for exp in selected}
    print(peak_value)
    save_series('peak_value', peak_value)


@register(ID_EXPORT_PEAK_LATENCY)
def ExportPeakLatency(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    peak_latency = {exp.name: exp.peak_latency for exp in selected}
    print(peak_latency)
    save_series('peak_latency', peak_latency)


@register(ID_EXPORT_ONSET_LATENCY)
def ExportOnsetLatency(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    onset_latency = {exp.name: exp.onset_latency for exp in selected}
    print(onset_latency)
    save_series('peak_latency', onset_latency)

@register(ID_EXPORT_ONSET_THRESHOLD)
def ExportOnsetThreshold(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    onset_threshold = {exp.name: exp.onset_threshold for exp in selected}
    save_series('onset_threshold', onset_threshold)

@register(ID_MEAN_RESPONSE)
def MeanResponse(parent):
    exp_list = parent.GetTopLevelParent().exp_list
    gatherer = parent.GetTopLevelParent().gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]
    response_stack = np.array([exp.response for exp in selected if exp.response.shape == (81,)])
    res = response_stack.mean(axis = 0)

    plt.plot(res)
    plt.title('Mean response')
    plt.xlabel('Frames')
    plt.ylabel('dF/F')
    plt.show()



@register(ID_ROI_EXPORT_PEAK_VALUES)
def ExportRoiPeakValues(parent):
    exp = parent.GetParent()._experiment
    parser = HD5Parser(exp, exp._path)
    peak_value = parser.max_df(roi = True)
    if peak_value is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style = wx.ICON_ERROR)
        return
    print(peak_value)
    save_series('roi_peak_value', peak_value)


@register(ID_ROI_EXPORT_ONSET_THRESHOLD)
def ExportRoiThreshold(parent):
    exp = parent.GetParent()._experiment
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi = True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style = wx.ICON_ERROR)
        return
    #TODO Fixme
    threshold = fpi.onset_threshold(roi_response)
    print(threshold)
    save_series('roi_peak_threshold', threshold)


@register(ID_ROI_EXPORT_PEAK_LATENCY)
def ExportROIPeakLatency(parent):
    exp = parent.GetParent()._experiment
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi = True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style = wx.ICON_ERROR)
        return
    #TODO Fixme
    peak_latency = fpi.peak_latency(roi_response)
    print(peak_latency)
    save_series('roi_peak_latency', peak_latency)

@register(ID_ROI_EXPORT_ONSET_LATENCY)
def ExportROIOnsetLatency(parent):
    exp = parent.GetParent()._experiment
    mean_baseline = exp.mean_baseline
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi = True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style = wx.ICON_ERROR)
        return
    #TODO Fixme
    threshold = fpi.onset_threshold(mean_baseline)
    onset_latency = fpi.onset_latency(mean_baseline, roi_response)
    print(onset_latency)
    save_series('roi_onset_latency', threshold)


@register(ID_ROI_PLOT_RESPONSE)
def PlotROIResponse(parent):
    exp = parent.GetParent()._experiment
    parser = HD5Parser(exp, exp._path)
    roi_resp = parser.response(roi = True)
    resp = parser.response()
    plt.plot(resp, label = 'Full image response')
    plt.plot(roi_resp, label = 'ROI response')
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('dF/f')
    plt.show()

@register(ID_EXPORT_ROI_ATTRIBUTES)
def ExportROIAttributes(parent):
    exp = parent.GetParent()._experiment
    parser = HD5Parser(exp, exp._path)
    resp = parser.response(roi = True)
    if resp is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style = wx.ICON_ERROR)
        return
    mean_baseline = exp.mean_baseline
    threshold = fpi.onset_threshold(exp.mean_baseline)
    onset_latency = fpi.onset_latency(threshold, resp)
    peak_latency = fpi.peak_latency(resp)
    peak_val = resp.max()
    attrs = {'Peak value': peak_val,
             'Peak latency': peak_latency,
             'Mean Baseline': mean_baseline,
            'Onset Threshold': threshold,
            'Onset Latency': onset_latency}
    fname = f'ROI_attrs_{exp.name}'
    try:
        save_series(fname, attrs)
    except Exception as e:
        wx.MessageBox(f'Export failed: {e}', 'Export failed', style=wx.OK | wx.ICON_ERROR)
        return
    wx.MessageBox(f'ROI attributes saved at {fname}.csv', 'Attributes saved', style = wx.OK | wx.ICON_INFORMATION)



def save_series(fname, series: dict):
    fname = app_config.data_export_dir + '/' + fname + '.csv'
    with open(fname, 'w') as fd:
        for key, val in series.items():
            fd.write(f'{key},{val}\n')


