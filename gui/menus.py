import wx
from light_analyzer import do_analysis
import subprocess
from gui.dialogs import *
from app_config import config_manager as app_config
import app_config
import modified_intrinsic
from gui.dialogs import Preferences
import matplotlib.pyplot as plt
import numpy as np
from fpi import HD5Parser
import fpi
import pandas as pd

mgr = app_config.config_manager

# Create a bunch of Id's to use
ID_CHECK_FOLDERS = wx.NewId()
ID_CREATE_FOLDERS = wx.NewId()
ID_CONFIG_RESPONSE_PLOT = wx.NewId()
ID_SET_DATABASE_DIR = wx.NewId()
ID_SET_RAW_DIR = wx.NewId()
ID_PREFERENCES = wx.NewId()
ID_STATS = wx.NewId()

ID_EXPORT_RESPONSE = wx.NewId()
ID_EXPORT_PEAK_VALUES = wx.NewId()
ID_EXPORT_ONSET_LATENCY = wx.NewId()
ID_EXPORT_ONSET_THRESHOLD = wx.NewId()
ID_EXPORT_PEAK_LATENCY = wx.NewId()
ID_EXPORT_HALFWIDTH = wx.NewId()

ID_ROI_EXPORT_RESPONSE = wx.NewId()
ID_ROI_EXPORT_PEAK_VALUES = wx.NewId()
ID_ROI_EXPORT_ONSET_LATENCY = wx.NewId()
ID_ROI_EXPORT_ONSET_THRESHOLD = wx.NewId()
ID_ROI_EXPORT_PEAK_LATENCY = wx.NewId()
ID_ROI_EXPORT_HALFWIDTH = wx.NewId()

ID_EXPORT_ROI_ATTRIBUTES = wx.NewId()

ID_ROI_PLOT_RESPONSE = wx.NewId()

# Event ids
ID_PLOT_MEAN_BASELINE = wx.NewId()

ID_MEAN_RESPONSE = wx.NewId()

# Analysis ids
ID_INTRINSIC_ANALYSIS = wx.NewId()
ID_NEW_EXPERIMENT_ANALYSIS = wx.NewId()
ID_TIFF_MULTIANALYSIS = wx.NewId()

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

options_menu = [(ID_SET_DATABASE_DIR, 'Set experiments folder'),
                (ID_SET_RAW_DIR, 'Set root trials folder')
                ]

plot_menu = [(ID_STATS, 'Plot total stats'),
             (ID_MEAN_RESPONSE, 'Plot Mean Response')]

roi_plot_menu = [(ID_ROI_PLOT_RESPONSE, 'Plot Response')]

intrincic_menu = [(ID_INTRINSIC_ANALYSIS, 'Intrinsic analysis'),
                  (ID_NEW_EXPERIMENT_ANALYSIS, 'Analyze new experiment folder'),
                  (ID_TIFF_MULTIANALYSIS, 'TIFF Multianalysis')]

export_menu = [(ID_EXPORT_RESPONSE, 'Reponse'),
               (ID_EXPORT_PEAK_VALUES, 'Peak Values'),
               (ID_EXPORT_PEAK_LATENCY, 'Peak Latency'),
               (ID_EXPORT_ONSET_LATENCY, 'Onset Latency'),
               (ID_EXPORT_ONSET_THRESHOLD, 'Onset Threshold'),
               (ID_EXPORT_HALFWIDTH, 'Halfwidth')]

roi_export_menu = [(ID_ROI_EXPORT_RESPONSE, 'ROI Response'),
                   (ID_ROI_EXPORT_PEAK_VALUES, 'ROI Peak Values'),
                   (ID_ROI_EXPORT_PEAK_LATENCY, 'ROI Peak Latency'),
                   (ID_ROI_EXPORT_ONSET_LATENCY, 'ROI Onset latency'),
                   (ID_ROI_EXPORT_ONSET_THRESHOLD, 'ROI Onset Threshold'),
                   (ID_ROI_EXPORT_HALFWIDTH, 'ROI Halfwdith'),
                   (ID_EXPORT_ROI_ATTRIBUTES, 'ROI Attributes')]


class FPIMenuBar(wx.MenuBar):
    def __init__(self):
        wx.MenuBar.__init__(self)
        self.setup()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def setup(self):
        # global file_menu
        # global edit_menu
        global options_menu
        global intrincic_menu
        global plot_menu
        global export_menu
        # self.FileMenu(file_menu)
        # self.EditMenu(edit_menu)
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
        mgr.base_dir = path
        print(f'Base dir changed to {mgr.base_dir}')
        parent = parent.GetParent()
        parent.gatherer.scan()
        parent.exp_list.clear()
        parent.exp_list.add_rows(parent.gatherer.to_tuple())
    else:
        ErrorDialog('Could not set requested path')


@register(ID_SET_RAW_DIR)
def SetRawPath(parent):
    path = DataPathDialog(parent, 'Select root trials folder ')
    if path is not None and app_config.check_trials_folder(path):
        # here we should check if the directory contains the proper data structure
        # if not we should offer to create it
        mgr.raw_dir = path
        print(f'Raw dir changed to {mgr.raw_dir}')
    else:
        ErrorDialog('Could not set requested path. Path is invalid or it does not contain Trial folders')


@register(ID_CREATE_FOLDERS)
def CreateFolderStructure(parent):
    with wx.MessageDialog(None, 'Do you really want to create a new folder structure', 'Warning',
                          style=wx.YES_NO | wx.ICON_WARNING) as dlg:
        res = dlg.ShowModal()
        if res == wx.YES:
            app_config.create_folders()


@register(ID_INTRINSIC_ANALYSIS)
def RunIntrinsic(parent):
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


@register(ID_EXPORT_RESPONSE)
def ExportPeakValue(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]
    exp_names = [exp.name for exp in selected]

    response = {exp.name: exp.response for exp in selected}
    save_series(f'aggregated_response', response)
    wx.MessageBox(f'Response values for {exp_names} saved...')


@register(ID_EXPORT_PEAK_VALUES)
def ExportPeakValue(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]
    exp_names = [exp.name for exp in selected]

    peak_value = {exp.name: [exp.max_df] for exp in selected}
    save_series(f'peak_values', peak_value)
    wx.MessageBox(f'Peak values for {exp_names} saved...')


@register(ID_EXPORT_PEAK_LATENCY)
def ExportPeakLatency(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]
    exp_names = [exp.name for exp in selected]

    peak_latency = {exp.name: [exp.peak_latency] for exp in selected}
    save_series('peak_latency', peak_latency)
    wx.MessageBox(f'Peak latency values for {exp_names} saved...')


@register(ID_EXPORT_ONSET_LATENCY)
def ExportOnsetLatency(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    onset_latency = {exp.name: [exp.onset_latency] for exp in selected}
    exp_names = [exp.name for exp in selected]
    save_series('onset_latency', onset_latency)
    wx.MessageBox(f'Onset latency values for {exp_names} saved...')


@register(ID_EXPORT_ONSET_THRESHOLD)
def ExportOnsetThreshold(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    onset_threshold = {exp.name: [exp.onset_threshold] for exp in selected}
    exp_names = [exp.name for exp in selected]
    save_series('onset_threshold', onset_threshold)
    wx.MessageBox(f'Onset threshold values for {exp_names} saved...')


@register(ID_EXPORT_HALFWIDTH)
def ExportHalfwidth(parent):
    root = wx.App.Get().GetRoot()
    exp_list = root.exp_list
    gatherer = root.gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]

    halfwidth = {exp.name: [exp.halfwidth()] for exp in selected}
    exp_names = [exp.name for exp in selected]
    save_series('halfwidth', halfwidth)
    wx.MessageBox(f'Halfwidth values for {exp_names} saved...')


@register(ID_MEAN_RESPONSE)
def MeanResponse(parent):
    exp_list = parent.GetTopLevelParent().exp_list
    gatherer = parent.GetTopLevelParent().gatherer
    selected = [gatherer.get_experiment(exp) for exp in exp_list.current_selection]
    response_stack = np.array([exp.response for exp in selected if exp.response.shape == (81,)])
    res = response_stack.mean(axis=0)

    plt.plot(res)
    plt.title('Mean response')
    plt.xlabel('Frames')
    plt.ylabel('dF/F')
    plt.show()


@register(ID_ROI_EXPORT_RESPONSE)
def ExportRoiResponse(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi=True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    save_series(f'{exp.name}-roi_response', roi_response)


@register(ID_ROI_EXPORT_PEAK_VALUES)
def ExportRoiPeakValues(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    peak_value = parser.max_df(roi=True)
    if peak_value is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    save_series(f'{exp.name}-roi_peak_value', peak_value)


@register(ID_ROI_EXPORT_ONSET_THRESHOLD)
def ExportRoiThreshold(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi=True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    # TODO Fixme
    threshold = fpi.onset_threshold(roi_response)
    save_series(f'{exp.name}-roi_peak_threshold', threshold)


@register(ID_ROI_EXPORT_PEAK_LATENCY)
def ExportROIPeakLatency(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi=True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    # TODO Fixme
    peak_latency = fpi.peak_latency(roi_response)
    save_series(f'{exp.name}-roi_peak_latency', peak_latency)


@register(ID_ROI_EXPORT_ONSET_LATENCY)
def ExportROIOnsetLatency(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    mean_baseline = exp.mean_baseline
    parser = HD5Parser(exp, exp._path)
    roi_response = parser.response(roi=True)
    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    onset_latency = fpi.onset_latency(mean_baseline, roi_response)
    save_series(f'{exp.name}-roi_onset_latency', onset_latency)


@register(ID_ROI_EXPORT_HALFWIDTH)
def ExportROIHalfwidth(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)

    mean_baseline = exp.mean_baseline
    peak_value = parser.max_df(roi=True)
    roi_response = parser.response(roi=True)

    if roi_response is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return
    halfwidth = fpi.halfwidth(roi_response, mean_baseline, peak_value)
    save_series(f'{exp.name}-roi_onset_latency', halfwidth)


@register(ID_ROI_PLOT_RESPONSE)
def PlotROIResponse(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    resp = parser.response(roi=True)
    if resp is None:
        wx.MessageBox('No response for this experiment', 'Plot failed')
        return
    else:
        plt.plot(resp, label='ROI image response')
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('dF/f')
    plt.show()


@register(ID_EXPORT_ROI_ATTRIBUTES)
def ExportROIAttributes(parent):
    exp = parent.GetParent()._experiment
    if not exp.has_roi():
        wx.MessageBox("you need to analyze the experiment before export")
        return
    parser = HD5Parser(exp, exp._path)
    resp = parser.response(roi=True)
    if resp is None:
        dlg = wx.MessageBox('No ROI for this experiment', 'Exception when reading ROI', style=wx.ICON_ERROR)
        return

    response = parser.response(roi=True)
    mean_baseline = exp.mean_baseline
    peak_value = parser.max_df(roi=True)

    threshold = fpi.onset_threshold(exp.mean_baseline)
    onset_latency = fpi.onset_latency(threshold, resp)
    peak_latency = fpi.peak_latency(resp)
    area = exp.roi_area()

    halfwidth = fpi.halfwidth(response, mean_baseline, peak_value)

    attrs = {'Peak value': [peak_value],
             'Mean Baseline': [mean_baseline],
             'Peak latency': [peak_latency],
             'Onset Latency': [onset_latency],
             'Onset Threshold': [threshold],
             'Halfwidth': [halfwidth],
             'Area': [area]
             }
    fname = f'ROI_attrs_{exp.name}'
    try:
        save_series(fname, attrs)
    except Exception as e:
        wx.MessageBox(f'Export failed: {e}', 'Export failed', style=wx.OK | wx.ICON_ERROR)
        print(e)
        return
    wx.MessageBox(f'ROI attributes saved at {fname}.csv', 'Attributes saved', style=wx.OK | wx.ICON_INFORMATION)


@register(ID_NEW_EXPERIMENT_ANALYSIS)
def NewAnalysis(parent):
    with AnalysisPanel(None) as dlg:
        dlg.ShowModal()

@register(ID_TIFF_MULTIANALYSIS)
def Multianalysis(parent):
    def notify():
        wx.MessageBox('Analysis finished', 'Finished')
    path = DataPathDialog(parent, 'Select experiments folder ')
    with wx.BusyInfo("Analyzing TIFF directories...") as info:
        do_analysis(path)
    notify()



def save_series(fname, series):
    if not os.path.exists(mgr.data_export_dir):
        os.mkdir(mgr.data_export_dir)
    fname = mgr.data_export_dir + '/' + fname + '.xlsx'
    df = pd.DataFrame(series)

    print(series)
    '''
    if type(series) == dict:
        with open(fname, 'w') as fd:
            for key, val in series.items():
                fd.write(f'{key},{val}\n')
    elif type(series) == list:
        np.savetxt(fname, np.array(series))
    elif type(series) == np.ndarray:
        np.savetxt(fname, series)
    else:
        wx.MessageBox(f'Export failed. Unexpected series type: {type(series)}, Value: {series}')
        return
        '''
    df.to_excel(fname)
    print('Export complete...')
