import wx
import time
import wx.lib.agw.aui as aui
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from fpi import *
from pubsub import pub
from gui.menus import *
from gui.dialogs import *
from fpi_plotter import FPIPlotter
from gui.fpi_image import DetailsPanel
from gui.popups import PopupMenuMixin
from gui.util import BoxPlotChoices

CHOICES_CHANGED = 'choices.changed'
LINE_CHANGED = 'line.changed'
STIMULUS_CHANGED = 'stimulus.changed'
TREATMENT_CHANGED = 'treatment.changed'
GENOTYPE_CHANGED = 'genotype.changed'
EXPERIMENT_CHANGED = 'experiment.changed'
CLEAR_FILTERS = 'clear.filters'
EXPERIMENT_LIST_CHANGED = 'experiments.list.changed'


class MainFrame(wx.Frame):
    def __init__(self, parent, id=wx.ID_ANY, title='FPIAnalyzer'):
        super(MainFrame, self).__init__(parent, id, title)
        self.Maximize(True)

        # Menubar
        self.menubar = FPIMenuBar()
        self.SetMenuBar(self.menubar)

        with wx.BusyInfo('FPIPlotter initializing...'):
            self.setup()
        self.exp_list = FPIExperimentList(self)
        self.exp_list.add_columns(app_config.categories)
        self.exp_list.add_rows(self.gatherer.to_tuple())

        self.filter = FilterPanel(self)
        self.plotter = PlotNotebook(self)

        self.boxplot_choices = BoxPlotChoices(self)


        self.response_btn = wx.Button(self, label='Plot Response')
        self.peak_value_btn = wx.Button(self, label='Plot Peak Value')
        self.latency_button = wx.Button(self, label='Plot Onset Latency')
        self.peak_button = wx.Button(self, label='Plot Peak Latency')
        self.anat_button = wx.Button(self, label='Plot Anat')
        self.area_button = wx.Button(self, label='Plot Area')


        # Bindings
        self.Bind(wx.EVT_BUTTON, self.OnPeakValue, self.peak_value_btn)
        self.Bind(wx.EVT_BUTTON, self.OnResponse, self.response_btn)
        self.Bind(wx.EVT_BUTTON, self.OnResponseLatency, self.latency_button)
        self.Bind(wx.EVT_BUTTON, self.OnPeakLatency, self.peak_button)
        self.Bind(wx.EVT_BUTTON, self.OnAnat, self.anat_button)
        self.Bind(wx.EVT_BUTTON, self.OnArea, self.area_button)

        self.Bind(wx.EVT_MENU, self.OnMenu)

        # Layout
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header_sizer.Add(self.filter, 0, wx.EXPAND | wx.ALL, 2)

        plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        plot_sizer.Add(self.exp_list, 0, wx.EXPAND)
        plot_sizer.Add(self.plotter, 1, wx.EXPAND)

        footer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        footer_sizer.Add(self.boxplot_choices, 0)
        footer_sizer.Add(self.response_btn, 0)
        footer_sizer.Add(self.peak_value_btn, 0)
        footer_sizer.Add(self.latency_button)
        footer_sizer.Add(self.peak_button)
        footer_sizer.Add(self.anat_button)
        footer_sizer.Add(self.area_button)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(header_sizer, 0, wx.EXPAND)
        main_sizer.Add(plot_sizer, 1, wx.EXPAND)
        main_sizer.Add(footer_sizer, 0, wx.EXPAND)
        self.SetSizer(main_sizer)
        self.Fit()

        # Publishing
        pub.subscribe(self.OnLineChange, LINE_CHANGED)
        pub.subscribe(self.OnGenChange, GENOTYPE_CHANGED)
        pub.subscribe(self.OnTreatChange, TREATMENT_CHANGED)
        pub.subscribe(self.OnStimChange, STIMULUS_CHANGED)
        pub.subscribe(self.OnClear, CLEAR_FILTERS)
        pub.subscribe(self.OnChoicesChanged, CHOICES_CHANGED)

    def setup(self):
        if not os.path.exists(app_config.base_dir):
            path = SetDataPath(self)
        # here we should pop a
        try:
            self.gatherer = ExperimentManager(app_config.base_dir)
        except Exception as e:
            print(e)
            with wx.MessageDialog(self, 'Something is wrong with the FPI configuration file', 'Configuration error',
                                  wx.OK | wx.ICON_ERROR) as dlg:
                dlg.ShowModal()
                exit(1)

    def OnMenu(self, event):
        evt_id = event.GetId()


    def OnLineChange(self, args):
        res = self.gatherer.filterLine(args)
        self.exp_list.update(res)

    def OnGenChange(self, args):
        res = self.gatherer.filterGenotype(args)
        self.exp_list.update(res)

    def OnStimChange(self, args):
        res = self.gatherer.filterStimulus(args)
        self.exp_list.update(res)

    def OnTreatChange(self, args):
        res = self.gatherer.filterTreatment(args)
        self.exp_list.update(res)

    def OnChoicesChanged(self, selections):
        res = self.gatherer.filterAll(selections)
        self.exp_list.update(res)


    def OnBaseline(self, event):
        with wx.BusyInfo('Plotting baseline...'):
            # Get the selected items from the list ctrl
            choice = self.boxplot_choices.GetSelection()
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            # Return the experiments that correspond to the selected items
            exp = self.gatherer.filterSelected(selected)
            # Create a new tab to our notebook to hold the plots and plot them using the FPIPlotter
            self.plotter.add(exp, 'Baseline', choice)

    def OnResponse(self, event):
        with wx.BusyInfo('Plotting response'):
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            self.plotter.add(exp, 'Response')

    def OnResponseLatency(self, event):
        with wx.BusyInfo('Plotting OnSet latency'):
            choice = self.boxplot_choices.GetSelection()
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            if len(exp) <= 1:
                ErrorDialog("You must select more than one experiment for boxplots")
                return
            self.plotter.add(exp, 'Onset_latency', choice)

    def OnPeakLatency(self, event):
        with wx.BusyInfo('Plotting peak latency'):
            choice = self.boxplot_choices.GetSelection()
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            if len(exp) <= 1:
                ErrorDialog("You must select more than one experiment for boxplots")
                return
            self.plotter.add(exp, 'Peak_Latency', choice)

    def OnPeakValue(self, event):
        with wx.BusyInfo('Plotting peak value'):
            choice = self.boxplot_choices.GetSelection()
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            if len(exp) <= 1:
                ErrorDialog("You must select more than one experiment for boxplots")
                return
            self.plotter.add(exp, 'Peak_Value', choice)

    def OnAnat(self, event):
        with wx.BusyInfo('Plotting anat image'):
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            self.plotter.add(exp, 'anat')


    def OnArea(self, event):
        with wx.BusyInfo('Plotting area'):
            choice = self.boxplot_choices.GetSelection()
            selected = self.exp_list.GetSelection()
            if not selected:
                return
            exp = self.gatherer.filterSelected(selected)
            if len(exp) <= 1:
                ErrorDialog("You must select more than one experiment for boxplots")
                return
            self.plotter.add(exp, 'area', choice)

    def OnClear(self, args=None):
        res = self.gatherer.clear_filters()
        self.exp_list.update(res)


class FilterPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.choices = []

        an_line_lbl = wx.StaticText(self, label='Mouseline')
        stim_lbl = wx.StaticText(self, label='Stimulus')
        treat_lbl = wx.StaticText(self, label='Treatment')
        gen_lbl = wx.StaticText(self, label='Genotype')

        self.animal_line_choice = wx.Choice(self, choices=app_config.animal_lines + [''])
        self.animal_line_choice.SetSelection(-1)

        self.stim_choice = wx.Choice(self, choices=app_config.stimulations + [''])
        self.stim_choice.SetSelection(-1)

        self.treat_choice = wx.Choice(self, choices=app_config.treatments + [''])
        self.treat_choice.SetSelection(-1)

        self.gen_choice = wx.Choice(self, choices=app_config.genotypes + [''])
        self.gen_choice.SetSelection(-1)

        self.choices = [self.animal_line_choice, self.stim_choice, self.treat_choice, self.gen_choice]

        self.clear_btn = wx.Button(self, label='Clear')


        # Bindings
        self.animal_line_choice.Bind(wx.EVT_CHOICE, self.OnLineChoice)
        self.stim_choice.Bind(wx.EVT_CHOICE, self.OnStimChoice)
        self.gen_choice.Bind(wx.EVT_CHOICE, self.OnGenChoice)
        self.treat_choice.Bind(wx.EVT_CHOICE, self.OnTreatChoice)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.OnClear)


        # Layout
        sizer = wx.GridBagSizer(vgap=5, hgap=5)
        sizer.Add(an_line_lbl, (0, 0))
        sizer.Add(self.animal_line_choice, (1, 0))

        sizer.Add(stim_lbl, (0, 1))
        sizer.Add(self.stim_choice, (1, 1))

        sizer.Add(treat_lbl, (0, 2))
        sizer.Add(self.treat_choice, (1, 2))

        sizer.Add(gen_lbl, (0, 3))
        sizer.Add(self.gen_choice, (1, 3))

        sizer.Add(self.clear_btn, (1, 4))

        self.SetSizer(sizer)
        self.Fit()

    def SetLineChoice(self, items):
        self.animal_line_choice.SetItems(items)

    def SetGenotypeChoice(self, items):
        self.gen_choice.SetItems(items)

    def SetStimulusChoices(self, items):
        self.stim_choice.SetItems(items)

    def OnLineChoice(self, choices):
        selection = self.animal_line_choice.GetStringSelection()
        all = self.GetChoices()
        #pub.sendMessage(LINE_CHANGED, args=selection)
        pub.sendMessage(CHOICES_CHANGED, selections=all)

    def OnStimChoice(self, event):
        all = self.GetChoices()
        selection = self.stim_choice.GetStringSelection()
        #pub.sendMessage(STIMULUS_CHANGED, args=selection)
        pub.sendMessage(CHOICES_CHANGED, selections=all)

    def OnTreatChoice(self, event):
        all = self.GetChoices()
        selection = self.treat_choice.GetStringSelection()
        #pub.sendMessage(TREATMENT_CHANGED, args=selection)
        pub.sendMessage(CHOICES_CHANGED, selections=all)

    def OnGenChoice(self, event):
        all = self.GetChoices()
        selection = self.gen_choice.GetStringSelection()
        #pub.sendMessage(GENOTYPE_CHANGED, args=selection)
        pub.sendMessage(CHOICES_CHANGED, selections=all)

    def OnClear(self, event):
        self.animal_line_choice.SetSelection(-1)
        self.treat_choice.SetSelection(-1)
        self.stim_choice.SetSelection(-1)
        self.gen_choice.SetSelection(-1)

        pub.sendMessage(CLEAR_FILTERS, args=None)

    def GetChoices(self):
        return [choice.GetStringSelection() for choice in self.choices]



class FPIExperimentList(wx.Panel, PopupMenuMixin):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        PopupMenuMixin.__init__(self)
        self.list = wx.ListCtrl(self, -1, style=wx.LC_REPORT)

        self.current_selection = []

        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelect)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnDeselect)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnActivate)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.list, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

        pub.subscribe(self.update, EXPERIMENT_LIST_CHANGED)

    def OnActivate(self, event):
        item = self.list.GetItem(event.GetIndex())
        exp_name = item.GetText()

        path = self.GetTopLevelParent().gatherer.get_experiment(exp_name)
        with DetailsPanel(parent = None, name = path) as exp_dialog:
            exp_dialog.ShowModal()

    def OnSelect(self, event):
        item = self.list.GetItem(event.GetIndex())
        text = item.GetText()
        self.current_selection.append(text)

    def OnDeselect(self, event):
        item = self.list.GetItem(event.GetIndex())
        text = item.GetText()
        self.current_selection.remove(text)

    def clear(self):
        self.list.DeleteAllItems()

    def update(self, choices):
        print('Received message')
        self.clear()
        self.add_rows(choices)

    def add_columns(self, columns):
        self.list.InsertColumn(0, 'Experiment')
        for index, col in enumerate(columns, 1):
            self.list.InsertColumn(index, col)

    def add_rows(self, data):
        for row in data:
            self.list.Append(row)

    def GetSelection(self):
        return self.current_selection

    def CreateContextMenu(self, menu):
        menu.Append(wx.ID_COPY)
        menu.Append(wx.ID_CUT)
        menu.Append(wx.ID_PASTE)

class Plot(wx.Panel):
    def __init__(self, parent, id=wx.ID_ANY, dpi = 100, experiment=None, **kwargs):
        wx.Panel.__init__(self, parent, id)
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(10, 5))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        # Bindings
        self.Bind(wx.EVT_BUTTON, self.OnPlot)
        self.canvas.mpl_connect('button_press_event', self.OnClick)

        # Layouts
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    def OnPlot(self, event):
        """
        Pass the canvas to a possibly empyt list of experiments to plot themselves
        :param event:
        :return:
        """
        gatherer = self.GetTopLevelParent().gatherer
        experiment_list = gatherer.to_tuple()
        if experiment_list is None:
            with wx.MessageDialog(self, 'Please select an fpi experiment before plotting', 'No experiment(s) provided',
                                  wx.OK | wx.ICON_ERROR) as dlg:
                dlg.ShowModal()
            return
        else:
            ax = self.figure.gca()
            [gatherer.get_experiment(exp).plot(ax) for exp in experiment_list]
            self.canvas.draw()

    def plot(self, plot_type = None, experiment_list = None, choice = None):
        gatherer = self.GetTopLevelParent().gatherer
        experiment_data = [gatherer.get_experiment(exp.name) for exp in experiment_list]
        plotter = FPIPlotter(self.figure, experiment_data)
        print(f'Choices for plot: {choice}')
        plotter.plot(plot_type, choice)

        self.canvas.draw()

    def OnClick(self, event):
        pass


class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        wx.Panel.__init__(self, parent, id)
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self, exp, title, choice = None):
        page = Plot(self.nb, experiment=exp)
        self.nb.AddPage(page, caption=title)
        page.plot(title.lower(), exp, choice)
        self.nb.AdvanceSelection(True)
        return page


class FPI(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None, -1, 'FPI Plotter')
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == '__main__':
    app = FPI()
    app.MainLoop()
