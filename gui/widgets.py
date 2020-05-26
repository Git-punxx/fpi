import wx
import wx.lib.agw.aui as aui
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from fpi import *
from pubsub import pub
from gui.menus import *
from gui.dialogs import *

LINE_CHANGED = 'line.changed'
STIMULUS_CHANGED = 'stimulus.changed'
TREATMENT_CHANGED = 'treatment.changed'
GENOTYPE_CHANGED = 'genotype.changed'
EXPERIMENT_CHANGED = 'experiment.changed'
CLEAR_FILTERS = 'clear.filters'


class MainFrame(wx.Frame):
    def __init__(self, parent, id=wx.ID_ANY, title='FPIAnalyzer'):
        super(MainFrame, self).__init__(parent, id, title)
        self.Maximize(True)

        # Menubar
        self.menubar = FPIMenuBar()
        self.SetMenuBar(self.menubar)

        self.CreateStatusBar()

        self.setup()
        self.exp_list = FPIExperimentList(self)
        self.exp_list.add_columns(app_config.categories)
        self.exp_list.add_rows(self.gatherer.experiment_list())

        self.filter = FilterPanel(self)
        self.plotter = PlotNotebook(self)

        self.response_btn = wx.Button(self, label='Plot response')
        self.timecourse_btn = wx.Button(self, label='Plot timecourse')
        self.latency_button = wx.Button(self, label='Plot Response Latency')


        # Bindings
        self.Bind(wx.EVT_BUTTON, self.OnTimecourse, self.timecourse_btn)
        self.Bind(wx.EVT_BUTTON, self.OnResponse, self.response_btn)
        self.Bind(wx.EVT_BUTTON, self.OnLatency, self.latency_button)
        self.Bind(wx.EVT_MENU, self.OnMenu)

        # Layout
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header_sizer.Add(self.filter, 0, wx.EXPAND | wx.ALL, 2)

        plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        plot_sizer.Add(self.exp_list, 0, wx.EXPAND)
        plot_sizer.Add(self.plotter, 1, wx.EXPAND)

        footer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        footer_sizer.Add(self.response_btn, 0)
        footer_sizer.Add(self.timecourse_btn, 0)
        footer_sizer.Add(self.latency_button)

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

    def setup(self):
        if not os.path.exists(app_config.base_dir):
            SetDataPath(self)

        self.gatherer = FPIGatherer()
        # here we should pop a
        try:
            self.gatherer.gather()
        except Exception as e:
            with wx.MessageDialog(self, 'Something is wrong with the FPI configuration file', 'Configuration error',
                                  wx.OK | wx.ICON_ERROR) as dlg:
                dlg.ShowModal()
                exit(1)

    def OnMenu(self, event):
        evt_id = event.GetId()
        print(evt_id)


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

    def OnTimecourse(self, event):
        selected = self.exp_list.GetSelection()
        exp = self.gatherer.filter_selected(selected)
        canvas = self.plotter.add(exp, 'Timecourse')

    def OnResponse(self, event):
        selected = self.exp_list.GetSelection()
        exp = self.gatherer.filter_selected(selected)
        self.plotter.add(exp, 'Response')

    def OnLatency(self, event):
        selected = self.exp_list.GetSelection()
        exp = self.gatherer.filter_selected(selected)
        self.plotter.add(exp, 'Latency')

    def OnClear(self, args=None):
        self.gatherer.clear()
        res = self.gatherer.experiment_list()
        self.exp_list.update(res)


class FilterPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        an_line_lbl = wx.StaticText(self, label='Mouseline')
        stim_lbl = wx.StaticText(self, label='Stimulus')
        treat_lbl = wx.StaticText(self, label='Treatment')
        gen_lbl = wx.StaticText(self, label='Genotype')

        self.an_line_choice = wx.Choice(self, choices=app_config.animal_lines)
        self.stim_choice = wx.Choice(self, choices=app_config.stimulations)
        self.treat_choice = wx.Choice(self, choices=app_config.treatments)
        self.gen_choice = wx.Choice(self, choices=app_config.genotypes)

        self.clear_btn = wx.Button(self, label='Clear')

        # Bindings
        self.an_line_choice.Bind(wx.EVT_CHOICE, self.OnLineChoice)
        self.stim_choice.Bind(wx.EVT_CHOICE, self.OnStimChoice)
        self.gen_choice.Bind(wx.EVT_CHOICE, self.OnGenChoice)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.OnClear)

        # Layout
        sizer = wx.GridBagSizer(vgap=5, hgap=5)
        sizer.Add(an_line_lbl, (0, 0))
        sizer.Add(self.an_line_choice, (1, 0))

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
        self.an_line_choice.SetItems(items)

    def SetGenotypeChoice(self, items):
        self.gen_choice.SetItems(items)

    def SetStimulusChoices(self, items):
        self.stim_choice.SetItems(items)

    def OnLineChoice(self, choices):
        selection = self.an_line_choice.GetStringSelection()
        pub.sendMessage(LINE_CHANGED, args=selection)

    def OnStimChoice(self, event):
        selection = self.stim_choice.GetStringSelection()
        pub.sendMessage(STIMULUS_CHANGED, args=selection)

    def OnTreatChoice(self, event):
        selection = self.treat_choice.GetStringSelection()
        pub.sendMessage(TREATMENT_CHANGED, args=selection)

    def OnGenChoice(self, event):
        selection = self.gen_choice.GetStringSelection()
        pub.sendMessage(GENOTYPE_CHANGED, args=selection)

    def OnClear(self, event):
        pub.sendMessage(CLEAR_FILTERS, args=None)


class FPIExperimentList(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.list = wx.ListCtrl(self, -1, style=wx.LC_REPORT)

        self.current_selection = []

        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelect, self.list)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnDeselect, self.list)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.list, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

    def clear(self):
        self.list.DeleteAllItems()

    def update(self, data):
        self.clear()
        self.add_rows(data)

    def add_columns(self, columns):
        self.list.InsertColumn(0, 'Experiment')
        for index, col in enumerate(columns, 1):
            self.list.InsertColumn(index, col)

    def add_rows(self, data):
        for row in data:
            self.list.Append(row)

    def OnSelect(self, event):
        self.current_selection.append(event.GetText())

    def OnDeselect(self, event):
        self.current_selection.remove(event.GetText())

    def GetSelection(self):
        return self.current_selection


class Plot(wx.Panel):
    def __init__(self, parent, id=wx.ID_ANY, dpi=None, fpi_list=None, **kwargs):
        wx.Panel.__init__(self, parent, id)
        self.fpi_list = fpi_list
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(5, 8))
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
        if self.fpi_list is None or self.fpi_list.empty():
            with wx.MessageDialog(self, 'Please select an fpi experiment before plotting', 'No experiment(s) provided',
                                  wx.OK | wx.ICON_ERROR) as dlg:
                dlg.ShowModal()
            return
        else:
            ax = self.figure.gca()
            [exp.plot(ax) for exp in self.fpi_list]
            self.canvas.draw()

    def plot(self, plot_type=None):
        ax = self.figure.gca()
        ax.grid(True, color = 'grey', linewidth = 0.5)
        print('Plotting ', self.fpi_list)
        [exp.plot(ax, plot_type) for exp in self.fpi_list]
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

    def add(self, fpi_list, title):
        page = Plot(self.nb, fpi_list=fpi_list)
        self.nb.AddPage(page, caption=title)
        page.plot(title.lower())
        return page


class FPI(wx.App):
    def OnInit(self):
        frame = MainFrame(None, -1, 'FPI Plotter')
        frame.Show()
        return True


if __name__ == '__main__':
    app = FPI()
    app.MainLoop()
