import os
import wx
import wx.lib.agw.aui as aui
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from fpi import TimecourseAnalyzer, ResponseAnalyzer


class MainFrame(wx.Frame):
    def __init__(self, parent, id=wx.ID_ANY, title='FPIAnalyzer'):
        super(MainFrame, self).__init__(parent, id, title)
        self._fpi = None
        self.plotter = PlotNotebook(self)

        self.response_btn = wx.Button(self, 'Analyze response')
        self.timecourse_btn = wx.Button(self, 'Analyze timecourse')
        self.latency_button = wx.Button(self, 'Response Latency')

        self.CreateStatusBar()

        self.Bind(wx.EVT_BUTTON, self.OnTimecourse, self.timecourse_btn)
        self.Bind(wx.EVT_BUTTON, self.OnResponse, self.response_btn)
        self.Bind(wx.EVT_BUTTON, self.OnLatency, self.latency_button)

        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        plot_sizer.Add(self.plotter, 1, wx.EXPAND)
        plot_sizer.Add(self.response_btn)
        plot_sizer.Add(self.timecourse_btn)
        plot_sizer.Add(self.latency_button)

        self.SetSizer(plot_sizer)


    def OnTimecourse(self, event):
        with wx.FileDialog(self, 'Open File', style=wx.FD_OPEN) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    fpi = TimecourseAnalyzer(path)
                    self.plotter.add(fpi)
                except Exception as e:
                    print(e)
                    with wx.MessageDialog(self, 'Something is wrong with the provided file', 'File error',
                                          wx.OK | wx.ICON_ERROR) as dlg:
                        dlg.ShowModal()
                    return
                self.SetStatusText(os.path.basename(path))

    def OnResponse(self, event):
        with wx.FileDialog(self, 'Open File', style=wx.FD_OPEN) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    fpi = ResponseAnalyzer(path)
                    self.plotter.add(fpi)
                except Exception as e:
                    print(e)
                    with wx.MessageDialog(self, 'Something is wrong with the provided file', 'File error',
                                          wx.OK | wx.ICON_ERROR) as dlg:
                        dlg.ShowModal()
                    return
                self.SetStatusText(os.path.basename(path))

    def OnLatency(self, event):
        with wx.FileDialog(self, 'Open File', style=wx.FD_OPEN) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    fpi = TimecourseAnalyzer(path)
                    self.plotter.add(fpi)
                except Exception as e:
                    print(e)
                    with wx.MessageDialog(self, 'Something is wrong with the provided file', 'File error',
                                          wx.OK | wx.ICON_ERROR) as dlg:
                        dlg.ShowModal()
                    return
                self.SetStatusText(os.path.basename(path))


class Plot(wx.Panel):
    def __init__(self, parent, id=wx.ID_ANY, dpi=None, fpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id)
        if fpi is not None:
            self.fpi = fpi
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(5, 8))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()


        # Bindings
        self.Bind(wx.EVT_BUTTON, self.OnPlot)
        self.canvas.mpl_connect('button_press_event', self.OnClick)

        # Layouts
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        right_sizer.Add(self.canvas, 1, wx.EXPAND)
        right_sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        sizer.Add(left_sizer, 0, wx.EXPAND)
        sizer.Add(right_sizer, 1, wx.EXPAND)
        self.SetSizer(sizer)



    def OnPlot(self, event):
        if self.fpi is None:
            with wx.MessageDialog(self, 'Please open an fpi file before plotting', 'File error',
                                  wx.OK | wx.ICON_ERROR) as dlg:
                dlg.ShowModal()
            return
        else:
            ax = self.figure.gca()
            self.fpi.plot(ax, choices)
            self.canvas.draw()

    def OnClick(self, event):
        pass


class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        super(wx.Panel, self).__init__(parent, id)
        self.nb = aui.AuiNotebook(self)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self, abf):
        page = Plot(self.nb, abf=abf)
        self.nb.AddPage(page, abf.protocol)
        return page.figure


class FPI(wx.App):
    def OnInit(self):
        frame = MainFrame(None, -1, 'Plotter')
        frame.Show()
