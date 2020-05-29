import wx
from pubsub import pub
from pub_messages import ANALYSIS_UPDATE


class AnalysisProgress(wx.Dialog):
    def __init__(self, parent, *args, count = 0, range = 100, **kwargs):
        wx.Dialog.__init__(self, parent, title = 'Analysis progress')
        self.count = count
        self.range = range
        self.progress = wx.Gauge(self, range = 100)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.progress, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

        pub.subscribe(self.OnUpdate, ANALYSIS_UPDATE)
    def OnUpdate(self, val):
        self.count += val
        if self.count >= self.range:
            self.Destroy()
        self.progress.SetValue(self.count)
