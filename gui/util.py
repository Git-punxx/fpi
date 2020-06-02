import wx
from app_config import config_manager


class BoxPlotChoices(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self._category_choices = list(config_manager.categories.keys())[:-1]

        self.choices = wx.Choice(self, choices = self._category_choices)
        self.choices.SetSelection(0)

        sizer = wx.BoxSizer(wx.VERTICAL)

    def GetSelection(self):
        choice = self.choices.GetString(self.choices.GetSelection())
        return choice

