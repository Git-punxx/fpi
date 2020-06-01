import PIL
import numpy as np
import wx
from app_config import config_manager


class BoxPlotChoices(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self._category_choices = list(config_manager.categories.keys()) + ['']

        self.choices = wx.Choice(self, choices = self._category_choices)

        sizer = wx.BoxSizer(wx.VERTICAL)

    def GetSelection(self):
        choice = self.choices.GetString(self.choices.GetSelection())
        return choice
