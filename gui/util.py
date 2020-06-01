import PIL
import numpy as np
import wx


class BoxPlotChoices(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self._category_choices = [' ', 'AnimalLines', 'Treatment', 'Stimulus']

        self.choices = wx.Choice(self, choices = self._category_choices)

        sizer = wx.BoxSizer(wx.VERTICAL)

    def GetSelection(self):
        choice = self.choices.GetString(self.choices.GetSelection())
        return choice
