import wx
from gui.image_roi import ImageControl, PIL2wx
from image_analysis.feature import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class FixedROIPanel(wx.Panel):
    def __init__(self, parent, exp, *args, **kwargs, ):
        super().__init__(parent = parent, *args, **kwargs)
        self.exp = exp.resp_map

class OperationPanel(wx.Panel):
    def __init__(self, parent, exp, *args, **kwargs):
        super().__init__(parent = parent, *args, **kwargs)
        self.exp = exp
        self.response = self.exp.response

        self.timecourse = wx.Button(self, label = 'Plot timecourse')
        self.to_csv = wx.Button(self, label = 'Save to csv')

        self.percentage = wx.SpinCtrl(self, value = '', style = wx.SP_ARROW_KEYS, min = 0, max = 100, initial = 0)
        self.total_pixels = wx.StaticText(self, label = f'Total Pixels: 0')

        self.Bind(wx.EVT_BUTTON, self.OnTimecourse, self.timecourse)
        self.Bind(wx.EVT_SPINCTRL, self.OnSpin, self.percentage)
        self.Bind(wx.EVT_BUTTON, self.OnSave, self.to_csv)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.timecourse, 0, wx.ALL, 3)
        sizer.Add(self.percentage, 0, wx.ALL, 3)
        sizer.Add(self.total_pixels, 0, wx.ALL, 3)
        sizer.Add(self.to_csv, 0, wx.ALL, 3)

        self.SetSizer(sizer)

    def OnSave(self, event):
        if self.response is not None:
            with wx.DirDialog(None, 'Choose save folder') as dlg:
                dlg.ShowModal()
                percentage = self.percentage.GetValue()
                save_path = dlg.GetPath()
                file_name = f'{save_path}/Response-{percentage}%-{self.exp.name}.csv'
                np.savetxt(file_name, self.response)
                wx.MessageBox(f'Data saved to {file_name}', 'Save succesful', wx.OK | wx.ICON_INFORMATION)

    def OnTimecourse(self, event):
        self.OnSpin(event)
        percent = self.percentage.GetValue()
        try:
            s = float(percent)
        except Exception:
            wx.MessageBox('Not a valid percentage. It should be a value between 0 and 100', 'Invalid percentage')
            return
        threshold = float(percent)
        mask, total_pixels = create_mask(self.exp.resp_map, threshold)
        response = masked_response(self.exp.stack, mask)
        self.reponse = response
        plt.plot(response, label = 'Mean Response Per Frame')

        # plt.plot(np.trim_zeros(timecourse[:, 2]), label = 'Std Deviation')
        plt.legend()
        plt.grid()
        plt.show()


    def OnSpin(self, event):
        percent = self.percentage.GetValue()
        try:
            s = float(percent)
        except Exception:
            wx.MessageBox('Not a valid percentage. It should be a value between 0 and 100', 'Invalid percentage')
            return
        processed, total_pixels = process_image(self.exp.resp_map, float(percent))
        self.total_pixels.SetLabel(f'Total Pixels: {str(total_pixels)}')
        self.Fit()
        parent = self.GetParent()
        new_image = ImageControl.PIL_image_from_array(processed)
        parent.set_image(new_image)

    def OnReset(self, event):
        parent = self.GetParent()
        parent.reset_image()
