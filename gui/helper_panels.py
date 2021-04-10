import wx
from gui.image_roi import ImageControl, PIL2wx
from image_analysis.feature import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fpi import HD5Parser
import cv2

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


        self.rect_size = wx.StaticText(self, label = "Rect Size: ")
        self.size_spin = wx.SpinCtrl(self, value = '', style = wx.SP_ARROW_KEYS, min = 0, max = 100, initial = 15)

        self.Bind(wx.EVT_BUTTON, self.OnTimecourse, self.timecourse)
        self.Bind(wx.EVT_SPINCTRL, self.OnSpin, self.percentage)
        self.Bind(wx.EVT_SPINCTRL, self.OnSizeChange, self.size_spin)
        self.Bind(wx.EVT_BUTTON, self.OnSave, self.to_csv)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.timecourse, 0, wx.ALL, 3)
        sizer.Add(self.percentage, 0, wx.ALL, 3)
        sizer.Add(self.total_pixels, 0, wx.ALL, 3)
        sizer.Add(self.rect_size, 0, wx.ALL, 3)
        sizer.Add(self.size_spin, 0, wx.ALL, 3)

        sizer.Add(self.to_csv, 0, wx.ALL, 3)

        self.SetSizer(sizer)

    def OnSizeChange(self, event):
        self.OnSpin(event)
        event.Skip()

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
        rect_size = self.size_spin.GetValue()
        try:
            s = float(percent)
        except Exception:
            wx.MessageBox('Not a valid percentage. It should be a value between 0 and 100', 'Invalid percentage')
            return
        threshold = float(percent)
        mask = create_mask(self.exp.resp_map, threshold)
        # response, *rest = masked_timeseries(self.exp.stack, mask)
        response = self.compute_timeseries(int(rect_size))
        self.reponse = response
        plt.plot(response, label = 'Mean Response Per Frame')

        # plt.plot(np.trim_zeros(timecourse[:, 2]), label = 'Std Deviation')
        plt.legend()
        plt.grid()
        plt.show()


    def OnSpin(self, event):
        '''

        Parameters
        ----------
        event
        func: a callable use to return

        Returns
        -------

        '''
        percent = self.percentage.GetValue()
        rect_size = int(self.size_spin.GetValue())
        try:
            s = float(percent)
        except Exception:
            wx.MessageBox('Not a valid percentage. It should be a value between 0 and 100', 'Invalid percentage')
            return
        # contours is a list of arrays, points? array([[[214, 32]]]) dtype = int32
        contoured_image, mean_area, max_area, min_area, area_count, mask = find_contours(self.exp.resp_map, float(percent), rect_size)
        self.Fit()
        parent = self.GetParent()
        parent.status_bar.SetStatusText(f'Mean Area: {mean_area:.2f} - Max Area: {max_area:.2f} - Min Area: {min_area:.2f} - Area Count: {area_count}')
        new_image = ImageControl.PIL_image_from_array(contoured_image, normalize = False)
        parent.set_image(new_image)

    def compute_timeseries(self, rect_size):
        '''
        Compute the df/F using the small red rect area that is the centroid of the contout(island)
        It
        :return:
        '''
        roi_frames = compute_rect_timeseries(self.exp.stack, rect_size)
        timeseries = roi_frames.mean(axis = (0, 1)) # compute the mean of every roi_frame
        return timeseries

    def OnReset(self, event):
        parent = self.GetParent()
        parent.reset_image()


class ROIOperationPanel(OperationPanel):
    def __init__(self, parent, exp):
        super().__init__(parent, exp)
        self.parser = HD5Parser(exp, exp._path)
        self.response = self.parser.response(roi = True)

    def OnTimecourse(self, event):
        self.OnSpin(event)
        percent = self.percentage.GetValue()
        try:
            s = float(percent)
        except Exception:
            wx.MessageBox('Not a valid percentage. It should be a value between 0 and 100', 'Invalid percentage')
            return
        threshold = float(percent)
        mask = create_mask(self.parser.resp_map(roi = True), threshold)
        try:
            response, *rest = masked_timeseries(self.GetParent().parser.stack(roi = True), mask)
        except Exception as e:
            wx.MessageBox('Could not plot timecourse. Saved roi in datastore and currently selected roi on window do not match. Maybe you changed the ROI and need to repeat analysis?', 'Plot failed')
            return
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
        # contours is a list of arrays, points? array([[[214, 32]]]) dtype = int32
        contoured_image, mean_area, max_area, min_area, area_count, mask = find_contours(self.parser.resp_map(roi = True), float(percent))
        self.Fit()
        parent = self.GetParent()
        parent.status_bar.SetStatusText(
            f'Mean Area: {mean_area:.2f} - Max Area: {max_area:.2f} - Min Area: {min_area:.2f} - Area Count: {area_count}')
        new_image = ImageControl.PIL_image_from_array(contoured_image, normalize=False)
        parent.set_image(new_image)


    def OnSave(self,event):
        if self.response is not None:
            with wx.DirDialog(None, 'Choose save folder') as dlg:
                dlg.ShowModal()
                percentage = self.percentage.GetValue()
                save_path = dlg.GetPath()
                file_name = f'{save_path}/ROI-Response-{percentage}%-{self.exp.name}.csv'
                np.savetxt(file_name, self.response)
                wx.MessageBox(f'Data saved to {file_name}', 'Save succesful', wx.OK | wx.ICON_INFORMATION)
