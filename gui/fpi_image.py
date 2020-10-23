import wx
import threading
import h5py
import os
import datetime
import intrinsic.imaging as intr
from fpi import HDF5Writer
import numpy as np
import gui.image_roi
from gui.custom_events import *


class DetailsPanel(wx.Dialog):
    def __init__(self, parent, experiment, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        self._experiment = experiment
        self._path = self._experiment._path
        self.details_panel= wx.Panel(self, style = wx.BORDER_RAISED)

        self._file_lbl = wx.StaticText(self.details_panel, label = 'Filename', style = wx.ALIGN_RIGHT)
        self._file_txt = wx.StaticText(self.details_panel, label = self._experiment.name)

        self._data_created_lbl = wx.StaticText(self.details_panel, label = 'Date Created')
        self._data_created_txt = wx.StaticText(self.details_panel, label = f'{datetime.datetime.fromtimestamp(os.stat(self._path).st_mtime).strftime("%H:%M:%S - %D %M %Y")}')

        self._file_size_txt = wx.StaticText(self.details_panel, label = f'{os.stat(self._path).st_size}')
        self._file_size_lbl = wx.StaticText(self.details_panel, label = 'File size')

        self._line_lbl = wx.StaticText(self.details_panel, label = 'Animal Line')
        self._line_txt = wx.StaticText(self.details_panel, label = self._experiment.animalline)

        self._stim_lbl = wx.StaticText(self.details_panel, label = 'Stimulus')
        self._stim_txt = wx.StaticText(self.details_panel, label = self._experiment.stimulation)

        self._treatment_lbl = wx.StaticText(self.details_panel, label = 'Treatment')
        self._treatment_txt = wx.StaticText(self.details_panel, label = self._experiment.treatment)

        self._genotype_lbl = wx.StaticText(self.details_panel, label = 'Genotype')
        self._genotype_txt = wx.StaticText(self.details_panel, label = self._experiment.genotype)

        self._no_trials_lbl = wx.StaticText(self.details_panel, label = '# trials')
        self._no_trials_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.no_trials}')

        self._no_baseline_lbl = wx.StaticText(self.details_panel, label = '# Baseline')
        self._no_baseline_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.no_baseline}')

        #self._area_lbl  = wx.StaticText(self.details_panel, label = 'Response area')
        #self._area_txt  = wx.StaticText(self.details_panel, label = f'{self._experiment.response_area}')

        self._max_df_lbl = wx.StaticText(self.details_panel, label = 'Max DF')
        self._max_df_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.max_df}')


        self._mean_baseline_lbl = wx.StaticText(self.details_panel, label = 'Baseline Mean')
        self._mean_baseline_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.mean_baseline}')

        self._roi_lbl = wx.StaticText(self.details_panel, label = 'Roi Range')
        self._roi_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.roi_range}')

        self._roi_analysis_btn = wx.Button(self.details_panel, label = 'Analyze Range of Interest')
        self._delete_roi = wx.Button(self.details_panel, label = 'Delete Range of interest')

        sizer = wx.GridBagSizer(hgap = 5, vgap = 5)
        sizer.Add(self._file_lbl, (0, 0))
        sizer.Add(self._file_txt, (0, 1))

        sizer.Add(self._data_created_lbl, (1, 0))
        sizer.Add(self._data_created_txt, (1, 1))

        sizer.Add(self._file_size_lbl, (2, 0))
        sizer.Add(self._file_size_txt, (2, 1))

        sizer.Add(self._line_lbl, (3, 0))
        sizer.Add(self._line_txt, (3, 1))

        sizer.Add(self._stim_txt, (4, 1))
        sizer.Add(self._stim_lbl, (4, 0))

        sizer.Add(self._treatment_lbl, (5, 0))
        sizer.Add(self._treatment_txt, (5, 1))

        sizer.Add(self._genotype_lbl, (6, 0))
        sizer.Add(self._genotype_txt, (6, 1))

        sizer.Add(self._no_trials_lbl, (7, 0))
        sizer.Add(self._no_trials_txt, (7, 1))

        #sizer.Add(self._area_lbl, (8, 0))
        #sizer.Add(self._area_txt, (8, 1))


        sizer.Add(self._max_df_lbl, (9, 0))
        sizer.Add(self._max_df_txt, (9, 1))

        sizer.Add(self._mean_baseline_lbl, (10, 0))
        sizer.Add(self._mean_baseline_txt, (10, 1))

        sizer.Add(self._no_baseline_lbl, (11, 0))
        sizer.Add(self._no_baseline_txt, (11, 1))

        sizer.Add(self._roi_lbl, (12, 0))
        sizer.Add(self._roi_txt, (12, 1))

        sizer.Add(self._roi_analysis_btn, (14, 0), flag = wx.EXPAND)
        sizer.Add(self._delete_roi, (15, 0), flag = wx.EXPAND)

        self.details_panel.SetSizer((sizer))
        # Load and place the image
        self.image_panel = self.load_image()
        im_sizer = wx.BoxSizer(wx.VERTICAL)
        im_sizer.Add(self.image_panel, 1, wx.EXPAND)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(self.details_panel, 0, wx.EXPAND | wx.ALL, 2)
        main_sizer.Add(im_sizer, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(main_sizer)
        self.Fit()

        # Check if the analysis button should be enabled
        if self._experiment.roi_range is None:
            self._roi_analysis_btn.Disable()
            self._delete_roi.Disable()

        self.Bind(wx.EVT_BUTTON, self.OnAnalysis, self._roi_analysis_btn)
        self.Bind(wx.EVT_BUTTON, self.OnDeleteROI, self._delete_roi)

        self.Bind(EVT_ROI_UPDATE, self.OnRoiUpdate)
            # self._response = None
        # self._timecourse = None
        #
        # self._no_trials = None
        # self._no_baseline = None
        # self._response_area = None
        # self._max_df = None
        # self._avg_df = None
        # self._mean_baseline = None
        # self._peak_latency = None

    def _datastore_structure(self):
        with h5py.File(self._path, 'r') as datastore:
            for key, items in datastore['df'].items():
                pass

    def load_image(self):
        im = self._experiment.resp_map
        image_panel = gui.image_roi.ImageControl.fromarray(self, im)
        return image_panel

    def _analyze(self, roi = None):
        with wx.BusyInfo('Performing analysis on ROI...'):
            norm_stack = intr.normalize_stack(self._experiment.stack)
            resp = intr.find_resp(self._experiment.stack)

            print('Threads finished...')
            resp_map, df = intr.resp_map(norm_stack)

            print('Analysis finished')
        data_dict = {'norm_stack': norm_stack, 'resp_map': resp_map, 'resp': resp,  'df': df, 'avg_df': df.mean(0), 'max_df': df.max(1).mean() , 'area': np.sum(resp_map > 0)}
        self._save_analysis(data_dict)

    def OnAnalysis(self, event):
        self._analyze()

    def OnDeleteROI(self, event):
        with wx.MessageDialog(None, 'Are you sure you want to delete this ROI?', 'Deleting ROI', style = wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING) as dlg:
            resp = dlg.ShowModal()
            if resp != wx.ID_YES:
                return
            else:
                writer = HDF5Writer(self._experiment._path)
                writer.delete_roi()
                self._roi_analysis_btn.Disable()
                self._delete_roi.Disable()

    def OnRoiUpdate(self, event):
        # Delete the previous roi
        # Update the h5 file with the new data
        # maybe use threads here
        print(event.roi)
        writer = HDF5Writer(self._experiment._path)
        writer.write_roi(event.roi)
        self._roi_analysis_btn.Enable()
        self._delete_roi.Enable()

    def _save_analysis(self, analysis_dict):
        writer = HDF5Writer(self._experiment._path)
        writer.insert_into_group('roi', analysis_dict)


class ROIPanel(wx.Panel):
    def __init__(self, exp_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_name = exp_name

        # Panel for scalar values of experiment
        self.roi_details = wx.StaticBox(self, wx.NewId(), 'ROI Details')

        # Panel for images
        self.roi_images = wx.Panel(self)

        # Panel for the average frame stack
        self.stack_frames = wx.Panel(self)

    def construct_images(self):
        """

        :return:
        """

