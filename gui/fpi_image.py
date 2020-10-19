import wx
import h5py
from PIL import Image
import numpy as np
import os
import datetime


class DetailsPanel(wx.Dialog):
    def __init__(self, parent, experiment, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        self._experiment = experiment
        self._path = self._experiment._path
        self.details_panels= wx.Panel(self)

        self._file_lbl = wx.StaticText(self, label = 'Filename', style = wx.ALIGN_RIGHT)
        self._file_txt = wx.StaticText(self, label = self._experiment.name)

        self._data_created_lbl = wx.StaticText(self, label = 'Date Created')
        self._data_created_txt = wx.StaticText(self, label = f'{datetime.datetime.fromtimestamp(os.stat(self._path).st_mtime).strftime("%H:%M:%S - %D %M %Y")}')

        self._file_size_txt = wx.StaticText(self, label = f'{os.stat(self._path).st_size}')
        self._file_size_lbl = wx.StaticText(self, label = 'File size')

        self._line_lbl = wx.StaticText(self, label = 'Animal Line')
        self._line_txt = wx.StaticText(self, label = self._experiment.animalline)

        self._stim_lbl = wx.StaticText(self, label = 'Stimulus')
        self._stim_txt = wx.StaticText(self, label = self._experiment.stimulation)

        self._treatment_lbl = wx.StaticText(self, label = 'Treatment')
        self._treatment_txt = wx.StaticText(self, label = self._experiment.treatment)

        self._genotype_lbl = wx.StaticText(self, label = 'Genotype')
        self._genotype_txt = wx.StaticText(self, label = self._experiment.genotype)

        self._no_trials_lbl = wx.StaticText(self, label = '# trials')
        self._no_trials_txt = wx.StaticText(self, label = f'{self._experiment.no_trials}')

        self._no_baseline_lbl = wx.StaticText(self, label = '# Baseline')
        self._no_baseline_txt = wx.StaticText(self, label = f'{self._experiment.no_baseline}')

        self._area_lbl  = wx.StaticText(self, label = 'Response area')
        self._area_txt  = wx.StaticText(self, label = f'{self._experiment.response_area}')

        self._max_df_lbl = wx.StaticText(self, label = 'Max DF')
        self._max_df_txt = wx.StaticText(self, label = f'{self._experiment.max_df}')


        self._mean_baseline_lbl = wx.StaticText(self, label = 'Baseline Mean')
        self._mean_baseline_txt = wx.StaticText(self, label = f'{self._experiment.mean_baseline}')


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

        sizer.Add(self._area_lbl, (8, 0))
        sizer.Add(self._area_txt, (8, 1))


        sizer.Add(self._max_df_lbl, (9, 0))
        sizer.Add(self._max_df_txt, (9, 1))

        sizer.Add(self._mean_baseline_lbl, (10, 0))
        sizer.Add(self._mean_baseline_txt, (10, 1))

        sizer.Add(self._no_baseline_lbl, (11, 0))
        sizer.Add(self._no_baseline_txt, (11, 1))

        # Load and place the image
        image_panel = self.load_image()
        im_sizer = wx.BoxSizer(wx.VERTICAL)
        im_sizer.Add(image_panel, 1, wx.EXPAND)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 2)
        main_sizer.Add(im_sizer, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(main_sizer)
        self.Fit()

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
        im = self._experiment.anat
        image_panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        im_max = im.max()
        im_min = im.min()
        divider = im_max - im_min
        if divider == 0:
            divider = 1
        im = 255*(im - im_min)/divider

        raw_image = Image.fromarray(im)
        image = wx.Image(*raw_image.size)
        image.SetData(raw_image.convert('RGB').tobytes())
        bitmap_image = wx.StaticBitmap(image_panel, -1, wx.BitmapFromImage(image))

        sizer.Add(bitmap_image, 1, wx.EXPAND | wx.ALL, 2)

        image_panel.SetSizer(sizer)
        image_panel.Fit()
        return image_panel



