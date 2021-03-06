import wx
import concurrent.futures
import h5py
import os
import datetime
import modified_intrinsic.imaging as intr
from fpi import HDF5Writer
import numpy as np
import gui.image_roi
from gui.custom_events import *
from gui.animator import animate, export_frames
from gui.helper_panels import *
from image_analysis import util
from gui.menus import FPIImageMenu


class DetailsPanel(wx.Frame):
    def __init__(self, parent, experiment, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, style = wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT, **kwargs)

        self.menubar = FPIImageMenu()
        self.SetMenuBar(self.menubar)

        self._experiment = experiment
        self._path = self._experiment._path
        self.SetTitle(f'Experiment {self._experiment.name}')

        self.status_bar = wx.StatusBar(self)
        self.status_bar.SetFieldsCount(1)
        self.status_bar.SetStatusText(f'Details for {self._experiment.name}')


        self.details_panel= wx.Panel(self, style = wx.BORDER_RAISED)
        self.roi_panel = FixedROIPanel(parent = self, exp=self._experiment)
        self.operation_panel = OperationPanel(self, exp = self._experiment)

        self.image_panel = self.build_image_panel()

        self._file_lbl = wx.StaticText(self.details_panel, label = 'Filename', style = wx.ALIGN_RIGHT)
        self._file_txt = wx.StaticText(self.details_panel, label = self._experiment.name)

        self._data_created_lbl = wx.StaticText(self.details_panel, label = 'Date Created')
        self._data_created_txt = wx.StaticText(self.details_panel, label = f'{datetime.datetime.fromtimestamp(os.stat(self._path).st_mtime).strftime("%D %M %Y")}')

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

        self._image_lbl  = wx.StaticText(self.details_panel, label = 'Image Shape')
        self._image_txt  = wx.StaticText(self.details_panel, label = f'{self.image_panel.image_size}')

        self._max_df_lbl = wx.StaticText(self.details_panel, label = 'Max DF')
        self._max_df_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.max_df:5.8f}')


        self._mean_baseline_lbl = wx.StaticText(self.details_panel, label = 'Baseline Mean')
        self._mean_baseline_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.mean_baseline:5.8f}')

        self._halfwidth_lbl = wx.StaticText(self.details_panel, label = 'Halfwitdh Mean')
        self._halfwidth_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.halfwidth()[0]} - {self._experiment.halfwidth()[1]:5.8f}')

        self._roi_lbl = wx.StaticText(self.details_panel, label = 'Roi Range')
        self._roi_txt = wx.StaticText(self.details_panel, label = f'{self._experiment.roi_range}')

        self._roi_analysis_btn = wx.Button(self.details_panel, label = 'Analyze Range of Interest')
        self._delete_roi = wx.Button(self.details_panel, label = 'Delete Range of interest')
        self._animate_button = wx.Button(self.details_panel, label = 'Animate')
        self._export_button = wx.Button(self.details_panel, label = 'Export Frames')


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

        sizer.Add(self._image_lbl, (8, 0))
        sizer.Add(self._image_txt, (8, 1))


        sizer.Add(self._max_df_lbl, (9, 0))
        sizer.Add(self._max_df_txt, (9, 1))

        sizer.Add(self._mean_baseline_lbl, (10, 0))
        sizer.Add(self._mean_baseline_txt, (10, 1))

        sizer.Add(self._no_baseline_lbl, (11, 0))
        sizer.Add(self._no_baseline_txt, (11, 1))

        sizer.Add(self._roi_lbl, (12, 0))
        sizer.Add(self._roi_txt, (12, 1))

        sizer.Add(self._halfwidth_lbl, (13, 0))
        sizer.Add(self._halfwidth_txt, (13, 1))

        sizer.Add(self._roi_analysis_btn, (15, 0), flag = wx.EXPAND)
        sizer.Add(self._delete_roi, (16, 0), flag = wx.EXPAND)
        sizer.Add(self._animate_button, (17, 0), flag = wx.EXPAND)
        sizer.Add(self._export_button, (18, 0), flag = wx.EXPAND)

        self.details_panel.SetSizer((sizer))
        # Load and place the image
        self.im_sizer = wx.BoxSizer(wx.VERTICAL)
        self.im_sizer.Add(self.image_panel, 1, wx.EXPAND)

        footer_sizer = wx.BoxSizer(wx.VERTICAL)
        footer_sizer.Add(self.status_bar, 1, wx.EXPAND)

        self.im_sizer.Add(footer_sizer, 0, wx.EXPAND)

        operation_sizer = wx.BoxSizer(wx.VERTICAL)

        operation_sizer.Add(self.roi_panel, 0, wx.EXPAND | wx.ALL, 5)
        operation_sizer.Add(self.operation_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.im_sizer.Add(operation_sizer, 0, wx.EXPAND)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(self.details_panel, 0, wx.EXPAND | wx.ALL, 2)
        main_sizer.Add(self.im_sizer, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(main_sizer)
        self.Fit()

        # Check if the analysis button should be enabled
        if self._experiment.roi_range is None:
            self._roi_analysis_btn.Disable()
            self._delete_roi.Disable()

        self.Bind(wx.EVT_BUTTON, self.OnAnalysis, self._roi_analysis_btn)
        self.Bind(wx.EVT_BUTTON, self.OnDeleteROI, self._delete_roi)
        self.Bind(wx.EVT_BUTTON, self.OnAnimate, self._animate_button)
        self.Bind(wx.EVT_BUTTON, self.OnExport, self._export_button)

        self.Bind(wx.EVT_CLOSE, self.OnClose, self)

        self.Bind(wx.EVT_MENU, self.OnMenu)
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

    def OnMenu(self, event):
        evt_id = event.GetId()

    def ShowModal(self):
        self.GetParent().Disable()
        self.Show()
        self.SetFocus()

    def OnClose(self, event):
        self.GetParent().Enable()
        self.GetParent().SetFocus()
        self.Destroy()

    def _datastore_structure(self):
        with h5py.File(self._path, 'r') as datastore:
            for key, items in datastore['df'].items():
                pass

    def update_stats(self):
        self._max_df_txt.SetLabel(f'{self._experiment.max_df}')

    def set_image(self, image: Image):
        self.image_panel.set_image(image)

    def reset_image(self):
        self.image_panel.reset_image()

    def build_image_panel(self):
        df = self._experiment.resp_map
        im = util.wx_fromarray(df)
        image_panel = gui.image_roi.ImageControl(self, image = im)
        return image_panel

    def _analyze(self, roi = None):
        with wx.BusyInfo('Performing analysis on ROI...'):
            if self._experiment.roi_range is not None:
                slice = self._experiment.roi_slice()
                self.status_bar.SetStatusText(f'Analyzing for slice {slice}')
                x_slice, y_slice = slice
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.status_bar.SetStatusText(f'Analyzing norm_stack')
                norm_stack_future = executor.submit(intr.normalize_stack, self._experiment.stack[x_slice, y_slice])
                resp_map_future = executor.submit(intr.find_resp, self._experiment.stack)
                norm_stack = norm_stack_future.result()
                resp = resp_map_future.result()
            # norm_stack = intr.normalize_stack(self._experiment.stack)
            # resp = intr.find_resp(self._experiment.stack)
            self.status_bar.SetStatusText(f'Analyzing resp_stack')
            im_resp, df = intr.resp_map(norm_stack)
            self.status_bar.SetStatusText(f'Calculating df, avg_df, max_df and area ')
            resp_map = im_resp
            avg_df = df.mean(0)
            max_df = df.max(1).mean()
            area = np.sum(im_resp > 0)

        print('Saving ROI analysis')
        data_dict = {'stack': self._experiment.stack,
                     'norm_stack': norm_stack,
                     'response': resp,
                     'resp_map': resp_map,
                     'df': df,
                     'avg_df': avg_df,
                     'max_df': max_df,
                     'area':area}
        self._save_analysis(data_dict)

    def OnAnalysis(self, event):
        self.status_bar.SetStatusText('Beginning analysis')
        self._analyze()
        self.status_bar.SetStatusText('Finished analysis')


    def OnAnimate(self, event):
        self.status_bar.SetStatusText(f'Beginning animation for {self._experiment.roi_range}')
        animate(self._experiment)
        self.status_bar.SetStatusText(f'Finished animation for {self._experiment.roi_range}')

    def OnExport(self, event):
        # Choose dialog to choose frames
        dlg =  wx.TextEntryDialog(self, 'Enter no of frames', 'Choose frames to export')
        dlg.ShowModal()
        res = dlg.GetValue()
        frame_list = [int(arg) for arg in res.split()]
        dlg.Destroy()
        export_frames(self._experiment, frame_list)

    def OnDeleteROI(self, event):
        with wx.MessageDialog(None, 'Are you sure you want to delete this ROI?', 'Deleting ROI', style = wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING) as dlg:
            resp = dlg.ShowModal()
            if resp != wx.ID_YES:
                return
            else:
                writer = HDF5Writer(self._experiment._path)
                writer.delete_roi()
                self._experiment._roi = None
                self._roi_analysis_btn.Disable()
                self._delete_roi.Disable()

    def OnRoiUpdate(self, event):
        # Delete the previous roi
        # Update the h5 file with the new data
        # maybe use threads here
        print(f'Updating ROI')

        writer = HDF5Writer(self._experiment._path)
        print(event.roi)
        writer.write_roi(event.roi)
        self._experiment._roi = event.roi
        self._roi_analysis_btn.Enable()
        self._delete_roi.Enable()
        self._roi_txt.SetLabel(f'{self._experiment.roi_range}')

    def _save_analysis(self, analysis_dict):
        writer = HDF5Writer(self._experiment._path)
        writer.insert_into_group(analysis_dict)


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
        pass

