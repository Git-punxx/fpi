import wx
from tqdm import tqdm

from light_analyzer import ThreadedIntrinsic, RawAnalysisController
from app_config import config_manager as mgr
import traceback
import os
from concurrent.futures import ThreadPoolExecutor


BUTTON_WIDTH = 20
BUTTON_HEIGHT = 20
MAX_IMAGES = 80
MAX_TRIALS = 50

def DataPathDialog(parent, msg):
    with wx.DirDialog(parent, msg) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            return path

def ErrorDialog(msg):
    result = wx.MessageBox(msg, 'Error', wx.ICON_ERROR | wx.OK)




class Preferences(wx.Dialog):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, -1, 'Preferences')

        apply_button = wx.Button(self, label = 'Apply')
        cancel_button = wx.Button(self, label = 'Cancel')

        base_dir_lbl = wx.StaticText(self, label = 'Root folder', style = wx.ALIGN_LEFT)
        self.base_dir_path = wx.TextCtrl(self, value = mgr.base_dir, style = wx.TE_READONLY)
        base_dir_browser = wx.Button(self, label = '...')

        stage_1_lbl = wx.StaticText(self, label = 'Stage 1 Experiment Color', style = wx.ALIGN_LEFT)
        self.stage_1_ctrl = wx.TextCtrl(self, value = mgr._json['stage_color']['1'], style = wx.TE_READONLY)
        stage_1_color_browser = wx.Button(self, label = '...')

        stage_2_lbl = wx.StaticText(self, label = 'Stage 2 Experiment Color', style = wx.ALIGN_LEFT)
        self.stage_2_ctrl = wx.TextCtrl(self, value = mgr._json['stage_color']['2'], style = wx.TE_READONLY)
        stage_2_color_browser = wx.Button(self, label = '...')

        stage_3_lbl = wx.StaticText(self, label = 'Stage 3 Experiment Color', style = wx.ALIGN_LEFT)
        self.stage_3_ctrl = wx.TextCtrl(self, value = mgr._json['stage_color']['3'], style = wx.TE_READONLY)
        stage_3_color_browser = wx.Button(self, label = '...')


        # Events
        self.Bind(wx.EVT_BUTTON, self.OnBrowse, base_dir_browser)
        self.Bind(wx.EVT_BUTTON, self.OnStage1, stage_1_color_browser)
        self.Bind(wx.EVT_BUTTON, self.OnStage2, stage_2_color_browser)
        self.Bind(wx.EVT_BUTTON, self.OnStage3, stage_3_color_browser)

        self.Bind(wx.EVT_BUTTON, self.OnApply, apply_button)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_button)



        # Styling
        self.grid = wx.GridSizer(rows = 4, cols = 3, hgap = 2, vgap = 2)

        self.grid.Add(base_dir_lbl, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.grid.Add(self.base_dir_path, 0, wx.ALIGN_LEFT)
        self.grid.Add(base_dir_browser, 0, wx.ALIGN_LEFT)

        self.grid.Add(stage_1_lbl, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.grid.Add(self.stage_1_ctrl, 0, wx.ALIGN_LEFT)
        self.grid.Add(stage_1_color_browser, 0, wx.ALIGN_LEFT)

        self.grid.Add(stage_2_lbl, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.grid.Add(self.stage_2_ctrl, 0, wx.ALIGN_LEFT)
        self.grid.Add(stage_2_color_browser, 0, wx.ALIGN_LEFT)


        self.grid.Add(stage_3_lbl, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.grid.Add(self.stage_3_ctrl, 0, wx.ALIGN_LEFT)
        self.grid.Add(stage_3_color_browser, 0, wx.ALIGN_LEFT)


        self.footer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.footer_sizer.Add(apply_button, 0, wx.ALIGN_LEFT)
        self.footer_sizer.Add(cancel_button, 0)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.footer_sizer, 0, wx.EXPAND)
        self.SetSizer(self.main_sizer)

    def OnBrowse(self, event):
        with wx.DirDialog(None, 'Choose Root Folder', mgr.raw_dir) as dlg:
            dlg.ShowModal()
            path = dlg.GetPath()
            self.base_dir_path.SetValue(path)
            dlg.Destroy()

    def OnApply(self, event):
        root_path = self.base_dir_path.GetValue()
        mgr.base_dir = root_path
        color_1 = self.stage_1_ctrl.GetValue()
        color_2 = self.stage_2_ctrl.GetValue()
        color_3 = self.stage_3_ctrl.GetValue()
        mgr._json['stage_color']['1'] = color_1
        mgr._json['stage_color']['2'] = color_2
        mgr._json['stage_color']['3'] = color_3
        self.Destroy()

    def OnCancel(self, event):
        self.Destroy()

    def OnStage1(self, event):
        with wx.ColourDialog(None) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                data = dlg.GetColourData().GetColour().Get()
                self.stage_1_ctrl.SetBackgroundColour(wx.Colour(data))
                self.stage_1_ctrl.SetValue(str(data))


    def OnStage2(self, event):
        with wx.ColourDialog(None) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                data = dlg.GetColourData().GetColour().Get()
                self.stage_2_ctrl.SetValue(str(data))

    def OnStage3(self, event):
        with wx.ColourDialog(None) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                data = dlg.GetColourData().GetColour().Get()
                self.stage_3_ctrl.SetValue(str(data))


class AnalysisPanel(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.controller = RawAnalysisController(mgr.raw_dir)
        self.IMG_TYPES = ['PNG', 'TIFF']
        self.choices = []
        self.SetSize(700, 400)
        self.SetTitle('Raw Data Analysis')
        self.path= wx.StaticText(self, label = 'Folder: ', style = wx.ALIGN_CENTER_VERTICAL)
        self.folder_input = wx.ComboBox(self, choices = self.choices, size = (500, 25))
        self.browse = wx.Button(self, label = 'Select Folder', style = wx.ALIGN_CENTER_VERTICAL)
        self.from_lbl = wx.StaticText(self, label = 'From Trial: ', size = (140, 25), style = wx.ALIGN_CENTER_VERTICAL)
        self.to_lbl = wx.StaticText(self, label = 'To Trial: ', style = wx.ALIGN_CENTER_VERTICAL)

        self.from_input = wx.TextCtrl(self, value = '0', size = (40, 25))
        self.to_input = wx.TextCtrl(self, value = '-1', size = (40, 25))

        self.binning_txt = wx.StaticText(self, label = 'Binning: ', size = (140, 25), style = wx.ALIGN_CENTER_VERTICAL)
        self.binning_input = wx.ComboBox(self, choices = '1 2 3'.split())
        self.binning_input.SetSelection(2)

        self.baseline_text = wx.StaticText(self, label = 'No Baseline Frames: ', size = (140, 24), style = wx.ALIGN_CENTER_VERTICAL)
        self.baseline_input = wx.TextCtrl(self, value = '30', size = (40, 25))

        self.imgtype_text = wx.StaticText(self, label = 'Image Type: ', size = (140, 25), style = wx.ALIGN_CENTER_VERTICAL)
        self.imgtype_input = wx.ComboBox(self, choices = self.IMG_TYPES, size = (80, 25))
        self.imgtype_input.SetSelection(0)

        self.analyze = wx.Button(self, label = 'Analyze')

        ########### here we should do TrialMatric and use a controller
        print('Trial folders analyzed. Loading. Please wait...')

        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        folder_sizer.Add(self.path, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        folder_sizer.Add(self.folder_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        folder_sizer.Add(self.browse, 0, flag = wx.ALIGN_CENTER_VERTICAL)


        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        range_sizer.Add(self.from_lbl, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.from_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.to_lbl, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.to_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)

        binning_sizer = wx.BoxSizer(wx.HORIZONTAL)
        binning_sizer.Add(self.binning_txt, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        binning_sizer.Add(self.binning_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)

        baseline_sizer = wx.BoxSizer(wx.HORIZONTAL)
        baseline_sizer.Add(self.baseline_text, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        baseline_sizer.Add(self.baseline_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)

        img_type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        img_type_sizer.Add(self.imgtype_text, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        img_type_sizer.Add(self.imgtype_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)


        #center_sizer = wx.BoxSizer(wx.VERTICAL)
        #center_sizer.Add(trial_panel, 0, wx.EXPAND)

        footer_sizer = wx.BoxSizer(wx.VERTICAL)
        footer_sizer.Add(self.analyze, 0)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(range_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(binning_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(baseline_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(img_type_sizer, 0, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(footer_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)


        self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.browse)
        self.Bind(wx.EVT_BUTTON, self.OnAnalyze, self.analyze)


    def OnAnalyze(self, event):
        self.Disable()
        try:
            val_from = int(self.from_input.GetValue())
            val_to = int(self.to_input.GetValue())
            path = self.folder_input.GetValue()
            binning = int(self.binning_input.GetValue())
            baseline = int(self.baseline_input.GetValue())
            img_type = self.imgtype_input.GetValue()
            if img_type == 'PNG':
                pattern = '*.png'
            else:
                pattern = '*.tif'
            with wx.BusyInfo("Analyzing images") as info:
                analysis = ThreadedIntrinsic(path, binning=binning, pattern = pattern, n_baseline=baseline, start = val_from, end = val_to)
                analysis.complete_analysis()
        except Exception as e:
            print(f'Exception: {e}')
            print(traceback.format_exc())

        self.Enable()


    def OnBrowse(self, event):
        with wx.DirDialog(None, 'Select trail folder', '', wx.DIRP_DIR_MUST_EXIST) as dlg:
            dlg.ShowModal()
            path = dlg.GetPath()
            self.choices.append(path)
            self.folder_input.Set(self.choices)
            self.folder_input.SetSelection(-1)
            self.folder_input.SetValue(path)
            self.SetTitle(os.path.basename(path))



class TrialMatrix(wx.Panel):
    def __init__(self, parent, trial_tree: dict, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.trial_tree = trial_tree
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.setup()
        self.SetSizer(self.sizer)

    @property
    def no_trials(self):
        return sum([1 for i in self.trial_tree.keys()])

    @property
    def no_images(self):
        if self.no_trials == 0:
            return 0
        else:
            return len(self.trial_tree['Trial_0'])

    def setup(self):
        print('Setting up trial matric')
        for trial, experiment_list in self.trial_tree.items():
            print(f'Trial {trial}')
            print(experiment_list)
            self.sizer.Add(TrialPanel(self, trial, experiment_list))




class TrialPanel(wx.Panel):
    def __init__(self, parent, name, image_paths: list, *args, **kwargs):
        super().__init__(parent, *args, style = wx.BORDER_SUNKEN, **kwargs)
        self.image_paths = image_paths
        self.name = name
        self.SetSize(BUTTON_HEIGHT + 5, BUTTON_WIDTH * MAX_IMAGES)
        self.name_txt = wx.StaticText(self, style = wx.BORDER_SUNKEN, label = self.name)
        self.grid = wx.GridBagSizer(2, 2)
        self.grid.Add(self.name_txt, pos = (0, 0), flag= wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 1)
        self.SetSizer(self.grid)
        self.setup()

    def _load_image(self, path):
        name = os.path.basename(path)
        image = wx.Image(path, wx.BITMAP_TYPE_PNG)
        image.SetOption('name', name)
        image.Rescale(BUTTON_WIDTH, BUTTON_HEIGHT)
        return image

    def _load_images(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._load_image, path) for path in self.image_paths]
        image_list = [future.result() for future in futures]
        print('Loading images finsideh. Constructing gui')
        return image_list #sorted

    def setup(self):
        self.SetBackgroundColour('red')
        images = self._load_images()
        for index, image in enumerate(images, 1):
            # make a thumbnail and set it as image to button
            bmp = wx.Bitmap(image)
            img = ImageButton(self, bmp, name = f'img_{index}.png', style = wx.BORDER_SUNKEN)
            self.grid.Add(img, pos = (0, index))



class ImageButton(wx.BitmapButton):
    def __init__(self, parent, image, *args, **kwargs):
        super().__init__(parent, bitmap = image, *args, **kwargs)
        self.parent = parent
        self.image = image

        self.SetToolTip(f'{parent.name}: {self.GetName()}')
        self.Bind(wx.EVT_BUTTON, self.OnClick)

    def OnClick(self, event):
        dlg = wx.MessageBox('Success', 'Ok we did it')


if __name__ == '__main__':
    app = wx.PySimpleApp()
    dlg = AnalysisPanel(None)
    res = dlg.ShowModal()


