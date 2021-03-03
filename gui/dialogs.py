import wx
from PIL import Image
from light_analyzer import ThreadedIntrinsic, RawAnalysisController
from app_config import config_manager as mgr
import traceback
import traceback


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
    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = RawAnalysisController(mgr.raw_dir)
        self.choices = []
        self.SetSize(MAX_IMAGES * BUTTON_WIDTH, MAX_TRIALS * BUTTON_WIDTH)
        self.path= wx.StaticText(self, label = 'Folder: ', style = wx.ALIGN_CENTER_VERTICAL)
        self.folder_input = wx.ComboBox(self, choices = self.choices, size = (500, 25))
        self.browse = wx.Button(self, label = 'Select Folder', style = wx.ALIGN_CENTER_VERTICAL)
        self.from_lbl = wx.StaticText(self, label = 'From Trial: ', style = wx.ALIGN_CENTER_VERTICAL)
        self.to_lbl = wx.StaticText(self, label = 'To Trial: ', style = wx.ALIGN_CENTER_VERTICAL)

        self.from_input = wx.TextCtrl(self, value = '0', size = (40, 25))
        self.to_input = wx.TextCtrl(self, value = '-1', size = (40, 25))

        self.strategy_txt = wx.StaticText(self, label = 'Strategy on corrupted photo: ', style = wx.ALIGN_CENTER_VERTICAL)
        self.strategy = wx.ComboBox(self, choices = ['Skip', 'Duplicate', 'Average'])

        self.analyze = wx.Button(self, label = 'Analyze')

        ########### here we should do TrialMatric and use a controller
        trial_tree = self.controller.trial_tree
        trial_panel = TrialMatrix(self, trial_tree)
        wx.MessageBox('Not a valid trial folder', 'Error on experiments folder')

        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        folder_sizer.Add(self.path, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        folder_sizer.Add(self.folder_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        folder_sizer.Add(self.browse, 0, flag = wx.ALIGN_CENTER_VERTICAL)


        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        range_sizer.Add(self.from_lbl, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.from_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.to_lbl, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(self.to_input, 0, flag = wx.ALIGN_CENTER_VERTICAL)

        strategy_sizer = wx.BoxSizer(wx.HORIZONTAL)
        strategy_sizer.Add(self.strategy_txt, 0, flag = wx.ALIGN_CENTER_VERTICAL)
        strategy_sizer.Add(self.strategy, 0, flag = wx.ALIGN_CENTER_VERTICAL)

        center_sizer = wx.BoxSizer(wx.VERTICAL)
        center_sizer.Add(trial_panel, 0, wx.EXPAND)

        footer_sizer = wx.BoxSizer(wx.VERTICAL)
        footer_sizer.Add(self.analyze, 0)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(range_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(strategy_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(center_sizer, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(footer_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)


        self.Bind(wx.EVT_BUTTON, self.OnBrowse, self.browse)
        self.Bind(wx.EVT_BUTTON, self.OnAnalyze, self.analyze)


    def OnAnalyze(self, event):
        try:
            val_from = int(self.from_input.GetValue())
            val_to = int(self.to_input.GetValue())
            path = self.folder_input.GetValue()
            strategy = self.strategy.GetCurrentSelection()
            with wx.BusyInfo("Analyzing images...") as info:
                analysis = ThreadedIntrinsic(path, start = val_from, end = val_to)
                analysis.complete_analysis()
        except Exception as e:
            print(f'Exception: {e}')
            print(traceback.format_exc())


    def OnBrowse(self, event):
        with wx.DirDialog(None, 'Select trail folder', '', wx.DIRP_DIR_MUST_EXIST) as dlg:
            dlg.ShowModal()
            path = dlg.GetPath()
            self.choices.append(path)
            self.folder_input.Set(self.choices)
            self.folder_input.SetSelection(-1)
            self.folder_input.SetValue(path)
            self.controller.set_root(path)



class TrialMatrix(wx.Panel):
    def __init__(self, parent, trial_tree: dict, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.trial_tree = trial_tree
        self.trial_length = len(self.trial_tree.keys())
        self.no_images = len(list(self.trial_tree.values())[0])
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.setup()
        self.SetSizer(self.sizer)

    def setup(self):
        for trial, experiment_list in self.trial_tree.items():
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


    def setup(self):
        self.SetBackgroundColour('red')
        for index, path in enumerate(self.image_paths, 1):
            print(f'Installing {path}')
            image = wx.Image(path, wx.BITMAP_TYPE_PNG)
            # make a thumbnail and set it as image to button
            image.Rescale(BUTTON_WIDTH, BUTTON_HEIGHT)
            bmp = wx.Bitmap(image)
            img = ImageButton(self, bmp, name = f'img_{index}.png', style = wx.BORDER_SUNKEN)
            self.grid.Add(img, pos = (0, index))
        self.Fit()



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


