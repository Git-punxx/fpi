import wx
from light_analyzer import ThreadedIntrinsic
from app_config import config_manager as mgr

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
        with wx.DirDialog(None, 'Choose Root Folder', mgr.base_dir) as dlg:
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
        self.path= wx.StaticText(self, label = 'Folder: ')
        self.folder_input = wx.TextCtrl(self, value = '', size = (250, 25))
        self.browse = wx.Button(self, label = 'Select Folder')
        self.from_lbl = wx.StaticText(self, label = 'From: ')
        self.to_lbl = wx.StaticText(self, label = 'To: ')

        self.from_input = wx.TextCtrl(self, value = '0', size = (40, 25))
        self.to_input = wx.TextCtrl(self, value = '-1', size = (40, 25))

        self.strategy_txt = wx.StaticText(self, label = 'Strategy on corrupted photo: ')
        self.strategy = wx.Choice(self, choices = ['Skip', 'Duplicate', 'Average'])

        self.analyze = wx.Button(self, label = 'Analyze')

        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        folder_sizer.Add(self.path, 0)
        folder_sizer.Add(self.folder_input, 0)
        folder_sizer.Add(self.browse, 0)


        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        range_sizer.Add(self.from_lbl, 0)
        range_sizer.Add(self.from_input, 0)
        range_sizer.Add(self.to_lbl, 0)
        range_sizer.Add(self.to_input, 0)

        strategy_sizer = wx.BoxSizer(wx.HORIZONTAL)
        strategy_sizer.Add(self.strategy_txt, 0)
        strategy_sizer.Add(self.strategy, 0)

        footer_sizer = wx.BoxSizer(wx.VERTICAL)
        footer_sizer.Add(self.analyze, 0)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(range_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(strategy_sizer, 0, wx.EXPAND | wx.ALL, 5)
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
            ThreadedIntrinsic(path, val_from, val_to, strategy)
        except Exception as e:
            print(f'Exception: {e}')

    def OnBrowse(self, event):
        with wx.DirDialog(None, 'Select trail folder', '', wx.DIRP_DIR_MUST_EXIST) as dlg:
            dlg.ShowModal()
            path = dlg.GetPath()
            self.folder_input.SetValue(path)

if __name__ == '__main__':
    app = wx.PySimpleApp()
    dlg = Preferences(None)
    res = dlg.ShowModal()


