import wx
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



        # Events
        self.Bind(wx.EVT_BUTTON, self.OnBrowse, base_dir_browser)

        self.Bind(wx.EVT_BUTTON, self.OnApply, apply_button)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_button)



        # Styling
        self.grid = wx.GridSizer(rows = 3, cols = 3, hgap = 2, vgap = 2)

        self.grid.Add(base_dir_lbl, 0, wx.EXPAND | wx.ALIGN_LEFT)
        self.grid.Add(self.base_dir_path, 0, wx.ALIGN_LEFT)
        self.grid.Add(base_dir_browser, 0, wx.ALIGN_LEFT)

        self.footer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.footer_sizer.Add(apply_button, 0, wx.ALIGN_LEFT)
        self.footer_sizer.Add(cancel_button, 0, wx.ALIGN_RIGHT)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.footer_sizer, 0, wx.EXPAND)
        self.SetSizer(self.main_sizer)

    def OnBrowse(self, event):
        with wx.DirDialog(None, 'Choose Root Folder', mgr.base_dir) as dlg:
            dlg.ShowModal()
            path = dlg.GetPath()
            self.base_dir_path.SetValue(path)

    def OnApply(self, event):
        root_path = self.base_dir_path.GetValue()
        mgr.base_dir = root_path
        self.Destroy()

    def OnCancel(self, event):
        self.Destroy()

if __name__ == '__main__':
    app = wx.PySimpleApp()
    dlg = Preferences(None)
    res = dlg.ShowModal()


