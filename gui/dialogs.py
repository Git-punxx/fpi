import wx

def DataPathDialog(parent, msg):
    with wx.DirDialog(parent, msg) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            return path

def ErrorDialog(msg):
    result = wx.MessageBox(msg, 'Error', wx.ICON_ERROR | wx.OK)