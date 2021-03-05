import wx

class PopupMenuMixin:
    def __init__(self):
        self._menu = None

        self.Bind(wx.EVT_CONTEXT_MENU, self.OnContextMenu)

    def OnContextMenu(self, event):
        if self._menu is not None:
            self._menu.Destroy()

        self._menu = wx.Menu()
        self.CreateContextMenu(self._menu)
        self.PopupMenu(self._menu)

    def CreateContextMenu(self, menu):
        raise NotImplementedError
