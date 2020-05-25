import wx
from gui.dialogs import *
import app_config
from app_config import config

ID_CHECK_FOLDERS = wx.NewId()
ID_CREATE_FOLDERS = wx.NewId()
ID_CONFIG_RESPONSE_PLOT = wx.NewId()
ID_SET_DATABASE_DIR = wx.NewId()

# A dict that associates wx.Ids with functions
# We will use a decorator to populate it
command_registry = {}

def register(wx_id):
    def deco(func):
        global command_registry
        command_registry[wx_id] = func
        print(f'{func.__name__} associated with wx id')
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return deco

file_menu = [(wx.ID_OPEN, 'Open\tCtrl+o'),
             (wx.ID_CLOSE, 'Close\tCtrl+x')
             ]
edit_menu = [(wx.ID_COPY, 'Copy\tCtrl+c'),
             (wx.ID_CUT, 'Cut\tCtrl+x'),
             (wx.ID_PASTE, 'Paste\tCtrl+v')
             ]

options_menu = [(ID_CREATE_FOLDERS, 'Create folder structure'),
                (ID_CHECK_FOLDERS, 'Check folder structure'),
                (ID_CONFIG_RESPONSE_PLOT, 'Configure response plot'),
                (ID_SET_DATABASE_DIR, 'Set experiments folder')
                 ]

class FPIMenuBar(wx.MenuBar):
    def __init__(self):
        wx.MenuBar.__init__(self)
        self.setup()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def setup(self):
        global file_menu
        global edit_menu
        global options_menu
        self.FileMenu(file_menu)
        self.EditMenu(edit_menu)
        self.OptionsMenu(options_menu)

    def OnMenu(self, event):
        try:
            print(event.GetId())
            command_registry[event.GetId()](self)
        except KeyError as e:
            ErrorDialog('This action is not yet implemented')
            print(f'Error {e}')

    def FileMenu(self, items):
        filem = wx.Menu()
        for item_id, description in items:
            filem.Append(item_id, description)
        self.Append(filem, '&File')

    def EditMenu(self, items):
        editm = wx.Menu()
        for item_id, description in items:
            editm.Append(item_id, description)
        self.Append(editm, '&Edit')

    def OptionsMenu(self, items):
        optionsm = wx.Menu()
        for item_id, description in items:
            optionsm.Append(item_id, description)
        self.Append(optionsm, 'Options')

@register(ID_SET_DATABASE_DIR)
def SetDataPath(parent):
    path = DataPathDialog(parent, 'Select experiments folder ')
    if path is not None:
        # here we should check if the directory contains the proper data structure
        # if not we should offer to create it
        app_config.set_base_dir(path)
        top = parent.GetTopLevelParent()
        status = top.GetStatusBar()
        status.SetStatusText(f'Path set to {path}')
    else:
        ErrorDialog('Could not set requested path')

@register(ID_CREATE_FOLDERS)
def CreateFolderStructure(parent):
    folders = config['Categories'].keys()
    for folder in folders:
        val = config['Categories'][folder]
        print(val)
    print(folders)

print(command_registry)


