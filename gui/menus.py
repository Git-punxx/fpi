import wx
import subprocess
from gui.dialogs import *
from app_config import config_manager as app_config




# Create a bunch of Id's to use
ID_CHECK_FOLDERS = wx.NewId()
ID_CREATE_FOLDERS = wx.NewId()
ID_CONFIG_RESPONSE_PLOT = wx.NewId()
ID_SET_DATABASE_DIR = wx.NewId()


# Event ids
ID_PLOT_MEAN_BASELINE = wx.NewId()



# Analysis ids
ID_INTRINSIC_ANALYSIS = wx.NewId()




# A dict that associates wx.Ids with functions
# We will use a decorator to populate it
command_registry = {}

def register(wx_id):
    def deco(func):
        global command_registry
        command_registry[wx_id] = func
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

intrincic_menu = [(ID_INTRINSIC_ANALYSIS, 'Intrinsic analysis')]

class FPIMenuBar(wx.MenuBar):
    def __init__(self):
        wx.MenuBar.__init__(self)
        self.setup()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def setup(self):
        global file_menu
        global edit_menu
        global options_menu
        global intrincic_menu
        self.FileMenu(file_menu)
        self.EditMenu(edit_menu)
        self.OptionsMenu(options_menu)
        self.AnalysisMenu(intrincic_menu)

    def OnMenu(self, event):
        try:
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

    def AnalysisMenu(self, items):
        analysism = wx.Menu()
        for item_id, description in items:
            analysism.Append(item_id, description)
        self.Append(analysism, 'Intrinsic Analysis')

@register(ID_SET_DATABASE_DIR)
def SetDataPath(parent):
    path = DataPathDialog(parent, 'Select experiments folder ')
    if path is not None:
        # here we should check if the directory contains the proper data structure
        # if not we should offer to create it
        app_config.base_dir = path
    else:
        ErrorDialog('Could not set requested path')

@register(ID_CREATE_FOLDERS)
def CreateFolderStructure(parent):
    app_config.create_folders()

@register(ID_INTRINSIC_ANALYSIS)
def RunIntrinsic(parent):
    print('Running intrinsic')

    subprocess.run(['python', '../intrinsic/explorer.py'])






