import wx
import h5py
from PIL import Image
import numpy as np


class ImagePanel(wx.Dialog):
    def __init__(self, parent, path, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        self._path = path
        self._datastore_structure()
        self.image = None
        self.image_panel = wx.Panel(self)
        self.load_image()

        self.bitmap = wx.StaticBitmap(self.image_panel, -1, wx.Bitmap(self.image))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.bitmap, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

    def _datastore_structure(self):
        with h5py.File(self._path, 'r') as datastore:
            print(list(datastore.keys()))
            for key, items in datastore['trials'].items():
                print(key)
                print(items)

    def load_image(self):
        with h5py.File(self._path, 'r') as datastore:
            image = Image.fromarray(datastore['anat'][()])
            print(np.asarray(image))
            self.image = wx.Image(*image.size)
            self.image.SetData(image.convert('RGB').tobytes())

