import wx
from wx.adv import SplashScreen

class Splash(SplashScreen):
    def __init__(self, image, parent = None):
        image = wx.Bitmap(name = image, type = wx.BITMAP_TYPE_JPEG)
        splash = wx.adv.SPLASH_CENTER_ON_SCREEN | wx.adv.SPLASH_TIMEOUT
        duration = 3000
        super().__init__(bitmap = image,
                         splashStyle=splash,
                         milliseconds=duration,
                         parent = None,
                         id = wx.NewId()
                         )

