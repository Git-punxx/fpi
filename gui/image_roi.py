import wx
from PIL import Image
import numpy as np
import sys
from matplotlib import cm
import random
from gui.custom_events import *
from image_analysis import feature as ft


def PIL2wx(image):
    width, height = image.size
    return wx.Bitmap.FromBuffer(width, height, image.tobytes())

def wx2PIL(bitmap):
    size = tuple(bitmap.GetSize())
    try: 
        buf = size[0] * size[1] * 2 * '\x00'
        bitmap.CopyToBuffer(buf)
    except:
        del buf
        buf = bitmap.ConvertToImage().GetData()
    return Image.frombuffer('rgb', size, buf, 'raw', 'rgb', 0, 1)


def log(msg):
    print(msg)
    sys.stdout.flush()

class ImageControl(wx.Panel):
    def __init__(self, parent, image: Image, rescale = True,  *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.rescale = rescale

        self.start = None
        self.end = None

        self.buffer = None
        self.image: wx.Image = image
        self._original = None
        width, height = self.image.GetSize()
        self.SetMinSize((width, height))

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)

         
    @staticmethod
    def FromBitmap(self, bmp):
        pass

    @staticmethod
    def FromPIL(self, pil_image):
        pass

    @property
    def image_size(self):
        return self.image.GetSize()

    @staticmethod
    def fromarray(parent, nparray):
        #im = Image.fromarray(np.uint8(cm.viridis(nparray)*255))
        data = nparray.copy()
        wxim = PIL2wx(ImageControl.PIL_image_from_array(nparray))
        ctrl = ImageControl(parent = parent, image = wxim, style = wx.BORDER_RAISED)
        ctrl.data = data
        return ctrl

    @staticmethod
    def PIL_image_from_array(nparray, normalize = True, cm = 'viridis'):
        if normalize:
            #https: // stackoverflow.com / questions / 1735025 / how - to - normalize - a - numpy - array - to - within - a - certain - range
            nparray -= nparray.min()
            nparray /= nparray.max()
            nparray *= 255
        im = Image.fromarray(np.uint8(nparray)).convert('RGB')
        return  im

    def InitBuffer(self):
        '''
        Initialize the buffer
        We should get the client size and create a bitmap
        Then using a buffereddc we should draw the image on the bitmap
        '''
        w, h = self.GetClientSize()
        scaled = self.image.ConvertToImage()
        # Here maybe we could have preallocated a large enough bitmap
        # and use GetSubBitmap(rect)
        if self.rescale:
            bmp = scaled.Rescale(w, h).ConvertToBitmap()
        else:
            bmp = scaled.ConvertToBitmap()

        self.buffer = wx.Bitmap(w, h)
        bdc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        bdc.DrawBitmap(bmp, 0, 0)

    def OnSize(self, evt):
        self.InitBuffer()
        wx.BufferedDC(wx.ClientDC(self), self.buffer)

    def OnPaint(self, evt):
        if not self.buffer:
            self.InitBuffer()
        wx.BufferedPaintDC(self, self.buffer)

    def OnRightDown(self, event):
        x, y = event.GetPosition()
        box = ft.check_hitbox(x, y)


    def OnLeftDown(self, event):
        self.InitBuffer()
        self.CaptureMouse()
        x, y = event.GetPosition()
        self.start = (x, y)
    

    def OnLeftUp(self, event):
        self.ReleaseMouse()
        try:
            x, y, w, h = self.roi_to_rect()
        except TypeError:
            pass

        #TODO Create an event about roi change

        # Generate the event
        evt = UpdatedROI(id = EVT_ROI_UPDATE, roi = self.roi_to_slice())
        wx.PostEvent(self.GetParent(), evt)

    def OnMotion(self, event):
        if event.Dragging():
            x, y = event.GetPosition()
            self.end = (x, y)
            dx = self.end[0] - self.start[0]
            dy = self.end[1] - self.start[1]
            if self.start is not None:
                dc = wx.ClientDC(self)
                if dc: 
# TODO here we init buffer in every motion event. We must keep the image buffer in tact and draw on 
# a tmp buffer and then write it on the image buffer
                    self.InitBuffer()
                    dc.SetPen(wx.Pen('yellow', 1))
                    dc.SetBrush(wx.TRANSPARENT_BRUSH)
                    dc.DrawRectangle(*self.start, dx, dy)

    def roi_to_rect(self):
        try:
            dx = abs(self.end[0] - self.start[0])
            dy = abs(self.end[1] - self.start[1])
            startx, starty = (min(self.start[0], self.end[0]), min(self.start[1], self.end[1]))
            return (startx, starty, dx, dy)
        except TypeError:
            return

    def roi_to_slice(self):
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])

        print(f'Mouse down at {self.start[0]}, {self.start[1]}')
        print(f'Mouse up at {self.end[0]}, {self.end[1]}')

        x_start = min(self.start[0], self.end[0])
        y_start = min(self.start[1], self.end[1])

        print(f'Starting at {x_start}, {y_start}')

        x_start = min(self.start[0], self.end[0])
        y_start = min(self.start[1], self.end[1])
        x_end = max(self.start[0], self.end[0])
        y_end = max(self.start[1], self.end[1])

        print(x_start, y_start, x_end, y_end)

        x_from = min(self.start[0], self.end[0])
        x_to = x_from + dx

        y_from = min(self.start[1], self.end[1])
        y_to = y_from + dy
        return (x_start, x_end, y_start, y_end)




    def range_of_interest(self):
        self.roi_to_slice()
                    
    def image_region(self):
        return self.buffer.GetSubBitmap(wx.Rect(*self.roi_to_rect())).ConvertToImage()

    def bitmap_region(self):
        return self.buffer.GetSubBitmap(wx.Rect(*self.roi_to_rect()))
            
    def set_image(self, image: Image):
        wxim = PIL2wx(image)
        self._original = self.image
        self.image = wxim
        self.InitBuffer()
        self.Refresh()
        self.Update()


    def reset_image(self):
        self.image = self._original
        self.InitBuffer()
        self.Refresh()
        self.Update()

    
class MyFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arr = create_array(600, 400)
        self.panel = ImageControl.fromarray(self, arr)


class MyApp(wx.App):
    def OnInit(self):
        f = MyFrame(None)
        f.Show()
        return True

def create_array(n, m):
    print('Creating image')
    return np.array([random.randint(0, 255) for i in range(n * m)]).reshape(n, m)

