import wx
from PIL import Image
import numpy as np
import sys


def PIL2wx(image):
    width, height = image.size
    return wx.BitmapFromBuffer(width, height, image.tobytes())

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

class ImageControl(wx.Window):
    def __init__(self, image_path, rescale = True,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rescale = rescale

        self.start = None
        self.end = None

        self.buffer = None
        self.image = wx.Image(image_path)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        
         
    @staticmethod
    def FromBitmap(self, bmp):
        pass

    @staticmethod
    def FromPIL(self, pil_image):
        pass

    @staticmethod
    def FromNpArray(self, nparr):
        pass
    
    def InitBuffer(self):
        '''
        Initialize the buffer
        We should get the client size and create a bitmap
        Then using a buffereddc we should draw the image on the bitmap
        '''
        w, h = self.GetClientSize()
        scaled = self.image.Copy()

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
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self, self.buffer)

    def OnLeftDown(self, event):
        self.InitBuffer()
        self.CaptureMouse()
        x, y = event.GetPosition()
        self.start = (x, y)
    

    def OnLeftUp(self, event):
        self.ReleaseMouse()
        log(self.roi_to_rect())
        
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
                    dc.SetPen(wx.Pen('yellow', 3))
                    dc.SetBrush(wx.TRANSPARENT_BRUSH)
                    dc.DrawRectangle(*self.start, dx, dy)

    def roi_to_rect(self):
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])
        startx, starty = (min(self.start[0], self.end[0]), min(self.start[1], self.end[1]))
        return (startx, starty, dx, dy)

    def range_of_interest(self):
        self.roi_to_rect()
                    
    def image_region(self):
        return self.buffer.GetSubBitmap(wx.Rect(*self.roi_to_rect())).ConvertToImage()

    def bitmap_region(self):
        return self.buffer.GetSubBitmap(wx.Rect(*self.roi_to_rect()))
            


    
    
class MyFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel = ImageControl(parent = self, image_path = 'test.png', rescale = False)


class MyApp(wx.App):
    def OnInit(self):
        f = MyFrame(None)
        f.Show()
        return True

app = MyApp()
app.MainLoop()

