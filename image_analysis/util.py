import wx
import PIL.Image as Image
import numpy as np
from scipy.optimize import linprog
from skimage.io import imread
import matplotlib.pyplot as plt
import logging


logging.basicConfig(filename = './tiff_analysis.log')


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

def wx_fromarray(nparray):
    # im = Image.fromarray(np.uint8(cm.viridis(nparray)*255))
    wxim = PIL2wx(PIL_image_from_array(nparray))
    return wxim


def PIL_image_from_array(nparray, normalize=True):
    if normalize:
        # https: // stackoverflow.com / questions / 1735025 / how - to - normalize - a - numpy - array - to - within - a - certain - range
        nparray -= nparray.min()
        nparray /= nparray.max()
        nparray *= 255
    im = Image.fromarray(np.uint8(nparray)).convert('RGB')
    return im

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def plot_triangulation(points):
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    xs = points[:, 0]
    ys = points[:, 1]
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

'''    
create a point that is definitely outside your hull. Call it Y
produce a line segment connecting your point in question (X) to the new point Y.
loop around all edge segments of your convex hull. check for each of them if the segment intersects with XY.
If the number of intersection you counted is even (including 0), X is outside the hull. Otherwise X is inside the hull.
if so occurs XY pass through one of your vertexes on the hull, or directly overlap with one of your hull's edge, move Y a little bit.
the above works for concave hull as well. You can see in below illustration (Green dot is the X point you are trying to determine. Yellow marks the intersection points.  

def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success
'''

fnmames = ['C:/Users/thor/Downloads/Archive/lo1.tif', 'C:/Users/thor/Downloads/Archive/lo2.tif', 'C:/Users/thor/Downloads/Archive/lo3.tif']

class TiffStack:
    def __init__(self, path):
        self.path = path
        self._pics = imread(path)
        logging.info(f'Tiff image shape for {self.path}: {self._pics.shape}')
        self.images = self._pics

    def shape(self):
        return self._pics.shape

    def __iter__(self):
        return self

    def __next__(self):
        for im in self._pics:
            yield im

    def __getitem__(self, item):
        return self._pics[item]

    def __len__(self):
        return len(self._pics)

    def __str__(self):
        return f'{self.path}: {self.shape()}'

if __name__ == '__main__':
    for f in fnmames:
        TiffStack(f)