import glob
import re
from os.path import join, basename
import numpy as np
from PIL import Image

try:
    from itertools import izip_longest
except ImportError:
    from itertools import zip_longest as izip_longest


class Stack(object):
    """
    Define a stack image to handle long series of images
    """
    def __init__(self, path, pattern):
        """
        Initialize an image stack

        Parameters
        ----------
        path: str
            Path to the serie of pictures
        pattern: str
            Pattern to select the images among the files in the folder
        """
        self._path = path
        self._pattern = pattern
        # List of the paths to all the stack images
        self.images = []
        self.get_im_list()

    @property
    def path(self):
        return self._path

    @property
    def pattern(self):
        return self._pattern

    def get_im_list(self):
        """ Make a list of all pictures"""
        l_path = np.array(glob.glob(join(self.path, self.pattern)))
        l_filenames = [basename(p) for p in l_path]

        regex = re.compile('([0-9]*)')
        try:
            index = [int(''.join(regex.findall(flnm)))
                     for flnm in l_filenames]
        except ValueError:
            print("No number detected in one file")
        # get the indexes of the last number in the file name
        index = np.argsort(index)
        self.images = l_path[index]

    @staticmethod
    def get_pic(im):
        """
        Load an image

        Parameters
        ----------
        im: str
            Path to the pictures

        Returns
        -------
        pic: Numpy `ndarray`
            Image
        """
        with Image.open(im) as img:
            pic = np.asarray(img)
        return pic

    def next(self):
        for im in self.images:
            yield self.get_pic(im)

    def __iter__(self):
        return self.next()

    def __getitem__(self, item):
        # Help get some iteration capabilities
        # If we want a slice, check start and stop values
        if isinstance(item, slice):
            if item.start is None:
                item = slice(0, item.stop, item.step)
            if item.stop is None:
                item = slice(item.start, -1, item.step)
        # if asking for a range: construct the corresponding slice
        elif isinstance(item, list) or isinstance(item, tuple):
            item = slice(item[0], item[1])
        # idem if asking for just one
        else:
            s = 1 if item >= 0 else -1
            item = slice(item, item+s)
        # Chekc we are in the range
        if item.stop > len(self.images):
            raise IndexError('Folder: {} contains only {} images. '
                             '{} is out of bounds'.format(self.path,
                                                          len(self.images),
                                                          item))
        else:
            # Load all images
            pics = [self.get_pic(x) for x in self.images[item]]
            # And return them as a 3D array
            return np.squeeze(pics)

    def __repr__(self):
        return 'Stack of {} images from folder {}'.format(len(self.images),
                                                          self.path)

    def __len__(self):
        return len(self.images)
