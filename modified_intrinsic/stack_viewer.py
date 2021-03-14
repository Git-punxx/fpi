import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from os.path import basename
from collections import Counter
from buffer import ImBuffer
from stack import Stack


class Viewer(object):
    def __init__(self, path, pattern='', n_thumb=5, fig=None, grid=None, buffer_size=7, **kwargs):
        """

        Parameters
        ----------
        path: str
            Path to the stack of images
        pattern: str
            Pattern to use to select the files (e.g. extension)
            If nothing is given, the most common extension in the directory is used
            Default: ''
        n_thumb: int
            Number of thumbnails to show in the navigation bar
        buffer_size: int
            Number of images in the buffer, which are preloaded for faster viewing
            Default: 7

        """
        self._path = path
        folder = basename(path)
        self._buffer_size = buffer_size
        self._c_ix = 0
        # If not pattern given, use the most common extension in the given folder
        if pattern == '':
            lst_files = os.listdir(path)
            ext = Counter([os.path.splitext(fn)[1] for fn in lst_files])
            most_common = ext.most_common(1)[0][0]
            pattern = '*' + most_common
        # Stack initialization
        self._stack = Stack(path, pattern)
        n_im = len(self.stack)
        self.n_thumb = min(n_im, n_thumb)
        self.title = kwargs.get('title', 'Frame {} / ' + '{}'.format(n_im))
        # Buffer initialization
        self._buffer = ImBuffer(buffer_size, self.stack)
        self.buffer.filling = True
        self.buffer.fill()
        # Load the first image
        self._c_im = self.buffer[self.c_ix]
        # Figure initialization
        self._thumb_ix = range(0, n_im, max(1, int(n_im/self.n_thumb)))
        if fig is None:
            self.fig = plt.figure(num=folder, figsize=(8, 8))
        else:
            self.fig = fig
        if grid is None:
            gs = gridspec.GridSpec(4, self.n_thumb)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(4, self.n_thumb, subplot_spec=grid)
        self.gs = gs
        self.ax_im = plt.subplot(gs[:3, :])
        # Thumbnails
        self.ax_min = [plt.subplot(gs[3, i]) for i in range(self.n_thumb)]
        _ = [ax.set_axis_off() for ax in self.ax_min]
        _ = [ax.imshow(self.stack[i], interpolation='none', **kwargs)
             for i, ax in zip(self._thumb_ix, self.ax_min)]
        # Add a marker to the thumbnails axes
        xlim, ylim = self.ax_min[0].get_xlim(), self.ax_min[0].get_ylim()
        self._rects = [Rectangle((0, 0), max(xlim), max(ylim),
                                 facecolor='none', edgecolor='orange', linewidth=5,
                                 visible=False)
                       for _ in self.ax_min]
        _ = [ax.add_patch(r) for ax, r in zip(self.ax_min, self._rects)]
        self._c_rect = self._rects[0]
        self._pic = self.ax_im.imshow(self.c_im, aspect=1, interpolation='none', **kwargs)
        self._cid_press = None
        self._cid_mouse = None
        self._cid_scroll = None
        self._cids = [self._cid_press, self._cid_mouse, self._cid_scroll]
        # Connect the callbacks for figure events
        self.connect()
        self.update_pic()
        plt.ion()
        self.fig.subplots_adjust(wspace=0)
        self.fig.show()

    def update_pic(self):
        """Update the currently displayed image"""
        self.pic.set_data(self.c_im)
        self.ax_im.set_title(self.title.format(self.c_ix+1))
        # To which thumbnail do we belong?
        ix_thumb = self.c_ix // (len(self.stack) // self.n_thumb)
        self._c_rect.set_visible(False)
        self._c_rect = self._rects[ix_thumb]
        self._c_rect.set_visible(True)
        self.fig.canvas.draw()

    def connect(self):
        """Connect the callbacks functions to handle figure events"""
        self._cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._cid_mouse = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self._cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        """
        Callback function for when a the mouse wheel is scrolled

        Parameters
        ----------
        event: `Matplotlib` `on_scroll` event
        """
        if event.button == 'up':
            self.c_ix = int(min(self.c_ix + event.step, len(self.stack)))
        elif event.button == 'down':
            self.c_ix = int(max(self.c_ix + event.step, 0))

    def on_click(self, event):
        """
        Callback function for when a mouse button is pressed

        Parameters
        ----------
        event: `Matplotlib` `button_press_event` event
        """
        if event.button == 1:
            # Get the axis the event occured in
            ax = event.inaxes
            try:
                # One of the thumbnail was clicked
                ix = self.ax_min.index(ax)
                # Set the current figure to the index of this thumbnail
                self.c_ix = self._thumb_ix[ix]
            except ValueError:
                # Not an axis we are interested in
                pass

    def on_press(self, event):
        """
        Callback function for when a key is pressed

        Parameters
        ----------
        event: `Matplotlib` `on_press` event
        """
        self.buffer.filling = False
        # Increase/decrease current index with right/left arrow
        if event.key == 'right':
            self.c_ix += 1
            self.buffer.add_next(self.c_im, self.c_ix)
        elif event.key == 'left':
            self.c_ix -= 1
            self.buffer.add_prev(self.c_im, self.c_ix)
        elif event.key == 'pageup':
            self.c_ix = max(self.c_ix - 5, 0)
        elif event.key == 'pagedown':
            self.c_ix = min(self.c_ix + 5, len(self.stack)-1)
        elif event.key == 'w':
            with open('saved.txt', 'a') as f:
                f.write(self.stack.images[self.c_ix])
                f.write('\n')
        elif event.key == 'q':
            # Quit properly
            self.close()

    def close(self):
        """
        Release the callbacks and close the figure
        """
        [self.fig.canvas.mpl_disconnect(cid) for cid in self._cids]
        plt.close(self.fig)

    @property
    def pic(self):
        """Reference to the AxesImage object"""
        return self._pic

    @property
    def c_im(self):
        """Image currently displayed"""
        return self._c_im

    @c_im.setter
    def c_im(self, value):
        self._c_im = value
        self.update_pic()

    @property
    def buffer(self):
        """Reference to the buffer object used"""
        return self._buffer

    @property
    def stack(self):
        """Stack object, listing all images"""
        return self._stack

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def c_ix(self):
        """Index of the image currently displayed"""
        return self._c_ix

    @c_ix.setter
    def c_ix(self, value):
        if 0 <= value < len(self.stack):
            self._c_ix = value
            self.c_im = self.buffer[value]
            self.buffer.current = self.c_ix
            self.buffer.filling = True
            self.buffer.fill()


if __name__ == '__main__':
    PATH = "/media/remi/owncloud/suminagashi/171101/pinceau_encre_SDS0.001pr_14cm_1"
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, wspace=.1)
    left = gs[:, 0]
    ax = plt.subplot(left)
    ax.plot(range(12))
    sv = Viewer(PATH, n_thumb=10, cmap=plt.cm.Greys_r)
