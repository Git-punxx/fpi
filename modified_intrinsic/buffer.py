import abc
from PIL import Image
import numpy as np

__author__ = 'remi'


class Buffer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, size=3):
        # Ensure buffer size is always odd
        if size % 2 == 0:
            size += 1
        self._size = size
        # Items will be buffered in this dictionary
        self.d_buffer = dict()
        self._filling = False
        self._current = 0

    @property
    def size(self):
        return self._size

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new_current):
        self._current = new_current
        # When changing the current position of the buffer, discard every
        # item that is too far away
        index, obj = self.items
        for ix in index:
            if ix > new_current + self.size or ix < new_current - self.size:
                self.d_buffer.pop(ix)
        # And add the current object to the buffer
        new_item = self.get_new_item(new_current)
        self.add_item(new_item, new_current)

    @property
    def filling(self):
        return self._filling

    @filling.setter
    def filling(self, new_filling):
        self._filling = new_filling

    @property
    def items(self):
        """
        Return all the buffered objects, in the correct order as well as
        their index

        Return:
        -------
        ix_obj: list
            Index of all objects, sorted
        l_obj: list
            List of all the objects
        """
        ix_obj = list(self.d_buffer.keys())
        ix_obj.sort()
        l_obj = [self.d_buffer[ix] for ix in ix_obj]

        return ix_obj, l_obj

    def __len__(self):
        return len(self.d_buffer)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if index in self.d_buffer.keys():
            return self.d_buffer[index]
        else:
            raise IndexError('{0} is not buffered'.format(index))

    def add_item(self, item, index):
        """
        Add one particular item, with a specific index to the buffer
        If the item is already buffered do not add it again

        Argument
        --------
        item: object
            Object to be buffered
        index: int
            Index of that element

        Return
        ------
        True in case of successful buffering
        False if buffer is full
        """
        if index in self.d_buffer.keys():
            return True
        elif len(self) < self._size:
            self.d_buffer.update({index: item})
            return True
        else:
            return False

    def pop_item(self, index):
        """
        Remove one particular item from the buffer
        """
        ix, obj = self.items
        if index < len(ix):
            self.d_buffer.pop(ix[index])
        else:
            raise IndexError('Buffer does not have {0} elements'.format(index))

    def pop_first(self):
        """
        Remove the first item in the buffer
        """
        self.pop_item(0)

    def pop_last(self):
        """
        Remove the last item in the buffer
        """
        self.pop_item(-1)

    def add_next(self, item, index):
        """
        Add a new item at the end of the buffer and drop one item if
        necessary (from the beginning)

        Argument:
        --------
        item: object
            Object to be buffered
        index: int
            Index of that element
        """
        if index in self.d_buffer.keys():
            return
        if len(self) == self.size:
            self.pop_first()
        self.add_item(item, index)

    def add_prev(self, item, index):
        """
        Add a new item at the beginning of the buffer and drop one item if
        necessary (from the end)

        Argument:
        --------
        item: object
            Object to be buffered
        index: int
            Index of that element
        """
        if index in self.d_buffer.keys():
            return
        if len(self) == self._size:
            self.pop_last()
        self.add_item(item, index)

    def stop_filling(self, signum, frame):
        """
        In response to a signal, stops filling the buffer

        Argument
        --------
        signum
        frame

        Note:
        -----
        See module :py:module:`signal` for details about the function signature
        """
        self._filling = False

    def start_filling(self, signum, frame):
        self._filling = True
        self.fill()

    @abc.abstractmethod
    def fill(self):
        return 'Should be implemented in the concrete class'

    @abc.abstractmethod
    def get_new_item(self,index):
        return 'Should be implemented in the concrete class'


class ImBuffer(Buffer):
    def __init__(self, size, stack):
        """
        Argument
        --------
        size: int
            Number of elements the buffer can hold
        stack: Stack
            Stack object containing the paths to all images
        """
        super(ImBuffer, self).__init__(size)
        self.stack = stack
        self.n_im = len(stack)
        first_img = self.get_new_item(0)
        self.add_item(first_img, 0)

    def __getitem__(self, index):
        try:
            return super(ImBuffer, self).__getitem__(index)
        except IndexError:
            return self.stack[index]

    def get_new_item(self, index):
        """
        Get a new picture to put in the buffer
        """
        if index not in self.d_buffer.keys():
            if 0 <= index < self.n_im:
                return self.stack[index]
            # Does not raise an error in case of wrong index for easier
            # handling upstream
            else:
                return None
        else:
            return self.d_buffer[index]

    def fill(self):
        if self.filling:
            if len(self) >= self.size or len(self) >= len(self.stack):
                self.filling = False
            else:
                ix, obj = self.items
                next_ix = max(ix) + 1
                prev_ix = min(ix) - 1
                next_img = self.get_new_item(next_ix)
                prev_img = self.get_new_item(prev_ix)
                if next_img is not None:
                    self.add_next(next_img, next_ix)
                if prev_img is not None:
                    self.add_prev(prev_img, prev_ix)
                self.fill()
        else:
            return
