from abc import abstractmethod, ABCMeta
from base.uow import AbstractUnitOfWork


class DomainObject(meta = ABCMeta):
    def __init__(self):
        self.observers = []

    @property
    def key(self):
        raise NotImplementedError

    def mark_new(self):
        AbstractUnitOfWork.get_current().register_new(self)

    def mark_dirty(self):
        AbstractUnitOfWork.get_current().register_dirty(self)

    def mark_removed(self):
        AbstractUnitOfWork.get_current().register_removed(self)

    def register(self, observer):
        # Check if the observer has a method update
        # or it is a subclass of observer
        self.observers.append(observer)

    def notify(self):
        for observer in self.observers:
            observer.update(self)

class FPIExperiment(DomainObject):
    def __init__(self):
        super().__init__()
        self._id = id

    @property
    def key(self):
        '''
        The key we will be using to identify this experiment
        It may be a unique id or the name of the experiment
        :return:
        '''
        return self._id
