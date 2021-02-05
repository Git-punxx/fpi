from abc import abstractmethod, ABCMeta

class Observer(meta = ABCMeta):

    @abstractmethod
    def update(self, obj):
        raise NotImplementedError
