from abc import ABCMeta, abstractmethod
import threading



class AbstractUnitOfWork(meta = ABCMeta):
    def __init__(self):
        self.new_objects = []
        self.dirty_objects = []
        self.removed_obejcts = []


    def register_new(self, obj):
        assert obj not in self.new_objects, f'Object {obj} already in new objects'
        assert obj not in self.removed_obejcts, f'Object {obj} already in removed objects'
        assert obj not in self.dirty_objects, f'Object {obj} already in dirty objects'
        self.new_objects.append(obj)

    def register_dirty(self, obj):
        assert obj not in self.removed_obejcts, f'Object {obj} already in removed objects'
        self.dirty_objects.append(obj)

    def register_removed(self, obj):
        try:
            self.new_objects.remove(obj)
            return
        except ValueError:
            if obj in self.dirty_objects:
                self.dirty_objects.remove(obj)
            else:
                if obj not in self.removed_obejcts:
                    self.removed_obejcts.append(obj)

    def register_clean(self, obj):
        pass

    def commit(self):
        self.insert_new()
        self.update_dirty()
        self.delete_removed()

    def insert_new(self):
        raise NotImplementedError

    def update_dirty(self):
        raise NotImplementedError

    def delete_removed(self):
        raise NotImplementedError


    @staticmethod
    def current_scope():
        return threading.local()

    @staticmethod
    def set_current(uow):
        AbstractUnitOfWork.current_scope.uow = uow

    @staticmethod
    def get_current():
        return AbstractUnitOfWork.current_scope().uow




