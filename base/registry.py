from abc import ABCMeta, abstractmethod
from base.domain import FPIExperiment

'''
We can have registries for different scopes (process scoped or threac scoped)
The preferred way to code a process scoped registry is to use a singleton
This works well for immutable data.
Such data can be loaded when the process starts up and never need change.

Another common kind of Registry data is thread scoped like a database connections.
In this case we may use the threading.local() as a thread specific storage.
So a request for such data results in a lookup in that map of the current thread.

For session scoped data we can also use a dictionary lookup.
If we need a session ID we can put it in the thread scoped registry when a request begins.

Registries are usually divided between the layers or by execution context.
'''

class AbstractRegistry(meta = ABCMeta):

    @staticmethod
    def get_instance():
        return AbstractRegistry()



class FPIExperimentRegistry(AbstractRegistry):
    def __init__(self):
        self._map = {}

    def get(self, experiment_key: str):
        return self._map.get(experiment_key)

    def add(self, experiment: FPIExperiment):
        self._map[experiment.key] = experiment
