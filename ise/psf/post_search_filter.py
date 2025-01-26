from abc import ABC, abstractmethod


class PostSearchFilter(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter(self, sample, search_results):
        pass

    def __call__(self, sample, search_results):
        return self.filter(sample, search_results)