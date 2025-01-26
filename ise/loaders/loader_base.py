from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self, datasample_key, batch_size=32):
        self.batch_counter = 0
        self.batch_size = batch_size
        self.datasample_key = datasample_key
        self.ids = []

    @abstractmethod
    def get_ids(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def load_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
        
    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    def get_samples(self, n=10):
        samples = []
        for batch in self:
            samples.extend(batch)
            if len(samples) >= n:
                return samples[:n]
        return samples

    def __iter__(self):
        self.ids = self.get_ids()
        self.batch_counter = 0
        return self

    def __next__(self):
        start_index = self.batch_size * self.batch_counter
        end_index = min(start_index + self.batch_size, len(self.ids))

        self.batch_counter += 1

        if start_index >= len(self.ids):
            raise StopIteration
        
        return self.load_batch(self.ids[start_index:end_index])