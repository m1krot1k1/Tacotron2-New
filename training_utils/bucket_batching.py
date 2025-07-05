import numpy as np
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    """
    Bucket batching для оптимизации обучения.
    Группирует последовательности по схожей длине.
    """
    def __init__(self, data_source, batch_size, bucket_size_multiplier=100):
        self.data_source = data_source
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.lengths = [len(item) for item in data_source]
        self.indices = list(range(len(data_source)))

    def __iter__(self):
        indices = np.random.permutation(self.indices)
        buckets = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            bucket = sorted(bucket, key=lambda x: self.lengths[x])
            buckets.append(bucket)
        batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.data_source) // self.batch_size 