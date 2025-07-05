
# Решение 4: Bucket batching для оптимизации
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np

class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, bucket_size_multiplier=100):
        self.data_source = data_source
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier

        # Получаем длины всех последовательностей
        self.lengths = [len(item) for item in data_source]
        self.indices = list(range(len(data_source)))

    def __iter__(self):
        # Перемешиваем индексы
        indices = np.random.permutation(self.indices)

        # Создаем buckets по длине
        buckets = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            # Сортируем bucket по длине
            bucket = sorted(bucket, key=lambda x: self.lengths[x])
            buckets.append(bucket)

        # Создаем батчи из каждого bucket
        batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Только полные батчи
                    batches.append(batch)

        # Перемешиваем батчи
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.data_source) // self.batch_size

# Использование bucket batching
def create_efficient_dataloader(dataset, batch_size):
    sampler = BucketBatchSampler(dataset, batch_size)
    collator = DynamicPaddingCollator()

    return DataLoader(
        dataset, 
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4
    )
