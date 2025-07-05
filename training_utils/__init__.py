# Utils package for Tacotron2 training
# Содержит утилиты для динамического паддинга и батчинга

from .dynamic_padding import DynamicPaddingCollator
from .bucket_batching import BucketBatchSampler

__all__ = ['DynamicPaddingCollator', 'BucketBatchSampler'] 