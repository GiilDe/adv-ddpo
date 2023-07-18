import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = []

    def update(self, rewards):
        rewards = np.array(rewards)
        advantages = np.empty_like(rewards)
        
        self.stats.extend(rewards)

        if len(self.stats) < self.min_count:
            mean = np.mean(rewards)
            std = np.std(rewards) + 1e-6
        else:
            mean = np.mean(self.stats)
            std = np.std(self.stats) + 1e-6
        advantages = (rewards - mean) / std

        return advantages

    def get_stats(self):
        return {"mean": np.mean(self.stats), "std": np.std(self.stats), "count": len(self.stats)}
