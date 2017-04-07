import numpy as np
import matplotlib.pyplot as plt

class SyntheticOne:
  def __init__(self):
    # Generate with script gaussian_mix.py
    self.dataset = np.load("gaussians_one.npy")
    self.current = 0
  
  def next_batch(self, size=1):
    batch = np.zeros([size, 1])

    for i in range(size):
        index = self.current % len(self.dataset)
        batch[i] = self.dataset[self.current]
        self.current = (self.current + 1) % len(self.dataset)

    return batch

  def examples(self):
    return len(self.dataset)
