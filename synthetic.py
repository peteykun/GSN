import numpy as np
import matplotlib.pyplot as plt

class Synthetic:
  def __init__(self):
    # Generate with script gaussian_mix.py
    (x,y) = np.load("gaussians.npy")
    self.dataset = np.array(zip(x,y))
    self.current = 0
  
  def next_batch(self, size=1):
    batch = np.zeros([size, 2])

    for i in range(size):
        index = self.current % len(self.dataset)
        batch[i,:] = self.dataset[self.current,:]
        self.current = (self.current + 1) % len(self.dataset)

    return batch

  def examples(self):
    return len(self.dataset)
