from PIL import Image
import numpy
import os

class Lfw:
  def __init__(self):
    self.filenames = []
    self.current = 0

    for filename in os.listdir('/home1/soham.pal/GSN/lfwcrop_grey/faces/'):
      self.filenames += [filename]

  def next_batch(self, size=1):
    images = numpy.zeros([size, 1024])

    for i in range(size):
        index = self.current % len(self.filenames)
        image = numpy.array(Image.open('/home1/soham.pal/GSN/lfwcrop_grey/faces/' + self.filenames[index]))[:,:,1]
        image = image * (1./image.max())
        images[i,:] += image.flatten()
        self.current = (self.current + 1) % len(self.filenames)

    return images

  def examples(self):
    return len(self.filenames)
