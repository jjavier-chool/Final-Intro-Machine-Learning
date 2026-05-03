"""
Intro to Machine Learning Final
Encompasses the solution to Task 1.
Map1: 20x20 (keep)
Map2: 400x400 -> 40x40
Map3: 532x528 -> 50x50?
Map4: 532x528 -> 50x50?
Custom (Map5): ?
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from typing import Any
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

MAPS = [ # (name, size, target)
  ("square", 40, 39),
  ("2block", 40, 39),
  ("stars", 50, 39),
  ("rooms", 50, 39),
  ("zigzag", 40, 39),
  ("spiral", 40, 20)
]

# BMP -> binary matrix. white = 1, black = 0
def load_bmp_matrix(path: str, threshold=128):
  img = Image.open(path).convert("L")
  w, h = img.size
  if w < 40 or h < 40:
    # Make it an integer multiple of (w, h) to avoid distortions
    w = (40 // w + 1)*w
    h = (40 // h + 1)*h
    img = img.resize((w, h), resample=Image.Resampling.BOX)
  matrix = np.array(img)
  return (matrix >= threshold).astype(int)

class Map:
  def __init__(self, num: int):#, binary_map: np.ndarray, rows: int, cols: int, target: int):
    name, size, target = MAPS[num-1]
    rows = cols = size
    self.name = name
    self.num = num
    self.target = (target, target)
    binary_map = load_bmp_matrix(f"maps/map{num}.bmp")
    orig_rows, orig_cols = binary_map.shape
    abstracted = np.ones((rows, cols), dtype=int)

    for i in range(rows):
      for j in range(cols):
        r_start = i * orig_rows // rows
        r_end   = (i + 1) * orig_rows // rows

        c_start = j * orig_cols // cols
        c_end   = (j + 1) * orig_cols // cols

        block = binary_map[r_start:r_end, c_start:c_end]

        if np.any(block == 0):
          abstracted[i, j] = 0
    
    self.data = abstracted
  
  def __getitem__(self, index: Any):
    return self.data[index]

  @property
  def shape(self):
    return self.data.shape
  
  @property
  def rows(self):
    return self.data.shape[0]
  
  @property
  def cols(self):
    return self.data.shape[1]

# Testing
def main():
  for i, (name, _, _) in enumerate(MAPS, 1):
    print(f"Processing: map{i} ({name})")

    og_map = load_bmp_matrix(f"maps/map{i}.bmp")
    abstracted = Map(i)

    print(f"Original size: {og_map.shape}")
    print(f"Abstracted size: {abstracted.shape}")
    img_array = (abstracted.data * 255).astype(np.uint8)
    # mode parameter is deprecated in .fromarray PIL 15
    img = Image.fromarray(img_array)
    img.save(f"map{i}.bmp")

    plt.subplot(1, len(MAPS), i)
    plt.imshow(img, cmap='gray')
  plt.show()

  print("All maps processed successfully.")

if __name__ == "__main__":
    main()
