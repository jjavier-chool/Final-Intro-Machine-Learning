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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# BMP -> binary matrix. white = 1, black = 0
def load_bmp_matrix(path: str, threshold=128):
  img = Image.open(path).convert("L")
  matrix = np.array(img)
  return (matrix >= threshold).astype(int)

# Build abstraction via pooling. Any black pixels -> black (= 0)
def abstract_map(binary_map: np.ndarray, rows: int, cols: int):
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

  return abstracted

# Retrieve the specific map asked for
def get_abstract_map(num: int, rows: int=40, cols: int=40):
  path = f"maps/map{num}.bmp"
  og_map = load_bmp_matrix(path)
  return abstract_map(og_map, rows, cols)

# Testing
def main():
  rows = [20, 40, 50, 50]
  cols = [20, 40, 50, 50]

  for i, (row, col) in enumerate(zip(rows, cols), 1):
    print(f"Processing: map{i}")

    og_map = load_bmp_matrix(f"maps/map{i}.bmp")
    abstracted = abstract_map(og_map, row, col)

    print(f"Original size: {og_map.shape}")
    print(f"Abstracted size: {abstracted.shape}")
    img_array = (abstracted * 255).astype(np.uint8)
    # mode parameter is deprecated in .fromarray PIL 15
    img = Image.fromarray(img_array)
    img.save(f"map{i}.bmp")

    plt.subplot(1, 4, i)
    plt.imshow(img, cmap='gray')
  plt.show()

  print("All maps processed successfully.")

if __name__ == "__main__":
    main()
