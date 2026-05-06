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
import sys

# BMP -> binary matrix. white = 1, black = 0
def load_bmp_matrix(path, threshold=128):
  img = Image.open(path).convert("L")
  w, h = img.size
  if w < 40 or h < 40:
    # Make it an integer multiple of (w, h) to avoid distortions
    w = (40 // w + 1)*w
    h = (40 // h + 1)*h
    img = img.resize((w, h), resample=Image.Resampling.BOX)
  matrix = np.array(img)
  return (matrix >= threshold).astype(int)

# Build abstraction via pooling. Any black pixels -> black (= 0)
def abstract_map(binary_map, target_rows, target_cols):
  orig_rows, orig_cols = binary_map.shape
  abstracted = np.ones((target_rows, target_cols), dtype=int)

  for i in range(target_rows):
    for j in range(target_cols):
      r_start = int(i * orig_rows / target_rows)
      r_end   = int((i + 1) * orig_rows / target_rows)

      c_start = int(j * orig_cols / target_cols)
      c_end   = int((j + 1) * orig_cols / target_cols)

      block = binary_map[r_start:r_end, c_start:c_end]

      if np.any(block == 0):
        abstracted[i, j] = 0

  return abstracted

# Retrieve the specific map asked for
def get_abstract_map(num, target_rows=40, target_cols=40):
  path = "maps/map" + str(num) + ".bmp"
  og_map = load_bmp_matrix(path)
  return abstract_map(og_map, target_rows, target_cols)

# Testing
def main():
  target_rows = [40, 40, 40, 40]
  target_cols = [40, 40, 40, 40]
  for i in range(1,5):
    print(f"Processing: map{i}")

    og_map = load_bmp_matrix("maps/map" + str(i) + ".bmp")
    abstracted = abstract_map(og_map, target_rows[i-1], target_cols[i-1])

    og_shape = og_map.shape
    new_shape = abstracted.shape

    print(f"Original size: {og_shape}")
    print(f"Abstracted size: {new_shape}")
    img_array = (abstracted * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save("map" + str(i) + ".bmp")

  print("All maps processed successfully.")


if __name__ == "__main__":
    main()
