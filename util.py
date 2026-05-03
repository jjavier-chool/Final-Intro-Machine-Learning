import numpy as np

def softmax(x: np.ndarray):
  a = np.exp(x - x.max(axis=-1, keepdims=True))
  return a/a.sum(axis=-1, keepdims=True)