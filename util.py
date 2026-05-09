"""
Intro to Machine Learning Final
Utilities used by multiple tasks
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from typing import TYPE_CHECKING
import numpy as np
import torch

# Without this it's a circular dependency
if TYPE_CHECKING:
  from task2 import GridEnvironment
  from task3 import Agent

def softmax(x: np.ndarray):
  e = np.exp(x - x.max())
  return e/e.sum(axis=0)

# Helper: accuracy + extra path metrics
#TODO: length kinda pointless? remove?
def accuracy_path(env: 'GridEnvironment', agent: 'Agent', max_steps: int=500):
  rows, cols = env.rows, env.cols

  valid = 0
  total = 0
  path_lengths = []

  for r in range(rows):
    for c in range(cols):
      if env.map[r, c] == 0:
        continue

      state = (r, c)
      total += 1
      # Need this in order to prevent loops right?
      visited = set()
      steps = 0
      success = False

      while steps < max_steps:
        if state == env.target:
          success = True
          break

        if state in visited:
          break  # Loop

        visited.add(state)

        # Pretty sure it's correct to do the best action here
        action = agent.best_action(state)
        next_state, _ = env.step(state, action)

        if next_state == state:
          break  # Invalid to just stand in place

        state = next_state
        steps += 1

      if success:
        valid += 1
        path_lengths.append(steps)

  accuracy = valid / total if total else 0
  avg_len = np.mean(path_lengths) if path_lengths else 0
  longest = np.max(path_lengths) if path_lengths else 0

  return accuracy, avg_len, longest