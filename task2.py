"""
Intro to Machine Learning Final
Encompasses the solution to Task 2.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import numpy as np
import matplotlib.pyplot as plt
from task1 import get_abstract_map

# The Environment Class
class GridEnvironment:
  # Constructor that takes name (for plotting purposes), the abstracted map, target pos, and S1/S2
  def __init__(self, name, grid_map, target, reward_strategy="S1"):
    """
    grid_map: 2D numpy array (1 = free, 0 = obstacle)
    target: (row, col)
    reward_strategy: "S1" or "S2"
    """
    self.name = name
    self.map = grid_map
    self.rows, self.cols = grid_map.shape
    self.target = target
    self.strategy = reward_strategy

    # Action space
    self.actions = {
      "up": (-1, 0),
      "down": (1, 0),
      "left": (0, -1),
      "right": (0, 1)
    }

  # For S2, closer to target reward calculation
  def manhattan(self, s1, s2):
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

  # Agent interaction: take state and action, return next state and immediate reward
  def step(self, state, action):
    """
    state: (row, col)
    action: "up", "down", "left", "right"
    returns: next_state (row, col), reward
    """

    dr, dc = self.actions[action]
    next_state = (state[0] + dr, state[1] + dc)

    # Boundary check
    if not (0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.cols):
      return state, -100

    # Obstacle check
    if self.map[next_state] == 0:
      return state, -100

    # Reached target
    if next_state == self.target:
      return next_state, 100

    # Reward if nothing happens
    if self.strategy == "S1":
      return next_state, 0

    else:
      old_dist = self.manhattan(state, self.target)
      new_dist = self.manhattan(next_state, self.target)

      if new_dist < old_dist:
        reward = +1
      elif new_dist > old_dist:
        reward = -1
      else:
        reward = 0

      return next_state, reward

  # Plot the map abstraction and target position
  def plot(self):
    plt.imshow(self.map, cmap="gray", origin="lower")
    plt.title("Grid Environment " + str(self.name))

    # Mark target
    plt.scatter(self.target[1], self.target[0], c='red', marker='X')

    plt.gca().invert_yaxis()
    plt.savefig(str(self.name) + ".png")

# Test
def main():
  grid = get_abstract_map(2, 40, 40)
  env = GridEnvironment(2, grid, target=(39, 39), reward_strategy="S2")
  state = (38, 38)
  next_state, reward = env.step(state, "right")
  print("Next:", next_state, "Reward:", reward)
  env.plot()

if __name__ == "__main__":
  main()
